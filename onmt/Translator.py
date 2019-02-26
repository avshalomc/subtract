import onmt.Models_decoder, onmt.Constants, onmt.Optim, onmt.Dict, onmt.Models, onmt.Dataset
import torch.nn as nn
import torch
import time
from st_gumbel import gumbel_softmax


class Translator(object):
    def __init__(self, opt):
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch

        checkpoint = torch.load(opt.model)

        model_opt = checkpoint['opt']
        self.src_dict = checkpoint['dicts']['src']
        self.tgt_dict = checkpoint['dicts']['tgt']

        encoder = onmt.Models.Encoder(model_opt, self.src_dict)
        decoder = onmt.Models.Decoder(model_opt, self.tgt_dict)
        model = onmt.Models.NMTModel(encoder, decoder)

        generator = nn.Sequential(
            nn.Linear(model_opt.rnn_size, self.tgt_dict.size()),
            nn.LogSoftmax())

        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])

        if opt.cuda:
            model.cuda()
            generator.cuda()
        else:
            model.cpu()
            generator.cpu()

        model.generator = generator

        self.model = model
        if opt.train_translator == False:
            self.model.eval()
        elif opt.train_translator == True:
            self.model.train()

    def buildData(self, srcBatch, goldBatch):
        srcData = [self.src_dict.convertToIdx(b,
                                              onmt.Constants.UNK_WORD) for b in srcBatch]
        tgtData = None
        if goldBatch:
            tgtData = [self.tgt_dict.convertToIdx(b,
                                                  onmt.Constants.UNK_WORD,
                                                  onmt.Constants.BOS_WORD,
                                                  onmt.Constants.EOS_WORD) for b in goldBatch]

        return onmt.Dataset.Dataset(srcData, tgtData, self.opt.batch_size, self.opt.cuda, volatile=True)

    def buildTargetTokens(self, pred, src, attn):
        tokens = self.tgt_dict.convertToLabels(pred, onmt.Constants.EOS)
        tokens = tokens[:-1]  # EOS
        if self.opt.replace_unk:
            for i in range(len(tokens)):
                if tokens[i] == onmt.Constants.UNK_WORD:
                    _, maxIndex = attn[i].max(0)
                    tokens[i] = src[maxIndex[0]]
        return tokens

    def translateBatch(self, srcBatch, tgtBatch):
        batchSize = srcBatch[0].size(1)
        beamSize = self.opt.beam_size

        #  (1) run the encoder on the src
        self.model.encoder.rnn.dropout = 0
        with torch.no_grad():
            encStates, context = self.model.encoder(srcBatch)
        srcBatch = srcBatch[0] # drop the lengths needed for encoder

        rnnSize = context.size(2)
        encStates = (self.model._fix_enc_hidden(encStates[0]),
                      self.model._fix_enc_hidden(encStates[1]))

        #  This mask is applied to the attention model inside the decoder
        #  so that the attention ignores source padding
        padMask = srcBatch.data.eq(onmt.Constants.PAD).t()
        def applyContextMask(m):
            if isinstance(m, onmt.modules.GlobalAttention.GlobalAttention):
                m.applyMask(padMask)

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        goldScores = context.data.new(batchSize).zero_()
        t0 = time.time()
        if tgtBatch is not None:
            decStates = encStates
            decOut = self.model.make_init_decoder_output(context)
            self.model.decoder.apply(applyContextMask)
            initOutput = self.model.make_init_decoder_output(context)


            with torch.no_grad():
                decOut, decStates, attn = self.model.decoder(tgtBatch[:-1], decStates, context, initOutput)

                for dec_t, tgt_t in zip(decOut, tgtBatch[1:].data):
                    gen_t = self.model.generator.forward(dec_t)
                    tgt_t = tgt_t.unsqueeze(1)
                    scores = gen_t.data.gather(1, tgt_t)
                    scores.masked_fill_(tgt_t.eq(onmt.Constants.PAD), 0)
                    goldScores += scores.squeeze()
        t1 = time.time()

        #  (3) run the decoder to generate sentences, using beam search
        # Expand tensors for each beam.
        context = context.data.repeat(1, beamSize, 1)
        decStates = (encStates[0].data.repeat(1, beamSize, 1), encStates[1].data.repeat(1, beamSize, 1))
        beam = [onmt.Beam(beamSize, self.opt.cuda) for k in range(batchSize)]
        decOut = self.model.make_init_decoder_output(context)
        padMask = srcBatch.data.eq(onmt.Constants.PAD).t().unsqueeze(0).repeat(beamSize, 1, 1)
        batchIdx = list(range(batchSize))
        remainingSents = batchSize
        t2 = time.time()


        for i in range(self.opt.max_sent_length):
            #t11 = time.time()
            self.model.decoder.apply(applyContextMask)

            # Prepare decoder input.
            input = torch.stack([b.getCurrentState() for b in beam
                               if not b.done]).t().contiguous().view(1, -1)
            #t22 = time.time()
            self.model.decoder.rnn.dropout.p = 0
            self.model.decoder.dropout.p = 0
            with torch.no_grad():
                decOut, decStates, attn = self.model.decoder(input, decStates, context, decOut)
            # decOut: 1 x (beam*batch) x numWords
            decOut = decOut.squeeze(0)
            #t33 = time.time()
            with torch.no_grad():
                out = self.model.generator.forward(decOut)

            # batch x beam x numWords
            wordLk = out.view(beamSize, remainingSents, -1).transpose(0, 1).contiguous()
            attn = attn.view(beamSize, remainingSents, -1).transpose(0, 1).contiguous()
            #t44 = time.time()
            active = []
            for b in range(batchSize):
                if beam[b].done:
                    continue

                idx = batchIdx[b]
                if not beam[b].advance(wordLk.data[idx], attn.data[idx]):
                    active += [b]

                for decState in decStates:  # iterate over h, c
                    # layers x beam*sent x dim
                    sentStates = decState.view(-1, beamSize, remainingSents, decState.size(2))[:, :, idx]
                    sentStates.data.copy_(sentStates.data.index_select(1, beam[b].getCurrentOrigin()))
            #t55 = time.time()
            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            activeIdx = self.tt.LongTensor([batchIdx[k] for k in active])
            #t551 = time.time()
            batchIdx = {beam: idx for idx, beam in enumerate(active)}
            #t552 = time.time()

            def updateActive(t):
                # select only the remaining active sentences
                view = t.data.view(-1, remainingSents, rnnSize)
                newSize = list(t.size())
                newSize[-2] = newSize[-2] * len(activeIdx) // remainingSents
                return view.index_select(1, activeIdx).view(*newSize)

            #t66 = time.time()

            decStates = (updateActive(decStates[0]), updateActive(decStates[1]))
            decOut = updateActive(decOut)
            context = updateActive(context)
            padMask = padMask.index_select(1, activeIdx)

            remainingSents = len(active)
            #t77 = time.time()
            # print("1-",t55-t44,"2-",t551-t55,"3-",t552-t551,"4-",t66-t552)#,"4-",t44-t33,"5-",t55-t44,"6-",t66-t55,"7-",t77-t66)
        t3 = time.time()



        #  (4) package everything up

        allHyp, allScores, allAttn = [], [], []
        n_best = self.opt.n_best
        t4 = time.time()

        for b in range(batchSize):
            scores, ks = beam[b].sortBest()

            allScores += [scores[:n_best]]
            valid_attn = srcBatch.data[:, b].ne(onmt.Constants.PAD).nonzero().squeeze(1)
            hyps, attn = zip(*[beam[b].getHyp(k) for k in ks[:n_best]])
            attn = [a.index_select(1, valid_attn) for a in attn]
            allHyp += [hyps]
            allAttn += [attn]
        t5 = time.time()

        #print("Translation time: part 1 - ",t1-t0, "part 2 - ", t2-t1, "part 3 - ", t3-t2, "part 4 - ", t4-t3, "part 5 - ", t5-t4)

        return allHyp, allScores, allAttn, goldScores

    def one_hot_translateBatch(self, srcBatch, tgtBatch):
        batchSize = len(srcBatch)
        beamSize = self.opt.beam_size

        #  (1) run the encoder on the src
        self.model.encoder.rnn.dropout = 0
        encStates, context = self.model.encoder(srcBatch)

        rnnSize = context.size(2)
        encStates = (self.model._fix_enc_hidden(encStates[0]),
                      self.model._fix_enc_hidden(encStates[1]))

        #  This mask is applied to the attention model inside the decoder
        #  so that the attention ignores source padding
        #padMask = srcBatch.data.eq(onmt.Constants.PAD).t()
        padMask = torch.stack([torch.stack([token[0] for token in sent]) for sent in srcBatch])
        def applyContextMask(m):
            if isinstance(m, onmt.modules.GlobalAttention.GlobalAttention):
                m.applyMask(padMask)

        # (3) run the decoder to generate sentences, using beam search
        # Expand tensors for each beam.
        context = context.repeat(1, beamSize, 1)
        #decStates = (encStates[0].repeat(1, beamSize, 1), encStates[1].repeat(1, beamSize, 1))
        decStates = encStates
        beam = [onmt.Beam(beamSize, self.opt.cuda) for k in range(batchSize)]
        decOut = self.model.make_init_decoder_output(context)
        #padMask = srcBatch.data.eq(onmt.Constants.PAD).t().unsqueeze(0).repeat(beamSize, 1, 1)
        padMask = torch.stack(tuple(torch.stack(tuple(torch.ByteTensor([token[0]]) for token in sent)) for sent in srcBatch)).squeeze(2).unsqueeze(0).to(decOut.device)
        #padMask = torch.tensor(torch.stack(tuple(torch.stack(tuple(torch.FloatTensor([token[0]]) for token in sent)) for sent in srcBatch)).squeeze(2).unsqueeze(0).repeat(beamSize, 1, 1), requires_grad=True)
        batchIdx = list(range(batchSize))
        remainingSents = batchSize

        outputs = []
        input = onmt.Constants.OH_BOS.unsqueeze(0).unsqueeze(0).repeat(1, batchSize, 1).to(decOut.device)
        for i in range(self.opt.max_sent_length):
            if i!=0:
                input = outputs[-1].unsqueeze(0)
            # checks if the prediction is of an BOS token
            if input[0][0][3].item()==1.:
                outputs = outputs[:-1]
                break

            self.model.decoder.apply(applyContextMask)

            # Prepare decoder input.
            # input = torch.stack([b.onehot_getCurrentState() for b in beam if not b.done])
            self.model.decoder.rnn.dropout.p = 0
            self.model.decoder.dropout.p = 0
            #with torch.no_grad():
            decOut, decStates, attn = self.model.decoder(input, decStates, context, decOut)
            # decOut: 1 x (beam*batch) x numWords
            decOut = decOut.squeeze(0)
            #with torch.no_grad():
            out = self.model.generator.forward(decOut)
            '''
            # batch x beam x numWords
            wordLk = out.view(beamSize, remainingSents, -1).transpose(0, 1).contiguous()
            attn = attn.view(beamSize, remainingSents, -1).transpose(0, 1).contiguous()
            active = []
            for b in range(batchSize):
                if beam[b].done:
                    continue

                idx = batchIdx[b]
                if not beam[b].onehot_advance(wordLk[idx], attn[idx]):
                    active += [b]

                for decState in decStates:  # iterate over h, c
                    # layers x beam*sent x dim
                    sentStates = decState.view(-1, beamSize, remainingSents, decState.size(2))[:, :, idx]
                    sentStates.copy_(sentStates.data.index_select(1, beam[b].onehot_getCurrentOrigin()))
            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            activeIdx = self.tt.LongTensor([batchIdx[k] for k in active])
            batchIdx = {beam: idx for idx, beam in enumerate(active)}

            def updateActive(t):
                # select only the remaining active sentences
                view = t.view(-1, remainingSents, rnnSize)
                newSize = list(t.size())
                newSize[-2] = newSize[-2] * len(activeIdx) // remainingSents
                return view.index_select(1, activeIdx).view(*newSize)

            decStates = (updateActive(decStates[0]), updateActive(decStates[1]))
            decOut = updateActive(decOut)
            context = updateActive(context)
            #padMask = padMask.index_select(1, activeIdx)

            #remainingSents = len(active)
            '''
            gumbel_pred = gumbel_softmax(out, 0.0001)
            #print(list(gumbel_pred[0]).index(1))
            outputs.append(gumbel_pred)


        gumbeled_output = torch.stack(outputs).cuda()
        print(len(gumbeled_output))

        return gumbeled_output

    def translate(self, srcBatch, goldBatch):
        #t0 = time.time()
        #  (1) convert words to indexes
        dataset = self.buildData(srcBatch, goldBatch)
        src, tgt, indices = dataset[0]
        #t1 = time.time()

        #  (2) translate
        pred, predScore, attn, goldScore = self.translateBatch(src, tgt)
        pred, predScore, attn, goldScore = list(zip(*sorted(zip(pred, predScore, attn, goldScore, indices), key=lambda x: x[-1])))[:-1]
        #t2 = time.time()

        #  (3) convert indexes to words
        predBatch = []
        for b in range(src[0].size(1)):
            predBatch.append(
                [self.buildTargetTokens(pred[b][n], srcBatch[b], attn[b][n])
                        for n in range(self.opt.n_best)]
            )
        #t3 = time.time()

        #print("Translation time: part 1 - ",t1-t0, "part 2 - ", t2-t1, "part 3 - ", t3-t2)

        return predBatch, predScore, goldScore

    def one_hot_translate(self, srcBatch, goldBatch):

        #  (1) convert words to indexes
        src = srcBatch
        tgt = goldBatch
        indices = tuple(range(len(srcBatch)))

        #  (2) translate
        #pred, predScore, attn = self.one_hot_translateBatch(src, tgt)
        pred = self.one_hot_translateBatch(src, tgt)
        #pred, predScore, attn = list(zip(*sorted(zip(pred, predScore, attn, indices), key=lambda x: x[-1])))[:-1]

        #pred = torch.stack([torch.stack([pred[i][0][j] for j in range(len(pred[i][0]))]) for i in range(len(pred))])

        return pred