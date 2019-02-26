import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt.modules.GlobalAttention
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch import cuda
import onmt.Constants

# *********** from Models_Decoder *****************

class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size


    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            #print(h_0[i].size())
            #print(c_0[i].size())
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)

class Encoder(nn.Module):

    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0
        self.rnn_size = opt.rnn_size // self.num_directions
        input_size = opt.word_vec_size # input size = 300

        super(Encoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=onmt.Constants.PAD)
        self.rnn = nn.LSTM(input_size, self.rnn_size,
                        num_layers=opt.layers,
                        dropout=opt.dropout,
                        bidirectional=opt.brnn)

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden=None):
        # input is subtracted sentences - [seq len X batch size X vocab size] - gumbel softmax output
        if input.shape[2]==self.word_lut.weight.shape[0]:
            emb_input = input@self.word_lut.weight # [seq len X batch size X emb size]
            context, encStates = self.rnn(torch.transpose(emb_input,0,1), hidden)
            return encStates, context

        # input is original batch with 2 sentences. [batch size X 2 sents X seq length]
        else:
            emb_input = self.word_lut(input) # [ batch size X 2 sents X seq length X emb size ]
            emb_null = self.word_lut(torch.zeros([1, emb_input.shape[2]], dtype=torch.int64).cuda())
            context1, encStates1 = self.rnn(torch.transpose(emb_input[:, 0, :, :], 0, 1), hidden)
            context2, encStates2 = self.rnn(torch.transpose(emb_input[:, 1, :, :], 0, 1), hidden)
            #self.eval()
            null_context, null_encS = self.rnn(torch.stack((torch.transpose(emb_null, 0, 1),) * input.shape[0], dim=1).squeeze(2), hidden)
            return encStates1, context1, encStates2, context2, null_encS, null_context


        #if isinstance(input, tuple):
        #    outputs = unpack(outputs)[0]

class Decoder(nn.Module):

    def __init__(self, args, dict):
        self.dict = dict
        self.layers = args.layers
        self.input_feed = args.input_feed
        input_size = args.word_vec_size
        if self.input_feed:
            input_size += args.hidden_size

        super(Decoder, self).__init__()
        self.word_lut = nn.Embedding(dict.size(),
                                     args.word_vec_size,
                                     padding_idx=onmt.Constants.PAD)
        self.rnn = StackedLSTM(args.layers, input_size, args.hidden_size, args.dropout)
        self.attn = onmt.modules.GlobalAttention.GlobalAttention(args.hidden_size)
        self.dropout = nn.Dropout(args.dropout)

        self.num_directions = 2 if args.brnn else 1
        self.hidden_size = args.hidden_size

    def load_pretrained_vectors(self, args):
        if args.pre_word_vecs_dec is not None:
            pretrained = torch.load(args.pre_word_vecs_dec)
            self.word_lut.weight.data.copy_(pretrained)

    # def forward(self, input, hidden, context, init_output):
    def forward(self, encStates, contexts, init_outputs, Zs, sent=1, sub=2):
        hidden = encStates[sent]
        context = contexts[sent]

        hidden = (torch.zeros(encStates[sent][0].shape).cuda(),torch.zeros(encStates[sent][1].shape).cuda())
        #context = torch.zeros(contexts[sub].shape).cuda()


        #emb = self.word_lut(input)

        # print(context.size())
        # n.b. you can increase performance if you compute W_ih * x for all
        # iterations in parallel, but that's only possible if
        # self.input_feed = False
        outputs = []
        output = init_outputs[sub]
        for emb_t in context.split(1):
            emb_t = emb_t.squeeze(0)
            if self.input_feed:
                emb_t = torch.cat((emb_t, output), 1)

            output, hidden = self.rnn(emb_t, hidden)
            output, attn = self.attn(output, context.transpose(0, 1)[:,:,:self.hidden_size])
            #output = self.dropout(output)
            outputs += [output]

        outputs = torch.stack(outputs)
        #return outputs, hidden, attn
        return outputs

class Subtract_Decoder(nn.Module):

    def __init__(self, sub_decoder):
        super(Subtract_Decoder, self).__init__()
        self.sub_decoder = sub_decoder
        #self.linear = nn.Linear(500, 250)

    def make_init_decoder_output(self, context):
        #batch_size = context.size(1)
        #h_size = (batch_size, self.sub_decoder.hidden_size)
        #return context.data.new(*h_size).zero_()
        #condition = self.linear(context)
        condition = ((context[0,:250]+context[0, -250:])/2).unsqueeze(0)
        return condition

    def _fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        if self.sub_decoder.num_directions == 2:
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                    .transpose(1, 2).contiguous() \
                    .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h

    def forward(self, encStates, contexts, Zs):

        init_output_null = self.make_init_decoder_output(Zs[0])
        init_output_1 = self.make_init_decoder_output(Zs[1])
        init_output_2 = self.make_init_decoder_output(Zs[2])
        init_outputs = (init_output_null, init_output_1, init_output_2)

        out_1_2 = self.sub_decoder(encStates, contexts, init_outputs, Zs, sent=1, sub=2)
        out_2_1 = self.sub_decoder(encStates, contexts, init_outputs, Zs, sent=2, sub=1)
        out_1_null = self.sub_decoder(encStates, contexts, init_outputs, Zs, sent=1, sub=0)
        out_2_null = self.sub_decoder(encStates, contexts, init_outputs, Zs, sent=2, sub=0)
        out_null_1 = self.sub_decoder(encStates, contexts, init_outputs, Zs, sent=0, sub=1)
        out_null_2 = self.sub_decoder(encStates, contexts, init_outputs, Zs, sent=0, sub=2)

        return out_1_2, out_2_1, out_1_null, out_2_null, out_null_1, out_null_2

class DecoderModel(nn.Module):

    def __init__(self, decoder):
        super(DecoderModel, self).__init__()
        self.decoder = decoder

    def make_init_decoder_output(self, context): # context [49,64,500]
        batch_size = context.size(1)
        h_size = (batch_size, self.decoder.rnn_size)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def _fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        if self.decoder.num_directions == 2:
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                    .transpose(1, 2).contiguous() \
                    .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h

    def forward(self, input, enc_hidden, context):
        # src = input[0]
        tgt = input[1][:-1]  # exclude last target from inputs
        # enc_hidden, context = self.encoder(src)
        init_output = self.make_init_decoder_output(context)

        enc_hidden = (self._fix_enc_hidden(enc_hidden[0]),
                      self._fix_enc_hidden(enc_hidden[1]))

        out, dec_hidden, _attn = self.decoder(tgt, enc_hidden, context, init_output)

        return out