import models, utils
import argparse, sys
import onmt.Dataset_sub, onmt.Models, onmt.Optim, onmt.Translator
import torch
import torch.nn as nn
from torch import cuda
from st_gumbel import gumbel_softmax
from torch.nn.utils import clip_grad_norm_
import copy

# what do yot want to do?
DEBUG = False
CREATE_DATA = False
DATASET = '0'                    # '0'-republican/democrat dataset, '1'-gender dataset
TRAIN_MODEL = True
EVALUATE_MODEL = False
TEST_MODEL = False

parser = argparse.ArgumentParser(description='train.py')

# Preprocess Options
parser.add_argument('-create_data_file', required=False, default=CREATE_DATA, help="indicates weather or not to create a data file")
parser.add_argument('-train_src', required=False, default=utils.DATA_PATH[DATASET]['train_src'], help="Path to the training source data")
parser.add_argument('-train_tgt', required=False, default=utils.DATA_PATH[DATASET]['train_tgt'], help="Path to the training target data")
parser.add_argument('-valid_src', required=False, default=utils.DATA_PATH[DATASET]['valid_src'], help="Path to the validation source data")
parser.add_argument('-valid_tgt', required=False, default=utils.DATA_PATH[DATASET]['valid_tgt'], help="Path to the validation target data")
parser.add_argument('-save_data', required=False, default=utils.DATA_PATH[DATASET]['dataset_name'], help="Output file for the prepared data")
parser.add_argument('-src_vocab_size', type=int, default=100000, help="Size of the source vocabulary")
parser.add_argument('-tgt_vocab_size', type=int, default=100000, help="Size of the target vocabulary")
parser.add_argument('-src_vocab', help="Path to an existing source vocabulary")
parser.add_argument('-tgt_vocab', help="Path to an existing target vocabulary")
parser.add_argument('-seq_length', type=int, default=50, help="Maximum sequence length")
parser.add_argument('-shuffle',    type=int, default=1, help="Shuffle data")
parser.add_argument('-seed',       type=int, default=3435, help="Random seed")
parser.add_argument('-lower', action='store_true', help='lowercase data')
parser.add_argument('-report_every', type=int, default=20000, help="Report status every this many sentences")
## Data options
parser.add_argument('-data', required=False, default='./synthetic_ident1.train.pt', help='Path to the *-train.pt file from preprocess.py')
parser.add_argument('-save_model', default='model', help="""Model filename (the model will be saved as <save_model>_epochN_PPL.pt where PPL is the validation perplexity""")
parser.add_argument('-train_from_state_dict', default='', type=str, help="""If training from a checkpoint then this is the path to the pretrained model's state_dict.""")
parser.add_argument('-classifier_model', default='./models/classifier/political_classifier/political_classifier.pt', type=str, help="""If training from a classifier then this is the path to the pretrained model's state_dict.""")
parser.add_argument('-train_from', default='', type=str, help="""If training from a checkpoint then this is the path to the pretrained model.""")
parser.add_argument('-encoder_model', required=False, default='./models/translation/french_english/french_english.pt', help='Path to the pretrained encoder model.')
parser.add_argument('-tgt_label', default=0, type=int, help="""Specify the target label i.e the label of the decoder you are training for OR the label you want the classifier to check.""")
## Operator-Decoder Models options
parser.add_argument('-layers', type=int, default=2, help='Number of layers in the LSTM encoder/decoder')
parser.add_argument('-hidden_size', type=int, default=250, help='Size of LSTM hidden states')
parser.add_argument('-word_vec_size', type=int, default=500, help='Word embedding sizes') # 128
parser.add_argument('-input_feed', type=int, default=1, help="""Feed the context vector at each time step as additional input (via concatenation with the word embeddings) to the decoder.""")
parser.add_argument('-brnn', action='store_true', default=True, help='Use a bidirectional encoder')
parser.add_argument('-brnn_merge', default='concat',help="""Merge action for the bidirectional hidden states: [concat|sum]""")
parser.add_argument('-sequence_length', type=int, default=100, help="""Max sequence kength for CNN. Give the one you gave while constructing the CNN!""") # 100
## Optimization options
parser.add_argument('-class_weight', type=float, default=1.0, help='weight of the classifier loss')
parser.add_argument('-nll_weight', type=float, default=1.0, help='weight of the cross entropy loss')
parser.add_argument('-temperature', type=float, default=1.0, help='temperature for softmax')
parser.add_argument('-batch_size', type=int, default=1, help='Maximum batch size')
parser.add_argument('-max_generator_batches', type=int, default=32, help="""Maximum batches of words in a sequence to run the generator on in parallel. Higher is faster, but uses more memory.""")
parser.add_argument('-epochs', type=int, default=10000, help='Number of training epochs')
parser.add_argument('-start_epoch', type=int, default=1, help='The epoch from which to start')
parser.add_argument('-param_init', type=float, default=0.1, help="""Parameters are initialized over uniform distribution with support (-param_init, param_init)""")
parser.add_argument('-optim', default='adam', help="Optimization method. [sgd|adagrad|adadelta|adam]")
parser.add_argument('-max_grad_norm', type=float, default=5, help="""If the norm of the gradient vector exceeds this, renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument('-dropout', type=float, default=0, help='Dropout probability; applied between LSTM stacks.')
parser.add_argument('-curriculum', action="store_true", help="""For this many epochs, order the minibatches based on source sequence length. Sometimes setting this to 1 will increase convergence speed.""")
parser.add_argument('-extra_shuffle', default=True, action="store_true", help="""By default only shuffle mini-batch order; when true, shuffle and re-assign mini-batches""")
#learning rate
parser.add_argument('-learning_rate', type=float, default=0.001, help="""Starting learning rate. If adagrad/adadelta/adam is used, then this is the global learning rate. Recommended settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001""") # 0.001
parser.add_argument('-learning_rate_decay', type=float, default=0.5, help="""If update_learning_rate, decay learning rate by this much if (i) perplexity does not decrease on the validation set or (ii) epoch has gone past start_decay_at""")
parser.add_argument('-start_decay_at', type=int, default=8, help="""Start decaying every epoch after and including this epoch""")
#pretrained word vectors
parser.add_argument('-pre_word_vecs_enc', help="""If a valid path is specified, then this will oad pretrained word embeddings on the encoder side. See README for specific formatting instructions.""")
parser.add_argument('-pre_word_vecs_dec', help="""If a valid path is specified, then this will load pretrained word embeddings on the decoder side. See README for specific formatting instructions.""")
# GPU
parser.add_argument('-gpus', default=[0], nargs='+', type=int, help="Use CUDA on the listed devices.")
parser.add_argument('-log_interval', type=int, default=50, help="Print stats at this interval.")
#translation
#parser = argparse.ArgumentParser(description='translate.py')
parser.add_argument('-model', required=False, default='./models/translation/english_french/english_french.pt', help='Path to model .pt file')
parser.add_argument('-src',   required=False, default='./data/political_data/republican_only.train.en', help='Source sequence to decode (one line per sequence)')
parser.add_argument('-tgt', help='True target sequence (optional)')
parser.add_argument('-output', default='pred.txt', help="""Path to output the predictions (each line will be the decoded sequence""")
parser.add_argument('-beam_size',  type=int, default=1, help='Beam size')
parser.add_argument('-max_sent_length', type=int, default=50, help='Maximum sentence length.') # 50
parser.add_argument('-replace_unk', action="store_true", help="""Replace the generated UNK tokens with the source
                    token that had the highest attention weight. If phrase_table
                    is provided, it will lookup the identified source token and
                    give the corresponding target token. If it is not provided
                    (or the identified source token does not exist in the
                    table) then it will copy the source token""")
parser.add_argument('-verbose', action="store_true", help='Print scores and predictions for each sentence')
parser.add_argument('-n_best', type=int, default=1, help="""If verbose is set, will output the n_best decoded sentences""")
parser.add_argument('-cuda', required=False, type=bool, default=True, help="""cuda""")
parser.add_argument('-train_translator', required=False, type=bool, default=True, help="""cuda""")

args = parser.parse_args()

if torch.cuda.is_available() and not args.gpus:
    print("WARNING: You have a CUDA device, so you should probably run with -gpus 0")

if args.gpus:
    cuda.set_device(args.gpus[0])

def print_couple(dictionary, couple):
    sentence_ind = couple[0]
    sentence = ""
    subtracted_ind = couple[1]
    subtracted = ""
    for i in range(len(sentence_ind)):
        sentence = sentence + dictionary.idxToLabel[sentence_ind[i].item()] + ' '
    for i in range(len(subtracted_ind)):
        subtracted = subtracted + dictionary.idxToLabel[subtracted_ind[i].item()] + ' '
    print("Sentence: " + sentence[:-1])
    print("Subtracted sentence: " + subtracted[:-1])
    print("--------------------------------------------")
    return

def print_batch(dictionary, batch):
    print("printing batch sentences...")
    print("")
    for i in range(batch[0][0].shape[1]):
        sentence = torch.transpose(batch[0][0], 0, 1)[i][:batch[0][1][i]]
        subtracted = torch.transpose(batch[1], 0, 1)[i]
        couple = (sentence[1:], subtracted[1:])
        print_couple(dictionary, couple)
    return

def print_output(dictionary, batch, output):
    #print("printing batch sentences...")
    #print("")
    for i in range(output.shape[1]):
        print("########################")
        # (1) print sent 1 and 2
        sent1ind = batch[0][i][0] # [data not indexes][number in batch][sent 1 or 2]
        sent1 = ""
        for j in range(len(sent1ind)):
            sent1 = sent1 + dictionary.idxToLabel[sent1ind[j].item()] + ' '
        sent2ind = batch[0][i][1]
        sent2 = ""
        for j in range(len(sent2ind)):
            sent2 = sent2 + dictionary.idxToLabel[sent2ind[j].item()] + ' '

        # (2) print subtracted sentence
        sub_Onehot = output[:,i,:]
        sub_ind = torch.stack([torch.argmax(sub_Onehot[i,:]) for i in range(len(sub_Onehot))])
        subbed = ""
        for j in range(len(sub_ind)):
            subbed = subbed + dictionary.idxToLabel[sub_ind[j].item()] + ' '

        # (3) print all sentences:
        print("Sentence 1: " + sent1[:-1])
        print("Sentence 2: " + sent2[:-1])
        #print("---------------------------")
        print("Subtracted: " + subbed[:-1])

    #print("########################################################################")

    return

def BCELoss():
    crit = nn.BCELoss()
    if args.gpus:
        crit.cuda()
    return crit

def back_translate_batch(batch, vocab, args, translator):
    # Back Translate sentences E->D (e->f) -> E (f->e) -> Z

    # TODO: (1) translate batch to regular english batch
    batch_sent, batch_lengths = batch
    batch_size = batch_sent.shape[0]

    sent1 = batch_sent[:, 0, :]
    sent1list = [[vocab.idxToLabel[int(token)] for token in sent] for sent in sent1]
    tgt1list = []

    sent2 = batch_sent[:, 1, :]
    sent2list = [[vocab.idxToLabel[int(token)] for token in sent] for sent in sent2]
    tgt2list = []

    # TODO: (2) translate batch to french using MT
    predBatch1, _, _ = translator.translate(sent1list, tgt1list)
    predBatch2, _, _ = translator.translate(sent2list, tgt2list)

    # TODO: (3) format batch as we want
    enc_check = torch.load(args.encoder_model, map_location=lambda storage, loc: storage)
    french_vocab = enc_check['dicts']['src']
    #Batch1_indices = [[french_vocab.labelToIdx[str(token)] for token in sent[0]] for sent in predBatch1]
    Batch1_indices = [[french_vocab.labelToIdx[str(token)] if str(token) in french_vocab.labelToIdx.keys() else 1 for token in sent[0]] for sent in predBatch1]
    for i in range(len(Batch1_indices)):
        if len(Batch1_indices[i]) > 50:
            Batch1_indices[i] = Batch1_indices[i][:50]
    Batch1_lengths = [len(sent) for sent in Batch1_indices]
    for i in range(len(Batch1_indices)):
        if len(Batch1_indices[i]) < 50:
            pad_len = 50-len(Batch1_indices[i])
            Batch1_indices[i] += [0]*pad_len
    Batch1_indices = torch.tensor(Batch1_indices)
    Batch1_indices = Batch1_indices.view(Batch1_indices.shape[0], 1, -1)

    #Batch2_indices = [[french_vocab.labelToIdx[str(token)] for token in sent[0]] for sent in predBatch2]
    Batch2_indices = [[french_vocab.labelToIdx[str(token)] if str(token) in french_vocab.labelToIdx.keys() else 1 for token in sent[0]] for sent in predBatch2]
    for i in range(len(Batch2_indices)):
        if len(Batch2_indices[i]) > 50:
            Batch2_indices[i] = Batch2_indices[i][:50]
    Batch2_lengths = [len(sent) for sent in Batch2_indices]
    for i in range(len(Batch2_indices)):
        if len(Batch2_indices[i]) < 50:
            pad_len = 50-len(Batch2_indices[i])
            Batch2_indices[i] += [0]*pad_len
    Batch2_indices = torch.tensor(Batch2_indices)
    Batch2_indices = Batch2_indices.view(Batch2_indices.shape[0],1,-1)

    BTsent12 = torch.cat([Batch1_indices,Batch2_indices], 1).cuda()
    BTlengths = [(Batch1_lengths[i], Batch2_lengths[i]) for i in range(len(Batch1_lengths))]


    BTBatch = (BTsent12, BTlengths)

    return BTBatch

def translate_1hot_output(X, vocab, args, translator):
    # Back Translate sentences E->D (e->f) -> E (f->e) -> Z

    # TODO: (1) translate batch to regular english batch
    batch_size = X.shape[1]
    sent1 = torch.transpose(X, 0, 1)
    sent1list = [[token for token in sent] for sent in sent1]
    tgt1list = []

    # TODO: (2) translate batch to french using MT
    predBatch1 = translator.one_hot_translate(sent1list, tgt1list)

    return predBatch1

def trainModel(model, trainData, validData, dataset, optim, MT_f_e_Encoder):
    #print(model)
    sys.stdout.flush()
    MT_f_e_Encoder2 = copy.copy(MT_f_e_Encoder)

    vocab = dataset['dicts']['vocab']

    all_train_losses = []
    all_valid_losses = []

    def trainEpoch(epoch, vocab):
        # shuffle
        if args.extra_shuffle and epoch > args.curriculum:
            trainData.shuffle()
        batchOrder = torch.randperm(len(trainData))

        model.train()
        MT_f_e_Encoder.train()
        MT_f_e_Encoder2.train()

        epoch_losses = []
        vocab = vocab
        args.train_translator = False
        translator = onmt.Translator.Translator(args)
        args.train_translator = True
        translator2 = onmt.Translator.Translator(args)
        for i in range(len(batchOrder)):#3-1):

            # (1) deterministic part:
            batchIdx = batchOrder[i] if epoch > args.curriculum else i
            batch = trainData[batchIdx] # batch is a tuple. (batch_data, batch_lengths)

            # TODO: delete this - check in bt batch and one hot bt batch are the same:
            if DEBUG:
                onehot_batch = torch.zeros([10, 1, 100004])
                for i, idx in enumerate(list(batch[0][0, 0, :])):
                    onehot_batch[i, 0, idx] = 1
                onehot_batch = onehot_batch.cuda()
                onehot_translated = translate_1hot_output(onehot_batch, vocab, args, translator2)
                for i in range(len(onehot_translated)):
                   print(list(onehot_translated[i][0]).index(1))

            model.zero_grad()

            with torch.no_grad():
                bt_batch = back_translate_batch(batch, vocab, args, translator)

            #with torch.no_grad():
            encStates1, context1, encStates2, context2, null_encS, null_context = MT_f_e_Encoder(bt_batch[0])

            encStates = (null_encS, encStates1, encStates2)
            contexts = (null_context, context1, context2)

            Z_1 = context1[-1, :].detach()
            Z_2 = context2[-1, :].detach()
            Z_null = null_context[-1, :].detach() #torch.stack((null_context[0][-1],)*args.batch_size)
            Zs = (Z_null, Z_1, Z_2)

            # (2) non determinictic part:
            output_1_2, output_2_1, output_1_null, output_null_1, output_2_null, output_null_2 = model(encStates, contexts, Zs)

            #del encStates, null_encS, encStates1, encStates2, context1, context2, contexts, null_context


            '''
            X_1_2 = torch.transpose(gumbel_softmax(torch.transpose(model.generator(output_1_2),0,2), 0.8), 0, 2) # .cuda
            # move this line to eval:
            print_output(vocab, batch, X_1_2)
            with torch.no_grad():
                X_1_2 = translate_1hot_output(X_1_2, vocab, args, translator2)#.cuda()
                _, context = MT_f_e_Encoder2(X_1_2)
            Z_1_2 = context[:, -1]
            del output_1_2, X_1_2, _, context
            '''

            '''
            X_2_1 = torch.transpose(gumbel_softmax(torch.transpose(model.generator(output_2_1),0,2), 0.8), 0, 2) # .cuda
            with torch.no_grad():
                X_2_1 = translate_1hot_output(X_2_1, vocab, args, translator2)#.cuda()
                _, context = MT_f_e_Encoder2(X_2_1)
            Z_2_1 = context[:, -1]
            del output_2_1, X_2_1, _, context
            '''


            X_1_null = torch.transpose(gumbel_softmax(torch.transpose(model.generator(output_1_null),0,2), 0.8), 0, 2) # .cuda
            #with torch.no_grad():
            X_1_null = translate_1hot_output(X_1_null, vocab, args, translator2)#.cuda()
            _, context = MT_f_e_Encoder2(X_1_null)
            Z_1_null = context[:, -1]
            print_output(vocab, batch, X_1_null)
            #del output_1_null, X_1_null, _, context


            '''
            X_null_1 = torch.transpose(gumbel_softmax(torch.transpose(model.generator(output_null_1),0,2), 0.8), 0, 2) # .cuda
            with torch.no_grad():
                X_null_1 = translate_1hot_output(X_null_1, vocab, args, translator2)#.cuda()
                _, context = MT_f_e_Encoder2(X_null_1)
            Z_null_1 = context[:, -1]
            del output_null_1, X_null_1, _, context
            '''

            '''
            X_2_null = torch.transpose(gumbel_softmax(torch.transpose(model.generator(output_2_null),0,2), 0.8), 0, 2) # .cuda
            #with torch.no_grad():
            X_2_null = translate_1hot_output(X_2_null, vocab, args, translator2)#.cuda()
            _, context = MT_f_e_Encoder2(X_2_null)
            Z_2_null = context[:, -1]
            del output_2_null, X_2_null, _, context
            '''

            '''
            X_null_2 = torch.transpose(gumbel_softmax(torch.transpose(model.generator(output_null_2), 0, 2), 0.8), 0, 2)  # .cuda
            #with torch.no_grad():
            X_null_2 = translate_1hot_output(X_null_2, vocab, args, translator2)  # .cuda()
            _, context = MT_f_e_Encoder2(X_null_2)
            Z_null_2 = context[:, -1]
            del output_null_2, X_null_2, _, context
            '''

            #inter_loss = Z_1 - Z_1_2 - (Z_2 - Z_2_1)
            #ae1_loss = Z_1 - Z_1_null # - Z_null
            #ae2_loss = Z_2 - Z_2_null # - Z_null
            #null1_loss = Z_null_1
            #null2_loss = Z_null_2
            diff_loss = Z_1_null - Z_1


            #loss = inter_loss + ae1_loss + ae2_loss + null1_loss + null2_loss  # [64,500]
            #loss = ae1_loss + ae2_loss #+ null2_loss + null1_loss
            loss = diff_loss

            loss = torch.pow(loss, 2)  # [64,500]
            loss = torch.mean(loss, 1)  # [64] # mean
            loss = torch.sum(loss)  # [1]


            #print("batch loss = ", loss.data)
            loss.backward()
            clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optim.step()

            epoch_losses.append(loss.item())


            '''
            report_loss += loss
            total_loss += loss
            total_num_correct += num_correct
            total_words += num_words
            if i % args.log_interval == -1 % args.log_interval:
                print(
                    "Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; closs: %6.4f; %3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed" %
                    (epoch, i + 1, len(trainData),
                     report_num_correct / report_tgt_words * 100,
                     math.exp(report_loss / args.log_interval),
                     report_closs / args.log_interval,
                     report_src_words / (time.time() - start),
                     report_tgt_words / (time.time() - start),
                     time.time() - start_time))

                sys.stdout.flush()
                report_loss = report_tgt_words = report_src_words = report_num_correct = report_closs = 0
                start = time.time()

        # total loss (float32), total_words (int64), total_num_correct (int64) , total_words (int64)
        '''
        return sum(epoch_losses) / float(len(epoch_losses))


    for epoch in range(args.start_epoch, args.epochs + 1):
        #print('')

        # (1) train for one epoch on the training set
        epoch_loss = trainEpoch(epoch, vocab)
        print("")
        print("Epoch Train Loss: ", epoch_loss)
        all_train_losses.append(epoch_loss)

        # (2) evaluate on the validation set
        #valid_loss = eval()
        #print('Epoch Validation Loss: ', valid_loss)
        #all_valid_losses.append(valid_loss)

        #  (3) update the learning rate
        #optim.updateLearningRate(valid_loss, epoch)

        '''
        model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items() if 'generator' not in k}
        generator_state_dict = model.generator.module.state_dict() if len(
            args.gpus) > 1 else model.generator.state_dict()
        #  (4) drop a checkpoint
        checkpoint = {
            'decoder': model.decoder.state_dict(),
            'generator': generator_state_dict,
            'dicts': dataset['dicts'],
            'args': args,
            'epoch': epoch,
            'optim': optim
        }
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d.pt' % (args.save_model, 100 * valid_acc, valid_ppl, epoch))
        '''
    print("All losses: ", all_train_losses)
    torch.save(all_train_losses, 'all_losses_synthetic')


def main(args):

    # (1) loading dataset from args.data
    print("Loading data from '%s'" % args.data)
    dataset = torch.load(args.data)  # args.data
    print_couple(dataset['dicts']['vocab'], (dataset['train']['sent1'][0], dataset['train']['sent2'][0]))

    dict_checkpoint = args.train_from if args.train_from else args.train_from_state_dict
    if dict_checkpoint:
        print('Loading dicts from checkpoint at %s' % dict_checkpoint)
        checkpoint = torch.load(dict_checkpoint)
        dataset['dicts'] = checkpoint['dicts']

    trainData = onmt.Dataset_sub.Dataset(dataset['train']['sent1'], dataset['train']['sent2'], args.batch_size, args.gpus)
    validData = onmt.Dataset_sub.Dataset(dataset['valid']['sent1'], dataset['valid']['sent2'], args.batch_size, args.gpus, volatile=True)

    dict = dataset['dicts']['vocab']
    print(' * vocabulary size. source = %d' % (dict.size()))
    print(' * number of training sentences. %d' % len(dataset['train']['sent1']))
    print(' * maximum batch size. %d' % args.batch_size)
    print(' * learning rate ', args.learning_rate)


    # (2) Building Subtract decoder model...
    print('Loading Encoder Model ...')
    enc_check = torch.load(args.encoder_model, map_location=lambda storage, loc: storage)
    m_opt = enc_check['opt']
    m_opt.dropout = 0
    MT_f_e_Encoder = models.Encoder(m_opt, enc_check['dicts']['src'])
    MT_f_e_Encoder.load_state_dict(enc_check['encoder'])


    print('Building Sub_Decoder, Generator and loading pretrained checkpoints...')
    SUB_Decoder = models.Decoder(args, dict)
    #Generator = nn.Sequential(nn.Linear(args.hidden_size, 300))
    Generator = nn.Sequential(
        nn.Linear(args.hidden_size, dict.size()),
        nn.LogSoftmax())

    # Continue training if exists
    if args.train_from:
        print('Loading model from checkpoint at %s' % args.train_from)
        chk_model = checkpoint['model']
        generator_state_dict = chk_model.generator.state_dict()
        model_state_dict = {k: v for k, v in chk_model.state_dict().items() if 'generator' not in k}
        model.load_state_dict(model_state_dict)
        Generator.load_state_dict(generator_state_dict)
        args.start_epoch = checkpoint['epoch'] + 1

    if args.train_from_state_dict:
        print('Loading model from checkpoint at %s' % args.train_from_state_dict)
        decoder.load_state_dict(checkpoint['decoder'])
        Generator.load_state_dict(checkpoint['generator'])
        args.start_epoch = checkpoint['epoch'] + 1

    model = models.Subtract_Decoder(SUB_Decoder)

    if len(args.gpus) >= 1:
        MT_f_e_Encoder.cuda()
        model.cuda()
        Generator.cuda()
        SUB_Decoder.cuda()
    else:
        MT_f_e_Encoder.cpu()
        model.cpu()
        Generator.cpu()
        SUB_Decoder.cpu()

    if len(args.gpus) > 1:
        MT_f_e_Encoder = nn.DataParallel(MT_f_e_Encoder, device_ids=args.gpus, dim=1)
        model = nn.DataParallel(model, device_ids=args.gpus, dim=1)
        Generator = nn.DataParallel(Generator, device_ids=args.gpus, dim=0)
        SUB_Decoder = nn.DataParallel(SUB_Decoder, device_ids=opt.gpus, dim=0)

    if not args.train_from_state_dict and not args.train_from:
        for p in model.parameters():
            p.data.uniform_(-args.param_init, args.param_init)

        SUB_Decoder.load_pretrained_vectors(args)

        optim = onmt.Optim.Optim(
            args.optim, args.learning_rate, args.max_grad_norm,
            lr_decay=args.learning_rate_decay,
            start_decay_at=args.start_decay_at
        )
    else:
        print('Loading optimizer from checkpoint:')
        optim = checkpoint['optim']
        print(optim)

    # model.encoder = MT_f_e_Encoder
    model.generator = Generator
    model.sub_decoder = SUB_Decoder

    optim.set_parameters(model.parameters())

    '''
    optim.set_parameters([{'params': model.generator.parameters()},
                          {'params': model.linear.parameters()},
                          {'params': model.sub_decoder.parameters()}])
    '''
    #optim.set_parameters(([model.generator.parameters(), model.linear.parameters(), model.sub_decoder.parameters()))

    if args.train_from or args.train_from_state_dict:
        optim.optimizer.load_state_dict(checkpoint['optim'].optimizer.state_dict())

    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)

    trainModel(model, trainData, validData, dataset, optim, MT_f_e_Encoder)

    print("TODO: complete main code...")


if __name__ == "__main__":
    main(args)
