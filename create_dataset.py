from __future__ import division
import onmt.Dict, onmt.Beam, onmt.CNNModels, onmt.Constants, onmt.Dataset, onmt.Models, onmt.Models_decoder, onmt.Optim, onmt.Translator, onmt.Translator_style
import codecs
import argparse, sys, time
import onmt.Dataset, onmt.Models, onmt.Optim
import torch

from pythonrouge.pythonrouge import Pythonrouge
import random
import time
import os.path

# what do yot want to do?
DATA_TYPE = '1'                    # '0'-democratic, '1'-republican, '2'-male, '3'-female
ROUGE = '1'                        # 'L' - ROUGE-L   '1'-ROUGE-1
PATHS_TO_DATA = {'0': './data/political_data/democratic_only.train.en',
                 '1': './data/political_data/republican_only.train.en',
                 '2': './data/gender_data/male_only.train.en',
                 '3': './data/gender_data/female_only.train.en'}
DATASET_NAMES = {'0': 'democratic',
                 '1': 'republican',
                 '2': 'male',
                 '3': 'female'}


# *************************************** main **************************************************

parser = argparse.ArgumentParser(description='train.py')

# Preprocess Options
parser.add_argument('-data_file', required=False, default=PATHS_TO_DATA[DATA_TYPE], help="path to data with all sentences")
parser.add_argument('-save_data', required=False, default=DATASET_NAMES[DATA_TYPE], help="Output file for the prepared data")
#parser.add_argument('-vocab_size', type=int, default=100000, help="Size of the vocabulary")
#parser.add_argument('-vocab', help="Path to an existing source vocabulary")
parser.add_argument('-model', required=False, default='./models/translation/english_french/english_french.pt', help='Path to model .pt file')
parser.add_argument('-seq_length', type=int, default=50, help="Maximum sequence length")
parser.add_argument('-shuffle',    type=int, default=0, help="Shuffle data")
parser.add_argument('-seed',       type=int, default=3435, help="Random seed")
parser.add_argument('-lower', action='store_true', help='lowercase data')
parser.add_argument('-report_every', type=int, default=20000, help="Report status every this many sentences")
## Data options
parser.add_argument('-data', required=False, default='./democratic_republican.train.pt', help='Path to the *-train.pt file from preprocess.py')
parser.add_argument('-save_model', default='model', help="""Model filename (the model will be saved as <save_model>_epochN_PPL.pt where PPL is the validation perplexity""")
parser.add_argument('-train_from_state_dict', default='', type=str, help="""If training from a checkpoint then this is the path to the pretrained model's state_dict.""")
parser.add_argument('-classifier_model', default='./models/classifier/political_classifier/political_classifier.pt', type=str, help="""If training from a classifier then this is the path to the pretrained model's state_dict.""")
parser.add_argument('-train_from', default='', type=str, help="""If training from a checkpoint then this is the path to the pretrained model.""")
parser.add_argument('-encoder_model', required=False, default='./models/translation/french_english/french_english.pt', help='Path to the pretrained encoder model.')
parser.add_argument('-tgt_label', default=0, type=int, help="""Specify the target label i.e the label of the decoder you are training for OR the label you want the classifier to check.""")
## Operator-Decoder Models options
parser.add_argument('-layers', type=int, default=2, help='Number of layers in the LSTM encoder/decoder')
parser.add_argument('-hidden_size', type=int, default=250, help='Size of LSTM hidden states')
parser.add_argument('-word_vec_size', type=int, default=128, help='Word embedding sizes')
parser.add_argument('-input_feed', type=int, default=1, help="""Feed the context vector at each time step as additional input (via concatenation with the word embeddings) to the decoder.""")
parser.add_argument('-brnn', action='store_true', default=True, help='Use a bidirectional encoder')
parser.add_argument('-brnn_merge', default='concat',help="""Merge action for the bidirectional hidden states: [concat|sum]""")
parser.add_argument('-sequence_length', type=int, default=100, help="""Max sequence kength for CNN. Give the one you gave while constructing the CNN!""")
## Optimization options
parser.add_argument('-class_weight', type=float, default=1.0, help='weight of the classifier loss')
parser.add_argument('-nll_weight', type=float, default=1.0, help='weight of the cross entropy loss')
parser.add_argument('-temperature', type=float, default=1.0, help='temperature for softmax')
parser.add_argument('-batch_size', type=int, default=64, help='Maximum batch size')
parser.add_argument('-max_generator_batches', type=int, default=32, help="""Maximum batches of words in a sequence to run the generator on in parallel. Higher is faster, but uses more memory.""")
parser.add_argument('-epochs', type=int, default=13, help='Number of training epochs')
parser.add_argument('-start_epoch', type=int, default=1, help='The epoch from which to start')
parser.add_argument('-param_init', type=float, default=0.1, help="""Parameters are initialized over uniform distribution with support (-param_init, param_init)""")
parser.add_argument('-optim', default='sgd', help="Optimization method. [sgd|adagrad|adadelta|adam]")
parser.add_argument('-max_grad_norm', type=float, default=5, help="""If the norm of the gradient vector exceeds this, renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument('-dropout', type=float, default=0.3, help='Dropout probability; applied between LSTM stacks.')
parser.add_argument('-curriculum', action="store_true", help="""For this many epochs, order the minibatches based on source sequence length. Sometimes setting this to 1 will increase convergence speed.""")
parser.add_argument('-extra_shuffle', default=True, action="store_true", help="""By default only shuffle mini-batch order; when true, shuffle and re-assign mini-batches""")
#learning rate
parser.add_argument('-learning_rate', type=float, default=1.0, help="""Starting learning rate. If adagrad/adadelta/adam is used, then this is the global learning rate. Recommended settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001""")
parser.add_argument('-learning_rate_decay', type=float, default=0.5, help="""If update_learning_rate, decay learning rate by this much if (i) perplexity does not decrease on the validation set or (ii) epoch has gone past start_decay_at""")
parser.add_argument('-start_decay_at', type=int, default=8, help="""Start decaying every epoch after and including this epoch""")
#pretrained word vectors
parser.add_argument('-pre_word_vecs_enc', help="""If a valid path is specified, then this will oad pretrained word embeddings on the encoder side. See README for specific formatting instructions.""")
parser.add_argument('-pre_word_vecs_dec', help="""If a valid path is specified, then this will load pretrained word embeddings on the decoder side. See README for specific formatting instructions.""")
# GPU
parser.add_argument('-gpus', default=[0], nargs='+', type=int, help="Use CUDA on the listed devices.")
parser.add_argument('-log_interval', type=int, default=50, help="Print stats at this interval.")

args = parser.parse_args()

if DATA_TYPE=='0' and os.path.exists('democratic_vocab'):
    args.vocab = './democratic_vocab'
if DATA_TYPE=='1' and os.path.exists('republican_vocab'):
    args.vocab = './republican_vocab'
if DATA_TYPE=='2' and os.path.exists('male_vocab'):
    args.vocab = './male_vocab'
if DATA_TYPE=='3' and os.path.exists('female_vocab'):
    args.vocab = './female_vocab'

if torch.cuda.is_available() and not args.gpus:
    print("WARNING: You have a CUDA device, so you should probably run with -gpus 0")

#if args.gpus:
#    cuda.set_device(args.gpus[0])

def print_sentence(vocab, sent):
    sentence = ""
    for idx in sent.item():
        sentence += vocab.idxToLabel[idx] + ' '

    print(sentence[:-1])

    return

def saveVocabulary(args, name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)

def load_dataset(args):
    return torch.load(args.data)

def print_couple(dictionary, couple):
    sentence_ind = couple[0]
    sentence = ""
    subtracted_ind = couple[1]
    subtracted = ""
    for i in range(len(sentence_ind)):
        sentence = sentence + dictionary[sentence_ind[i].item()] + ' '
    for i in range(len(subtracted_ind)):
        subtracted = subtracted + dictionary[subtracted_ind[i].item()] + ' '
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

def load_dataset(args):
    return torch.load(args.data)

def main(args):

    # creating vocabulary
    print("Creating dataset '%s'" % args.save_data)
    dicts = {}
    checkpoint = torch.load(args.model)
    vocab = checkpoint['dicts']['src']
    dicts['vocab'] = vocab


    # preparing data for ROUGH-L test
    print('Preparing data for ROUGE-%s test...' % ROUGE)
    train = {}

    sent1, sent2 = [], []
    sizes = []
    count, ignored = 0, 0

    print('Processing %s ...' % args.data_file)
    sent1F = codecs.open(args.data_file, "r", "utf-8")
    sent2F = codecs.open(args.data_file, "r", "utf-8")

    # TODO: list of all sentences..
    all_sentences = []

    while True:
        sline = sent1F.readline()
        tline = sent2F.readline()

        # normal end of file
        if sline == "" and tline == "":
            break

        # source or target does not have same number of lines
        if sline == "" or tline == "":
            print('WARNING: source and target do not have the same number of sentences')
            break

        sline = sline.strip()
        tline = tline.strip()

        # source and/or target are empty
        if sline == "" or tline == "":
            print('WARNING: ignoring an empty line (' + str(count + 1) + ')')
            continue

        # TODO: add code that saves sline and tline which are sentences. remember to remove the 'democratic ' starting
        clean_sline = sline[11:] # cutting the start word from the sentence
        all_sentences.append(clean_sline)

        sent1Words = sline.split()
        sent2Words = tline.split()

        if len(sent1Words) <= args.seq_length and len(sent2Words) <= args.seq_length:

            sent1 += [dicts['vocab'].convertToIdx(sent1Words, onmt.Constants.UNK_WORD)]
            sent2 += [dicts['vocab'].convertToIdx(sent2Words, onmt.Constants.UNK_WORD)]
            #sent2 += [dicts['vocab'].convertToIdx(sent2Words, onmt.Constants.UNK_WORD, onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD)]

            sizes += [len(sent1Words)]
        else:
            ignored += 1

        count += 1

        if count % args.report_every == 0:
            print('... %d sentences prepared' % count)

    sent1F.close()
    sent2F.close()

    # TODO: loop over all sentences and find for each the highest 10 ROUGE-L correlated sentences.

    if os.path.exists('%s_above_015_%s' % (args.save_data, ROUGE)):
        above_015 = torch.load('%s_above_015_%s' % (args.save_data, ROUGE))
    else:
        above_015 = []
    if os.path.exists('%s_above_02_%s' % (args.save_data, ROUGE)):
        above_02 = torch.load('%s_above_02_%s' % (args.save_data, ROUGE))
    else:
        above_02 = []
    if os.path.exists('%s_above_025_%s' % (args.save_data, ROUGE)):
        above_025 = torch.load('%s_above_025_%s' % (args.save_data, ROUGE))
    else:
        above_025 = []
    if os.path.exists('%s_above_03_%s' % (args.save_data, ROUGE)):
        above_03 = torch.load('%s_above_03_%s' % (args.save_data, ROUGE))
    else:
        above_03 = []
    if os.path.exists('%s_above_04_%s' % (args.save_data, ROUGE)):
        above_04 = torch.load('%s_above_04_%s' % (args.save_data, ROUGE))
    else:
        above_04 = []


    count = 0
    t0 = time.time()
    while True:
        t1 = time.time()
        count += 1
        index1 = random.randint(0, len(all_sentences) - 1)
        index2 = random.randint(0, len(all_sentences) - 1)
        reference = [[[all_sentences[index1]]]]
        summary = [[all_sentences[index2]]]

        rouge = Pythonrouge(summary_file_exist=False,
                            summary=summary, reference=reference,
                            n_gram=2, ROUGE_SU4=False, ROUGE_L=True,
                            recall_only=False, stemming=True, stopwords=True,
                            word_level=True, length_limit=True, length=50,
                            use_cf=False, cf=95, scoring_formula='average',
                            resampling=True, samples=1000, favor=True, p=0.5)
        score = rouge.calc_score()
        if ROUGE == 'L':
            rouge_l_1_2_score = (score['ROUGE-L-F'] + score['ROUGE-L-R']) / 2 # score['ROUGE-L']
        elif ROUGE == '1':
            rouge_l_1_2_score = (score['ROUGE-1-F'] + score['ROUGE-1-R']) / 2 # score['ROUGE-L']

        rouge = Pythonrouge(summary_file_exist=False,
                            summary=reference, reference=summary,
                            n_gram=2, ROUGE_SU4=False, ROUGE_L=True,
                            recall_only=False, stemming=True, stopwords=True,
                            word_level=True, length_limit=True, length=50,
                            use_cf=False, cf=95, scoring_formula='average',
                            resampling=True, samples=1000, favor=True, p=0.5)
        score = rouge.calc_score()
        if ROUGE == 'L':
            rouge_l_2_1_score = (score['ROUGE-L-F'] + score['ROUGE-L-R']) / 2 # score['ROUGE-L']
        elif ROUGE == '1':
            rouge_l_2_1_score = (score['ROUGE-1-F'] + score['ROUGE-1-R']) / 2 # score['ROUGE-L']

        rouge_l_score = (rouge_l_1_2_score + rouge_l_2_1_score) / 2

        if rouge_l_score > 0.15:
            above_015.append((index1, all_sentences[index1], index2, all_sentences[index2], rouge_l_score))
        if rouge_l_score > 0.2:
            above_02.append((index1, all_sentences[index1], index2, all_sentences[index2], rouge_l_score))
        if rouge_l_score > 0.25:
            above_025.append((index1, all_sentences[index1], index2, all_sentences[index2], rouge_l_score))
        if rouge_l_score > 0.3:
            above_03.append((index1, all_sentences[index1], index2, all_sentences[index2], rouge_l_score))
        if rouge_l_score > 0.4:
            above_04.append((index1, all_sentences[index1], index2, all_sentences[index2], rouge_l_score))
        t2 = time.time()

        if count % 1000 == 0:
            print("After %i tries..." % count)
            print("we have %i examples with above 0.15 score" % len(above_015))
            print("we have %i examples with above 0.2 score" % len(above_02))
            print("we have %i examples with above 0.25 score" % len(above_025))
            print("we have %i examples with above 0.3 score" % len(above_03))
            print("we have %i examples with above 0.4 score" % len(above_04))
            print("Time from beggining: ", round(t2-t0), " Time for iteration: ", round(t2-t1,3), "Number of iterations: ", count)
            if len(above_015) > 1:
                print("ROUGE-L score: ", above_015[-1][4])
                print("Sentence 1: ", all_sentences[above_015[-1][0]])
                print("Sentence 2: ", all_sentences[above_015[-1][2]])
                print("************************************************")
            if len(above_02) > 1:
                print("ROUGE-L score: ", above_02[-1][4])
                print("Sentence 1: ", all_sentences[above_02[-1][0]])
                print("Sentence 2: ", all_sentences[above_02[-1][2]])
                print("*********************************************************")
            if len(above_025) > 1:
                print("ROUGE-L score: ", above_025[-1][4])
                print("Sentence 1: ", all_sentences[above_025[-1][0]])
                print("Sentence 2: ", all_sentences[above_025[-1][2]])
                print("********************************************************************")
            if len(above_03) > 1:
                print("ROUGE-L score: ", above_03[-1][4])
                print("Sentence 1: ", all_sentences[above_03[-1][0]])
                print("Sentence 2: ", all_sentences[above_03[-1][2]])
                print("********************************************************************************")
            if len(above_04) > 1:
                print("ROUGE-L score: ", above_04[-1][4])
                print("Sentence 1: ", all_sentences[above_04[-1][0]])
                print("Sentence 2: ", all_sentences[above_04[-1][2]])
                print("******************************************************************************************")

        if count % 10000 == 0:

            lists = [above_015, above_02, above_025, above_03, above_04]
            names = ['above_015', 'above_02', 'above_025', 'above_03', 'above_04']
            for above_list, name in zip(lists,names):
                n = len(above_list)
                train_size = int(n*0.9)

                train = {}
                valid = {}

                sents1, sents2 = [], []
                sizes = []
                count, ignored = 0, 0

                # fixing 3 tuples to 5 tuples
                for i in range(len(above_list)):
                    if len(above_list[i])==3:
                        above_list[i] = (above_list[i][0], all_sentences[above_list[i][0]], above_list[i][1], all_sentences[above_list[i][1]], above_list[i][2])

                print("Saving Dataset...")
                torch.save(above_list, '%s_%s_%s' % (args.save_data, name, ROUGE))

                for pair in above_list:
                    sent1 = pair[1].strip()
                    sent2 = pair[3].strip()

                    sent1Words = sent1.split()
                    sent2Words = sent2.split()

                    if len(sent1Words) <= args.seq_length and len(sent2Words) <= args.seq_length:

                        sents1 += [vocab.convertToIdx(sent1Words, onmt.Constants.UNK_WORD)]
                        sents2 += [vocab.convertToIdx(sent2Words, onmt.Constants.UNK_WORD)]
                        #sents2 += [vocab.convertToIdx(sent2Words, onmt.Constants.UNK_WORD, onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD)]

                        sizes += [len(sent1Words)]
                    else:
                        ignored += 1

                    count += 1

                    #if count % args.report_every == 0:
                    #    print('... %d sentences prepared' % count)


                if args.shuffle == 1:
                    print('... shuffling sentences')
                    perm = torch.randperm(len(sents1))
                    sents1 = [sents1[idx] for idx in perm]
                    sents2 = [sents2[idx] for idx in perm]
                    #sizes = [sizes[idx] for idx in perm]

                    #print('... sorting sentences by size')
                    #_, perm = torch.sort(torch.Tensor(sizes))
                    #sents1 = [sents1[idx] for idx in perm]
                    #sents2 = [sents2[idx] for idx in perm]

                print('Prepared %d sentences (%d ignored due to length == 0 or > %d)' % (len(sents1), ignored, args.seq_length))

                train['sent1'], train['sent2'] = sents1[:train_size], sents2[:train_size]
                valid['sent1'], valid['sent2'] = sents1[train_size:], sents2[train_size:]

                print('Saving data to \'' + args.save_data + '_%s_%s_train.pt\'...' % (name, ROUGE))
                save_data = {'dicts': dicts,
                             'train': train,
                             'valid': valid,
                             }
                torch.save(save_data, args.save_data + '_%s_%s.train.pt' % (name, ROUGE))

        # *******************************************************

        if len(above_025) > 10000:
            break


if __name__ == "__main__":
    main(args)