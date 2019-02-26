from __future__ import division
import onmt.Dict, onmt.Beam, onmt.CNNModels, onmt.Constants, onmt.Dataset, onmt.Models, onmt.Models_decoder, onmt.Optim, onmt.Translator, onmt.Translator_style
import torch
import codecs


def makeVocabulary(args, filename, size):
    vocab = onmt.Dict.Dict([onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                  onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD], lower=args.lower)

    with codecs.open(filename, "r", "utf-8") as f:
        for sent in f.readlines():
            for word in sent.split():
                vocab.add(word)

    originalSize = vocab.size()
    vocab = vocab.prune(size)
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size(), originalSize))

    return vocab


def initVocabulary(args, name, dataFile, vocabFile, vocabSize):

    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        print('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = onmt.Dict()
        vocab.loadFile(vocabFile)
        print('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        print('Building ' + name + ' vocabulary...')
        genWordVocab = makeVocabulary(args, dataFile, vocabSize)

        vocab = genWordVocab

    print()
    return vocab


def saveVocabulary(args, name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def makeData(args, srcFile, tgtFile, srcDicts, tgtDicts):
    src, tgt = [], []
    sizes = []
    count, ignored = 0, 0

    print('Processing %s & %s ...' % (srcFile, tgtFile))
    srcF = codecs.open(srcFile, "r", "utf-8")
    tgtF = codecs.open(tgtFile, "r", "utf-8")

    while True:
        sline = srcF.readline()
        tline = tgtF.readline()

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
            print('WARNING: ignoring an empty line ('+str(count+1)+')')
            continue

        srcWords = sline.split()
        tgtWords = tline.split()

        if len(srcWords) <= args.seq_length and len(tgtWords) <= args.seq_length:

            src += [srcDicts.convertToIdx(srcWords,
                                          onmt.Constants.UNK_WORD)]
            tgt += [tgtDicts.convertToIdx(tgtWords,
                                          onmt.Constants.UNK_WORD,
                                          onmt.Constants.BOS_WORD,
                                          onmt.Constants.EOS_WORD)]

            sizes += [len(srcWords)]
        else:
            ignored += 1

        count += 1

        if count % args.report_every == 0:
            print('... %d sentences prepared' % count)

    srcF.close()
    tgtF.close()

    if args.shuffle == 1:
        print('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    print('... sorting sentences by size')
    _, perm = torch.sort(torch.Tensor(sizes))
    src = [src[idx] for idx in perm]
    tgt = [tgt[idx] for idx in perm]

    print('Prepared %d sentences (%d ignored due to length == 0 or > %d)' %
          (len(src), ignored, args.seq_length))

    return src, tgt


def create_dataset_file(args):

    dicts = {}
    print('Preparing source vocab ....')
    dicts['src'] = initVocabulary(args, 'source', args.train_src, args.src_vocab,
                                  args.src_vocab_size)
    print('Preparing target vocab ....')
    dicts['tgt'] = initVocabulary(args, 'target', args.train_tgt, args.tgt_vocab,
                                  args.tgt_vocab_size)

    print('Preparing training ...')
    train = {}
    train['src'], train['tgt'] = makeData(args, args.train_src, args.train_tgt,
                                          dicts['src'], dicts['tgt'])

    print('Preparing validation ...')
    valid = {}
    valid['src'], valid['tgt'] = makeData(args, args.valid_src, args.valid_tgt,
                                    dicts['src'], dicts['tgt'])

    if args.src_vocab is None:
        saveVocabulary(args, 'source', dicts['src'], args.save_data + '.src.dict')
    if args.tgt_vocab is None:
        saveVocabulary(args, 'target', dicts['tgt'], args.save_data + '.tgt.dict')


    print('Saving data to \'' + args.save_data + '.train.pt\'...')
    save_data = {'dicts': dicts,
                 'train': train,
                 'valid': valid,
                }
    torch.save(save_data, args.save_data + '.train.pt')


def load_dataset(args):
    return torch.load(args.data)
