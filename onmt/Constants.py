import torch

NUM_WORDS = 100004

PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

OH = torch.zeros(NUM_WORDS)#, requires_grad=True)
OH_PAD = torch.zeros(NUM_WORDS)#, requires_grad=True)
OH_UNK = torch.zeros(NUM_WORDS)#, requires_grad=True)
OH_BOS = torch.zeros(NUM_WORDS)#, requires_grad=True)
OH_EOS = torch.zeros(NUM_WORDS)#, requires_grad=True)
OH_PAD[0] = 1
OH_UNK[1] = 1
OH_BOS[2] = 1
OH_EOS[3] = 1