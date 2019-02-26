
DATA_PATH = {'0' : {'train_src':'./data/political_data/democratic_only.train.en',
                    'train_tgt': './data/political_data/republican_only.train.en',
                    'valid_src': './data/political_data/democratic_only.dev.en',
                    'valid_tgt': './data/political_data/republican_only.dev.en',
                    'dataset_name' : 'democratic_republican'},
             '1' : {'train_src':'./data/gender_data/male_only.train.en',
                    'train_tgt': './data/gender_data/female_only.train.en',
                    'valid_src': './data/gender_data/male_only.dev.en',
                    'valid_tgt': './data/gender_data/female_only.dev.en',
                    'dataset_name': 'male_female'}                            }

def print_sentence(args, dict, sent):
    print("TODO: Implement print sentence...")
    return

'''
import numpy as np
import torch, torch.nn as nn
from torch.autograd import Variable

random_input = Variable(torch.FloatTensor(5, 64, 128).normal_(), requires_grad=False)
#random_input[:, 0, 0]

bi_grus = torch.nn.GRU(input_size=128, hidden_size=256, num_layers=2, batch_first=False, bidirectional=True)

reverse_gru = torch.nn.GRU(input_size=128, hidden_size=256, num_layers=2, batch_first=False, bidirectional=False)

reverse_gru.weight_ih_l0 = bi_grus.weight_ih_l0_reverse
reverse_gru.weight_hh_l0 = bi_grus.weight_hh_l0_reverse
reverse_gru.bias_ih_l0 = bi_grus.bias_ih_l0_reverse
reverse_gru.bias_hh_l0 = bi_grus.bias_hh_l0_reverse

bi_output, bi_hidden = bi_grus(random_input)

reverse_output, reverse_hidden = reverse_gru(random_input[np.arange(4, -1, -1), :, :])

print("finished")
#reverse_output[:, 0, 0]

#bi_output[:, 0, 1]

#reverse_hidden

'''

#political_classifier = torch.load('./models/classifier/political_classifier/political_classifier.pt')
#print("debugging")