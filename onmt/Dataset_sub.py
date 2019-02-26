from __future__ import division
import math
import torch
from torch.autograd import Variable
import onmt.Constants


class Dataset(object):

    def __init__(self, sent1Data, sent2Data, batchSize, cuda, volatile=False):
        assert(len(sent1Data) == len(sent2Data))
        self.data = [(sent1Data[i], sent2Data[i]) for i in range(len(sent1Data))]
        self.cuda = cuda
        self.batchSize = batchSize
        self.numBatches = math.ceil(len(self.data)/batchSize)
        self.volatile = volatile

    def _batchify(self, data, align_right=False, include_lengths=False):
        lengths = [(x[0].size(0), x[1].size(0)) for x in data]
        max_length = max([max(lng) for lng in lengths])
        out = data[0][0].new(len(data), 2, max_length).fill_(onmt.Constants.PAD)
        for i in range(len(data)):
            data_length = (data[i][0].size(0), data[i][1].size(0))
            offset = (max_length - data_length[0] if align_right else 0, max_length - data_length[1] if align_right else 0)
            out[i][0].narrow(0, offset[0], data_length[0]).copy_(data[i][0])
            out[i][1].narrow(0, offset[0], data_length[1]).copy_(data[i][1])

        if include_lengths:
            return out, lengths
        else:
            return out

    def __getitem__(self, index): # index = batch index..
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches) # checking the index is legal, otherwise print error message
        batch, lengths = self._batchify(self.data[index*self.batchSize:(index+1)*self.batchSize], align_right=False, include_lengths=True)

        # within batch sorting by decreasing length for variable length rnns
        #indices = range(len(srcBatch))
        #batch = zip(indices, srcBatch) if tgtBatch is None else zip(indices, srcBatch, tgtBatch)
        #batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
        #if tgtBatch is None:
        #    indices, srcBatch = zip(*batch)
        #else:
        #    indices, srcBatch, tgtBatch = zip(*batch)

        def wrap(b):
            if b is None:
                return b
            #b = torch.stack([torch.stack(tpl, 0).t().contiguous() for tpl in b], 0).t().contiguous()
            if self.cuda:
                b = b.cuda()
            b = Variable(b, volatile=self.volatile)
            return b

        return wrap(batch), lengths

    def __len__(self):
        return self.numBatches

    '''
    def shuffle(self):
        data = list(zip(self.src, self.tgt))
        self.src, self.tgt = zip(*[data[i] for i in torch.randperm(len(data))])
    '''

    def shuffle(self):
        data_to_shuffle = list(self.data)
        self.data = [data_to_shuffle[i] for i in torch.randperm(len(data_to_shuffle))]