import torch
import os
from torch.nn.utils.rnn import pad_sequence

class Batch:
    def __init__(self, data_points: list):
        """
        :param data_points: Each data point is a dictionary of the following -
        ['src', 'tgt', 'src_sent_labels', 'segs', 'clss', 'src_txt', 'tgt_txt']
        """
        self.src = [torch.tensor(d['src']) for d in data_points]
        self.tgt = [torch.tensor(d['tgt']) for d in data_points]
        self.segs = [torch.tensor(d['segs']) for d in data_points]

        # Keeping src_txt and tgt_txt for debugging purposes
        self.src_txt = [d['src_txt'] for d in data_points]
        self.tgt_txt = [d['tgt_txt'] for d in data_points]
        self.max_pos = 512
        self.max_tgt_len = 140

    def __len__(self):
        return len(self.src)

    def pad(self):
        # PAD token_id in BertTokenizer is 0 by default
        self.src = pad_sequence(self.src, batch_first=True)
        self.tgt = pad_sequence(self.tgt, batch_first=True)
        self.segs = pad_sequence(self.segs, batch_first=True)

    def preprocess(self):
        # Truncate input to 512 sequence length
        tgt = self.tgt[:, :self.max_tgt_len][:, :-1]
        x = torch.full((tgt.shape[0], 1), 2, dtype=tgt.dtype)
        self.tgt = torch.cat([tgt, x], dim = -1)

        end_id = self.src[:,-1].unsqueeze(1)
        src = self.src[:, :-1][:,:self.max_pos - 1]
        self.src = torch.cat([src, end_id], dim=-1)

        self.segs = self.segs[:,:self.max_pos]
        # print(self.src.shape)
        # print(self.tgt.shape)
        # print(self.segs.shape)

        self.mask_src = ~ (self.src == 0)
        self.mask_tgt = ~ (self.tgt == 0)


    def to(self, device):
        self.src = self.src.to(device)
        self.tgt = self.tgt.to(device)
        self.segs = self.segs.to(device)
        self.mask_src = self.mask_src.to(device)
        self.mask_tgt = self.mask_tgt.to(device)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root):
        'Initialization'
        self.root = root
        self.files = os.listdir(root)  # take all files in the root directory

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.files)

    def __getitem__(self, idx):
        'Generates one sample of data'
        sample = torch.load(os.path.join(self.root, self.files[idx]))  # load the features of this sample
        return sample

    def collate_fn(self, data_points: list):
        batch = Batch(data_points)
        batch.pad()
        batch.preprocess()
        return batch


# sample call code
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if use_cuda else "cpu")
# print(device)
# torch.backends.cudnn.benchmark = True
# dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
#
# data = Dataset("individual")
#
# params = {'batch_size': 64,
#           'shuffle': True,
#           'num_workers': 4}
# training_generator = torch.utils.data.DataLoader(data, **params, drop_last=True, collate_fn=data.collate_fn)
#
# for local_batch in training_generator:
#     local_batch.to(device)
#     # print(len(local_batch))
#     print(local_batch.src[0])
#     print(local_batch.mask_tgt)
#     print(local_batch.src.shape)
#     import pdb; pdb.set_trace()
#     break
