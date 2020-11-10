"""
This file takes all input pt files from BERTSUM and splits them into files containing individual samples.
"""

import torch
import glob
import random


def load_dataset(bert_data_path, corpus_type, shuffle):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.
    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        print('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(bert_data_path + '.' + corpus_type + '.[0-9]*.bert.pt'))
    if pts:
        if (shuffle):
            random.shuffle(pts)

        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = bert_data_path + '.' + corpus_type + '.pt'
        yield _lazy_dataset_loader(pt, corpus_type)

print(torch.__version__)
data = load_dataset(bert_data_path="bert_data_cnndm_final/cnndm", corpus_type="train", shuffle=True)

i = 0
for pt in data:
    for sample in pt:
        torch.save(sample, "full_data/cnndm.train.bert." + str(i) + ".pt")
        # print(type())
        i+=1

print(i)