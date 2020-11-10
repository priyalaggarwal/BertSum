import torch
from model import AbsSummarizer
from transformers import BertTokenizer
from transformers import Trainer, TrainingArguments
import data_loader
import random
import optimizer
import loss
import torch.nn as nn
import utils
import logging

# device = "cpu" if args.visible_gpus == '-1' else "cuda"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.manual_seed(666)
random.seed(666)
torch.backends.cudnn.deterministic = True

logger = utils.init_logger('logs.txt', logging.INFO)
logger.info("FIRST LOG")

print("Setup done")

# Get AbsSummarizer model
cache_dir="/projects/tir4/users/mbhandar2/transformer_models_cache"
model = AbsSummarizer(cache_dir, device)
print("AbsSummarizer done")

# Get separate optimizers for BERT encoder and Transformer Decoder
optim_bert_args = optimizer.OptimizerArgs(lr=0.002, warmup_steps=20000)
optim_decoder_args = optimizer.OptimizerArgs(lr=0.2, warmup_steps=10000)

optim_bert = optimizer.optim_bert(optim_bert_args, model)
optim_dec = optimizer.optim_decoder(optim_decoder_args, model)
optims = [optim_bert, optim_dec]

# Get Tokenizer. BERT has its own tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=cache_dir)
symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
           'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}

print("BERT setup done")

# data = data_loader.Dataset("individual")
data = data_loader.Dataset("full_data")

# TODO: Look into creating checkpoints

print("Starting training...")

batch_size = 8
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 8,
          'pin_memory': True}
training_generator = torch.utils.data.DataLoader(data, **params, drop_last=True, collate_fn=data.collate_fn)

padding_index = 0

loss_fn = loss.LabelSmoothingLoss(label_smoothing=0.1,
                                  tgt_vocab_size=model.vocab_size,
                                  device=device,
                                  ignore_index=padding_index)

try_loss = nn.NLLLoss(ignore_index=padding_index, reduction='sum')

step = 0
total_steps = 200000
best_perplexity = None

while step <= total_steps:
    for batch in training_generator:
        batch.to(device)

        model.zero_grad()

        # Not taking the first token in the batch, because we are predicting from the second word onwards
        # First word is Start of Sentence, which we don't want to predict
        num_tokens = batch.tgt[:, 1:].ne(padding_index).sum() # not equal to Pad id, which is 0

        # any tensor on GPU has a number of fields associated
        # .item() gets scalar value
        normalization = num_tokens.item()

        # We don't give End of Sentence as input
        model_output = model(batch.src, batch.tgt[:, :-1], batch.segs, batch.mask_src)
        #
        # loss = loss_fn(model_output, batch.tgt[:, 1:].reshape(-1))
        loss = loss_fn(model_output, batch.tgt[:, 1:].contiguous().view(-1))
        # print(loss.item())

        # remove first element which is start of sentence for each document.
        # loss = try_loss(model_output, batch.tgt[:, 1:].reshape(-1))
        # loss = try_loss(model_output, batch.tgt[:, 1:].contiguous().view(-1))
        loss.div(float(normalization)).backward()

        for o in optims:
            o.step()

        if step%50==0:
            print(loss.div(float(normalization)).item())
            print(step)

        step += 1
        perplexity = utils.get_perplexity(loss, normalization)
        logger.info(perplexity)
        if best_perplexity is None or perplexity < best_perplexity:
            best_perplexity = perplexity

        if step%2000 == 0:
            if perplexity == best_perplexity:
                utils.save_checkpoint(model, step)

        if step >= total_steps:
            break

        # break

# print("Saving model")
# torch.save(model.state_dict(), "first_model.pt")

"""
TODO: 
1. Debug LabelSmoothingLoss - done
2. Add logging - done
3. Calculate Perplexity - done
4. Model checkpoint - check validation for all, test for lowest
5. Eval mode for model 
6. Generation - beam search
"""