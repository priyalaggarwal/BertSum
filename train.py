import torch
from model import AbsSummarizer
from transformers import BertTokenizer
from transformers import Trainer, TrainingArguments
import data_loader
import random
import optimizer

# device = "cpu" if args.visible_gpus == '-1' else "cuda"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.manual_seed(666)
random.seed(666)
torch.backends.cudnn.deterministic = True
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
optim = [optim_bert, optim_dec]

# Get Tokenizer. BERT has its own tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=cache_dir)
symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
           'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}

print("BERT setup done")

data = data_loader.Dataset("individual")

# TODO: Custom loss function

# TODO: Below code will have to be substantially changed
# training_args = TrainingArguments(
#     output_dir='./results',          # output directory
#     num_train_epochs=1,              # total # of training epochs
#     per_device_train_batch_size=64,  # batch size per device during training
#     per_device_eval_batch_size=64,   # batch size for evaluation
#     warmup_steps=500,                # number of warmup steps for learning rate scheduler
#     weight_decay=0.01,               # strength of weight decay
#     logging_dir='./logs',            # directory for storing logs
#     # drop_last=True
# )

# print("TrainingArguments done")

# trainer = Trainer(
#     model=model,                         # the instantiated 🤗 Transformers model to be trained
#     args=training_args,                  # training arguments, defined above
#     train_dataset=data,                  # training dataset
#     data_collator=data.collate_fn,
#     # eval_dataset=test_dataset          # evaluation dataset
# )

# print("Trainer done")

# TODO: Look into creating checkpoints
# TODO: Will have to write own train function to allow encoder and decoder to train using different optimizers
# trainer.train()

# TODO: Update params
epochs = 1000
curr_epoch = 0

print("Starting training...")

batch_size = 4
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 4}
training_generator = torch.utils.data.DataLoader(data, **params, drop_last=True, collate_fn=data.collate_fn)

while curr_epoch <= epochs:
    for batch in training_generator:
        batch.to(device)

        num_tokens = batch.tgt[:, 1:].ne(0).sum() # not equal to Pad id, which is 0
        # any tensor on GPU has a number of fields associated
        # .item() gets scalar value
        normalization = num_tokens.item()

        decoder_output = model(batch.src, batch.tgt[:, :-1], batch.segs, batch.mask_src)
        # TODO: Loss calculation, optimizer.step
        # print(decoder_output)
        # import pdb; pdb.set_trace()
        break
    break