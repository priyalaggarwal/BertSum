import torch
from model import AbsSummarizer
from transformers import BertTokenizer
from transformers import Trainer, TrainingArguments
import data_loader
import random

# device = "cpu" if args.visible_gpus == '-1' else "cuda"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.manual_seed(666)
random.seed(666)
torch.backends.cudnn.deterministic = True
print("Setup done")

cache_dir="/projects/tir4/users/mbhandar2/transformer_models_cache"
model = AbsSummarizer(cache_dir)
print("AbsSummarizer done")

# TODO: Optimizer here

# BERT has its own tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=cache_dir)
symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
           'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}

print("BERT setup done")

# TODO: Custom loss function

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=1,              # total # of training epochs
    per_device_train_batch_size=64,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    # drop_last=True
)

print("TrainingArguments done")

data = data_loader.Dataset("individual")
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=data,                  # training dataset
    data_collator=data.collate_fn,
    # eval_dataset=test_dataset          # evaluation dataset
)

print("Trainer done")

# TODO: Will run only after previous todos are completed
trainer.train()