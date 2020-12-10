from decoder import Translator
from model import AbsSummarizer
import data_loader
import torch
from transformers import BertTokenizer

from utils import logger

model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']


batch_size = 8
params = {'batch_size': batch_size,
          'shuffle': False,
          'num_workers': 8,
          'pin_memory': True}


def test_abs(args, device, pt, step, model_path):
    # pdb.set_trace()
    if pt != '':
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)

    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])

    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    model = AbsSummarizer(args.temp_dir, device, checkpoint, model_path=model_path)
    model.eval()

    data = data_loader.Dataset("test_data")
    testing_generator = torch.utils.data.DataLoader(data, **params, drop_last=True, collate_fn=data.collate_fn)

    # TODO: verify is_test label
    # test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
    #                                    args.test_batch_size, device,
    #                                    shuffle=False, is_test=True)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=args.temp_dir)
    symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
               'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}

    translator = Translator(args, device, model, tokenizer, symbols, logger=logger)
    translator.translate(testing_generator, step)

