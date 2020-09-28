from transformers import BertModel, BertConfig
import torch
import torch.nn as nn

class AbsSummarizer(nn.Module):
    # TODO: Decoder
    def __init__(self, temp_dir):
        super(AbsSummarizer, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)
        self.vocab_size = self.bert_model.config.vocab_size

        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert_model.config.hidden_size, padding_idx=0)

    def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        dec_state = self.decoder.init_decoder_state(src, top_vec)
        decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)
        return decoder_outputs, None
