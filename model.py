from transformers import BertModel, BertConfig
import torch
import torch.nn as nn
import math
from torch.nn.init import xavier_uniform_
import copy
import numpy as np

class AbsSummarizer(nn.Module):
    # TODO: Write eval mode for the entire model
    def __init__(self, temp_dir, device):
        super(AbsSummarizer, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)
        self.vocab_size = self.bert_model.config.vocab_size
        self.hidden_size = 768 # hidden_size is given as input to d_model which is the size of embedding
        self.dropout = 0.2

        # TODO: TransformerDecoder state - beam search
        # taking default feedforward size which is 2048
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_size, nhead=8, dropout=self.dropout)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, 6)

        self.linear_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.vocab_size),
            nn.LogSoftmax(dim=-1)
        )

        # Perform weight initializations
        for module in self.decoder.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        for p in self.linear_layer.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
            else:
                p.data.zero_()

        # use_bert_emb is true in input parameters

        # Token embeddings
        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert_model.config.hidden_size, padding_idx=0)
        tgt_embeddings.weight = copy.deepcopy(self.bert_model.embeddings.word_embeddings.weight)
        self.embeddings = tgt_embeddings
        self.linear_layer[0].weight = self.embeddings.weight

        # Positional embeddings
        self.pos_emb = PositionalEncoding(self.dropout, self.embeddings.embedding_dim)

        self.to(device)

    def forward(self, src, tgt, segs, mask_src=None, memory_bank=None, step=None):

        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim
        output = self.pos_emb(emb, step)

        # bs, input_token_size, _ = memory_bank.shape
        # _, num_decode_tokens, _ = output.shape

        padding_idx = self.embeddings.padding_idx

        # src_batch, src_len = src.size()
        tgt_batch, tgt_len = tgt.size()

        src_pad_mask = src.data.eq(padding_idx)#.unsqueeze(1).expand(src_batch, tgt_len, src_len)
        tgt_pad_mask = tgt.data.eq(padding_idx)#.unsqueeze(1).expand(tgt_batch, tgt_len, tgt_len)

        tgt_mask = np.triu(np.ones((tgt_len, tgt_len)), k=1).astype('uint8') # generates an upper triangular matrix
        tgt_mask = torch.from_numpy(tgt_mask)

        # tgt_mask = nn.Transformer().generate_square_subsequent_mask(sz=tgt_len)
        print(src.type())
        print(segs.type())
        print(mask_src.type())

        # batch size X sequence length X embedding dimension
        # import pdb; pdb.set_trace()
        top_vec = self.bert_model(input_ids=src, attention_mask=mask_src, token_type_ids=segs)
        # print(top_vec)
        # TODO: Send correct inputs and masks to decoder
        decoder_outputs, _ = self.decoder(tgt = output[:, :-1],
                                          memory = top_vec,
                                          tgt_mask = tgt_mask,
                                          memory_mask = None,
                                          tgt_key_padding_mask = tgt_pad_mask,
                                          memory_key_padding_mask = src_pad_mask)
        return decoder_outputs


class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if (step):
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]