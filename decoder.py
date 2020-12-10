import codecs
import math
import pdb
import torch
from utils import rouge_results_to_str, test_rouge, tile

# from Queue import PriorityQueue
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# SOS_token = 0
# EOS_token = 1
# MAX_LENGTH = 50
#
# class BeamSearchNode(object):
#     def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
#         '''
#         :param hiddenstate:
#         :param previousNode:
#         :param wordId:
#         :param logProb:
#         :param length:
#         '''
#         self.h = hiddenstate
#         self.prevNode = previousNode
#         self.wordid = wordId
#         self.logp = logProb
#         self.leng = length
#
#     def eval(self, alpha=1.0):
#         reward = 0
#         # Add here a function for shaping a reward
#
#         return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward
#
# def beam_decode(target_tensor, decoder_hiddens, device, encoder_outputs=None):
#     '''
#     :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
#     :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
#     :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
#     :return: decoded_batch
#     '''
#
#     beam_width = 10
#     topk = 1  # how many sentence do you want to generate
#     decoded_batch = []
#
#     # decoding goes sentence by sentence
#     for idx in range(target_tensor.size(0)):
#         if isinstance(decoder_hiddens, tuple):  # LSTM case
#             decoder_hidden = (decoder_hiddens[0][:,idx, :].unsqueeze(0),decoder_hiddens[1][:,idx, :].unsqueeze(0))
#         else:
#             decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)
#         encoder_output = encoder_outputs[:,idx, :].unsqueeze(1)
#
#         # Start with the start of the sentence token
#         decoder_input = torch.LongTensor([[SOS_token]], device=device)
#
#         # Number of sentence to generate
#         endnodes = []
#         number_required = min((topk + 1), topk - len(endnodes))
#
#         # starting node -  hidden vector, previous node, word id, logp, length
#         node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
#         nodes = PriorityQueue()
#
#         # start the queue
#         nodes.put((-node.eval(), node))
#         qsize = 1
#
#         # start beam search
#         while True:
#             # give up when decoding takes too long
#             if qsize > 2000: break
#
#             # fetch the best node
#             score, n = nodes.get()
#             decoder_input = n.wordid
#             decoder_hidden = n.h
#
#             if n.wordid.item() == EOS_token and n.prevNode != None:
#                 endnodes.append((score, n))
#                 # if we reached maximum # of sentences required
#                 if len(endnodes) >= number_required:
#                     break
#                 else:
#                     continue
#
#             # decode for one step using decoder
#             decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)
#
#             # PUT HERE REAL BEAM SEARCH OF TOP
#             log_prob, indexes = torch.topk(decoder_output, beam_width)
#             nextnodes = []
#
#             for new_k in range(beam_width):
#                 decoded_t = indexes[0][new_k].view(1, -1)
#                 log_p = log_prob[0][new_k].item()
#
#                 node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
#                 score = -node.eval()
#                 nextnodes.append((score, node))
#
#             # put them into queue
#             for i in range(len(nextnodes)):
#                 score, nn = nextnodes[i]
#                 nodes.put((score, nn))
#                 # increase qsize
#             qsize += len(nextnodes) - 1
#
#         # choose nbest paths, back trace them
#         if len(endnodes) == 0:
#             endnodes = [nodes.get() for _ in range(topk)]
#
#         utterances = []
#         for score, n in sorted(endnodes, key=operator.itemgetter(0)):
#             utterance = []
#             utterance.append(n.wordid)
#             # back trace
#             while n.prevNode != None:
#                 n = n.prevNode
#                 utterance.append(n.wordid)
#
#             utterance = utterance[::-1]
#             utterances.append(utterance)
#
#         decoded_batch.append(utterances)
#
#     return decoded_batch


class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
       logger(logging.Logger): logger.
    """

    def __init__(self,
                 args,
                 model,
                 vocab,
                 symbols,
                 # global_scorer=None,
                 logger=None,
                 dump_beam=""):
        self.logger = logger

        self.args = args
        self.model = model
        self.generator = self.model.generator
        self.vocab = vocab
        self.symbols = symbols
        self.start_token = symbols['BOS']
        self.end_token = symbols['EOS']

        # self.global_scorer = global_scorer
        self.beam_size = args.beam_size
        self.min_length = args.min_length
        self.max_length = args.max_length

        self.dump_beam = dump_beam

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None

        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert (len(translation_batch["gold_score"]) ==
                len(translation_batch["predictions"]))
        batch_size = batch.batch_size

        preds, pred_score, gold_score, tgt_str, src =  translation_batch["predictions"],translation_batch["scores"],translation_batch["gold_score"],batch.tgt_str, batch.src

        translations = []
        for b in range(batch_size):
            pred_sents = self.vocab.convert_ids_to_tokens([int(n) for n in preds[b][0]])
            pred_sents = ' '.join(pred_sents).replace(' ##','')
            gold_sent = ' '.join(tgt_str[b].split())
            raw_src = [self.vocab.ids_to_tokens[int(t)] for t in src[b]][:500]
            raw_src = ' '.join(raw_src)
            translation = (pred_sents, gold_sent, raw_src)
            translations.append(translation)

        return translations

    def translate(self,
                  data_iter, step,
                  attn_debug=False):

        self.model.eval()

        gold_path = self.args.result_path + '.%d.gold' % step
        can_path = self.args.result_path + '.%d.candidate' % step
        self.gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        self.can_out_file = codecs.open(can_path, 'w', 'utf-8')
        raw_src_path = self.args.result_path + '.%d.raw_src' % step
        self.src_out_file = codecs.open(raw_src_path, 'w', 'utf-8')

        # pred_results, gold_results = [], []
        ct = 0
        with torch.no_grad():
            for batch in data_iter:
                batch_data = self.translate_batch(batch)
                translations = self.from_batch(batch_data)

                for trans in translations:
                    pred, gold, src = trans
                    pred_str = pred.replace('[unused0]', '').replace('[unused3]', '').replace('[PAD]', '').replace('[unused1]', '').replace(r' +', ' ').replace(' [unused2] ', '<q>').replace('[unused2]', '').strip()
                    gold_str = gold.strip()
                    if(self.args.recall_eval):
                        _pred_str = ''
                        gap = 1e3
                        for sent in pred_str.split('<q>'):
                            can_pred_str = _pred_str+ '<q>'+sent.strip()
                            can_gap = math.fabs(len(_pred_str.split())-len(gold_str.split()))
                            # if(can_gap>=gap):
                            if(len(can_pred_str.split())>=len(gold_str.split())+10):
                                pred_str = _pred_str
                                break
                            else:
                                gap = can_gap
                                _pred_str = can_pred_str

                    self.gold_out_file.write(gold_str + '\n')
                    self.src_out_file.write(src.strip() + '\n')
                    ct += 1
                self.can_out_file.flush()
                self.gold_out_file.flush()
                self.src_out_file.flush()

        self.can_out_file.close()
        self.gold_out_file.close()
        self.src_out_file.close()

        if (step != -1):
            rouges = self._report_rouge(gold_path, can_path)
            self.logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
            # if self.tensorboard_writer is not None:
            #     self.tensorboard_writer.add_scalar('test/rouge1-F', rouges['rouge_1_f_score'], step)
            #     self.tensorboard_writer.add_scalar('test/rouge2-F', rouges['rouge_2_f_score'], step)
            #     self.tensorboard_writer.add_scalar('test/rougeL-F', rouges['rouge_l_f_score'], step)

    def _report_rouge(self, gold_path, can_path):
        self.logger.info("Calculating Rouge")
        results_dict = test_rouge(self.args.temp_dir, can_path, gold_path)
        return results_dict

    def translate_batch(self, batch, fast=False):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)

        Todo:
           Shouldn't need the original dataset.
        """
        with torch.no_grad():
            return self._fast_translate_batch(
                batch,
                self.max_length,
                min_length=self.min_length)

    def _fast_translate_batch(self,
                              batch,
                              max_length,
                              min_length=0):
        # TODO: faster code path for beam_size == 1.

        # TODO: support these blacklisted features.
        assert not self.dump_beam

        beam_size = self.beam_size
        batch_size = batch.batch_size
        src = batch.src
        segs = batch.segs
        mask_src = batch.mask_src
        # pdb.set_trace()
        tgt = batch.tgt

        src_features = self.model.bert_model(input_ids=src, attention_mask=mask_src, token_type_ids=segs)

        dec_states = self.model.decoder.init_decoder_state(src, src_features, with_cache=True)
        device = src_features.device

        # Tile states and memory beam_size times.
        dec_states.map_batch_fn(
            lambda state, dim: tile(state, beam_size, dim=dim))

        src_features = tile(src_features, beam_size, dim=0)
        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=device)
        beam_offset = torch.arange(
            0,
            batch_size * beam_size,
            step=beam_size,
            dtype=torch.long,
            device=device)
        alive_seq = torch.full(
            [batch_size * beam_size, 1],
            self.start_token,
            dtype=torch.long,
            device=device)

        # Give full probability to the first beam on the first step.
        topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (beam_size - 1),
                         device=device).repeat(batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["gold_score"] = [0] * batch_size
        results["batch"] = batch

        for step in range(max_length):
            # pdb.set_trace()
            decoder_input = alive_seq[:, -1].view(1, -1)

            # Decoder forward.
            decoder_input = decoder_input.transpose(0,1)
            # pdb.set_trace()
            decoder_outputs = self.decoder(tgt=output.view(output.shape[1], output.shape[0], -1),
                                           memory=top_vec[0].view(top_vec[0].shape[1], top_vec[0].shape[0], -1),
                                           tgt_mask=tgt_mask,
                                           memory_mask=None,
                                           tgt_key_padding_mask=tgt_pad_mask,
                                           memory_key_padding_mask=src_pad_mask)

            dec_out, dec_states = self.model.decoder(decoder_input, src_features, dec_states,
                                                     step=step)#, edit_vec=z)
            # pdb.set_trace()

            # Generator forward.
            log_probs = self.generator.forward(dec_out.transpose(0, 1).squeeze(0))
            vocab_size = log_probs.size(-1)

            if step < min_length:
                log_probs[:, self.end_token] = -1e20

            # Multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            alpha = 0.6 # TODO: change if necessary - self.global_scorer.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

            # Flatten probs into a list of possibilities.
            curr_scores = log_probs / length_penalty

            if self.args.block_trigram:
                cur_len = alive_seq.size(1)
                if(cur_len>3):
                    for i in range(alive_seq.size(0)):
                        fail = False
                        words = [int(w) for w in alive_seq[i]]
                        words = [self.vocab.ids_to_tokens[w] for w in words]
                        words = ' '.join(words).replace(' ##','').split()
                        if(len(words)<=3):
                            continue
                        trigrams = [(words[i-1],words[i],words[i+1]) for i in range(1,len(words)-1)]
                        trigram = tuple(trigrams[-1])
                        if trigram in trigrams[:-1]:
                            fail = True
                        if fail:
                            curr_scores[i] = -10e20

            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation.
            batch_index = (
                    topk_beam_index
                    + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)

            # Append last prediction.
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                 topk_ids.view(-1, 1)], -1)

            is_finished = topk_ids.eq(self.end_token)
            if step + 1 == max_length:
                is_finished.fill_(1)
            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(1)
            # Save finished hypotheses.
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append((
                            topk_scores[i, j],
                            predictions[i, j, 1:]))
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)
                        score, pred = best_hyp[0]

                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
                non_finished = end_condition.eq(0).nonzero().view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))
            # Reorder states.
            select_indices = batch_index.view(-1)
            src_features = src_features.index_select(0, select_indices)

            dec_states.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices))

        return results