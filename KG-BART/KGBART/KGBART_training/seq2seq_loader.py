from random import randint, shuffle, choice
from random import random as rand
import math
import torch

import sys
sys.path.append('~/RHO/KG-BART/KGBART')
from KGBART_training.loader_utils import get_random_word, batch_list_to_batch_tensors, Pipeline, get_id

import json
from random import sample, randint, random
from tqdm import tqdm


# Input file format :
# 1. One sentence per line. These should ideally be actual sentences,
#    not entire paragraphs or arbitrary spans of text. (Because we use
#    the sentence boundaries for the "next sentence prediction" task).
# 2. Blank lines between documents. Document boundaries are needed
#    so that the "next sentence prediction" task doesn't span between documents.


def truncate_tokens_pair(tokens_a, tokens_b, max_len, max_len_a=0, max_len_b=0, trunc_seg=None,
                         always_truncate_tail=False):
    num_truncated_a = [0, 0]
    num_truncated_b = [0, 0]
    while True:
        if len(tokens_a) + len(tokens_b) <= max_len:
            break
        if (max_len_a > 0) and len(tokens_a) > max_len_a:
            trunc_tokens = tokens_a
            num_truncated = num_truncated_a
        elif (max_len_b > 0) and len(tokens_b) > max_len_b:
            trunc_tokens = tokens_b
            num_truncated = num_truncated_b
        elif trunc_seg:
            # truncate the specified segment
            if trunc_seg == 'a':
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
        else:
            # truncate the longer segment
            if len(tokens_a) > len(tokens_b):
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
        # whether always truncate source sequences
        if (not always_truncate_tail) and (rand() < 0.5):
            del trunc_tokens[0]
            num_truncated[0] += 1
        else:
            trunc_tokens.pop()
            num_truncated[1] += 1
    return num_truncated_a, num_truncated_b



class Pretrain_Seq2SeqDataset(torch.utils.data.Dataset):
    """ Load sentence pair (sequential or random order) from corpus """

    def __init__(self, pretraining_KG, file_entity_id, file_relation_id, batch_size, tokenizer, max_len, short_sampling_prob=0.1, sent_reverse_order=False, bi_uni_pipeline=[], strict=True):
        super().__init__()
        self.tokenizer = tokenizer  # tokenize function
        self.max_len = max_len  # maximum length of tokens
        self.short_sampling_prob = short_sampling_prob
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size
        self.sent_reverse_order = sent_reverse_order
        

        # read the file into memory
        self.ex_list = []
        too_long_idxs = []
        with open(file_entity_id, "r", encoding='utf-8') as f_entity, \
        open(file_relation_id, "r", encoding='utf-8') as f_relation:
            entity_id = {}
            id_entity = {}
            entity_line = f_entity.readlines()
            for i, token in enumerate(["</s>", "<s>", "<pad>", "<mask>"]):
                entity_id[token] = i
                id_entity[i] = token
                
            for i, entityid in enumerate(entity_line):
                if i == 0: 
                    num_entity = int(entityid.strip())+4
                    continue
                items = entityid.split('\t')
                assert len(items) == 2
                entity_id[items[0]] = int(items[1]) + 4
                id_entity[int(items[1]) + 4] = items[0]
                
            if strict:
                assert num_entity == len(entity_id.keys())
            
            relation_id = {}
            id_relation = {}
            relation_line = f_relation.readlines()
            for i, relationid in enumerate(relation_line):
                if i == 0: continue
                items = relationid.split('\t')
                assert len(items) == 2
                relation_id[items[0]] = int(items[1])
                id_relation[int(items[1])] = items[0]


        with open(pretraining_KG, "r", encoding='utf-8') as f_src:
            for idx, line in enumerate(f_src.readlines()):
                line = line.strip()

                src_entity_relation_id = []
                word_subword = []
                src_tk = []
                tgt_tk = []


                entity_relation_id_per_line = []
                for triple in line.split('<sep>'):
                    
                    items = triple.split('\t')
                    assert len(items) == 3

                    entity_tk0 = tokenizer.tokenize(items[0])
                    entity_tk1 = tokenizer.tokenize(items[1])
                    relation_tk = tokenizer.tokenize(items[2])

                    word_subword.append([len(entity_tk0),
                                         len(entity_tk1),
                                         len(relation_tk)])

                    src_entity_relation_id.append([get_id(entity_id, items[0], strict),
                                                   get_id(entity_id, items[1], strict),
                                                   get_id(relation_id, items[2], strict)+num_entity])


                    src_tk += ['<s>']+entity_tk0+['<s>']+entity_tk1+['<s>']
                    tgt_tk += ['<s>']+entity_tk0+['<s>']+entity_tk1+['<s>']


                    for tk in relation_tk:
                        src_tk.append("<mask>")
                        tgt_tk.append(tk)

                    
                src_tk += ['</s>']
                tgt_tk += ['</s>']
                if len(src_tk) < max_len:
                    self.ex_list.append((src_tk, tgt_tk, src_entity_relation_id, word_subword))
                else:
                    too_long_idxs.append(idx)
                    
        print(f'Load {len(self.ex_list)} documents')
        print(f'too_long documents: {too_long_idxs}')

    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        instance = self.ex_list[idx]
        proc = choice(self.bi_uni_pipeline)
        instance = proc(instance)
        return instance

    def __iter__(self):  # iterator to load data
        for __ in range(math.ceil(len(self.ex_list) / float(self.batch_size))):
            batch = []
            for __ in range(self.batch_size):
                idx = randint(0, len(self.ex_list) - 1)
                batch.append(self.__getitem__(idx))
            # To Tensor
            yield batch_list_to_batch_tensors(batch)
            
            
class Seq2SeqDataset(torch.utils.data.Dataset):
    """ Load sentence pair (sequential or random order) from corpus """

    def __init__(self, file_triple, file_hist, file_tgt, file_entity_id, file_relation_id,
                 batch_size, tokenizer,
                 max_len_src, max_len_tgt, 
                 short_sampling_prob=0.1, sent_reverse_order=False, bi_uni_pipeline=[], strict=True):
        super().__init__()
        self.tokenizer = tokenizer  # tokenize function
        self.max_len_src = max_len_src  # maximum length of tokens
        self.max_len_tgt = max_len_tgt  # maximum length of tokens
        
        self.short_sampling_prob = short_sampling_prob
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size
        self.sent_reverse_order = sent_reverse_order

        # read the file into memory
        
        with open(file_entity_id, "r", encoding='utf-8') as f_entity, \
        open(file_relation_id, "r", encoding='utf-8') as f_relation:
            entity_id = {}
            id_entity = {}
            entity_line = f_entity.readlines()
            for i, token in enumerate(["</s>", "<s>", "<pad>", "<mask>"]):
                entity_id[token] = i
                id_entity[i] = token
                
            for i, entityid in enumerate(entity_line):
                if i == 0: 
                    num_entity = int(entityid.strip())+4
                    continue
                items = entityid.split('\t')
                assert len(items) == 2
                entity_id[items[0]] = int(items[1]) + 4
                id_entity[int(items[1]) + 4] = items[0]
                
            if strict:
                assert num_entity == len(entity_id.keys())
            
            relation_id = {}
            id_relation = {}
            relation_line = f_relation.readlines()
            for i, relationid in enumerate(relation_line):
                if i == 0: continue
                items = relationid.split('\t')
                assert len(items) == 2
                relation_id[items[0]] = int(items[1])
                id_relation[int(items[1])] = items[0]
                
        too_long_idxs = []
        self.ex_list = []
        with open(file_triple, "r", encoding='utf-8') as f_triple, open(file_hist, "r", encoding='utf-8') as f_hist, open(file_tgt, "r", encoding='utf-8') as f_tgt:
            for idx, (line_triple, line_hist, line_tgt) in enumerate(zip(f_triple.readlines(), f_hist.readlines(), f_tgt.readlines())):
                line_triple = line_triple.strip()
                line_hist = line_hist.strip()
                line_tgt = line_tgt.strip()

                src_entity_relation_id = []
                word_subword = []
                src_tk = []
                tgt_tk = []


                entity_relation_id_per_line = []
                for triple in line_triple.split('<sep>'):
                    
                    items = triple.split('\t')
                    assert len(items) == 3

                    entity_tk0 = tokenizer.tokenize(items[0])
                    entity_tk1 = tokenizer.tokenize(items[1])
                    relation_tk = tokenizer.tokenize(items[2])

                    word_subword.append([len(entity_tk0),
                                         len(entity_tk1),
                                         len(relation_tk)])
                    
                    src_entity_relation_id.append([get_id(entity_id, items[0], strict),
                                                   get_id(entity_id, items[1], strict),
                                                   get_id(relation_id, items[2], strict)+num_entity])
                    src_tk += ['<s>']+entity_tk0+['<s>']+entity_tk1+['<s>']
                    
                    
                    
                src_tk += tokenizer.tokenize(line_hist)+['</s>']
                tgt_tk = ['<s>']+tokenizer.tokenize(line_tgt)+['</s>']
                
                if len(src_tk) < max_len_src and len(tgt_tk) < max_len_tgt:
                    self.ex_list.append((src_tk, tgt_tk, src_entity_relation_id, word_subword))
                else:
                    too_long_idxs.append(idx)
        print(f'Load {len(self.ex_list)} documents')
        print(f'too_long documents: {too_long_idxs}')

        



    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        instance = self.ex_list[idx]
        proc = choice(self.bi_uni_pipeline)
        instance = proc(instance)
        return instance

    def __iter__(self):  # iterator to load data
        for __ in range(math.ceil(len(self.ex_list) / float(self.batch_size))):
            batch = []
            for __ in range(self.batch_size):
                idx = randint(0, len(self.ex_list) - 1)
                batch.append(self.__getitem__(idx))
            # To Tensor
            yield batch_list_to_batch_tensors(batch)


class Preprocess4Pretrain(Pipeline):
    def __init__(self, vocab_words, indexer, max_len=512, new_segment_ids=False, truncate_config={},
                 mask_source_words=False, mode="s2s", num_qkv=0, s2s_special_token=False,
                 s2s_add_segment=False, s2s_share_segment=False, pos_shift=False):
        super().__init__()
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.new_segment_ids = new_segment_ids
        self.always_truncate_tail = truncate_config.get(
            'always_truncate_tail', False)
        self.max_len_a = truncate_config.get('max_len_a', None)
        self.max_len_b = truncate_config.get('max_len_b', None)
        self.trunc_seg = truncate_config.get('trunc_seg', None)
        self.task_idx = 3  # relax projection layer for different tasks
        self.mask_source_words = mask_source_words
        assert mode in ("s2s", "l2r")
        self.mode = mode
        self.num_qkv = num_qkv
        self.s2s_special_token = s2s_special_token
        self.s2s_add_segment = s2s_add_segment
        self.s2s_share_segment = s2s_share_segment
        self.pos_shift = pos_shift
        

    def __call__(self, instance):
        tokens_a, tokens_b, entity_id_a, word_subword = instance
        
        entity_id_a = [item for sublist in entity_id_a for item in sublist]
        word_subword = [item for sublist in word_subword for item in sublist]
        
        
        sep_token = "</s>"
        cls_token = "<s>"
        pad_token = "<pad>"
        mask_token = "<mask>"


        n_pad = self.max_len_a - len(word_subword)
        word_subword.extend([0] * n_pad)

        labels = self.indexer(tokens_b)

        
        input_entity_relation_ids = self.indexer([cls_token]) + entity_id_a + self.indexer([sep_token])
        input_ids = self.indexer(tokens_a)
        decoder_input_ids = self.indexer(['</s>']+tokens_b[:-1])
        

        # Zero Padding
        n_pad = self.max_len_a - len(input_ids)
        subword_mask = [1] * len(input_ids)
        subword_mask.extend([0] * n_pad)
        input_ids.extend([self.indexer(pad_token)] * n_pad)

        n_pad = self.max_len_a - len(input_entity_relation_ids)
        word_mask = [1] * len(input_entity_relation_ids)
        word_mask.extend([0] * n_pad)
        input_entity_relation_ids.extend([self.indexer(pad_token)] * n_pad)

        n_pad = self.max_len_b - len(decoder_input_ids)
        decoder_input_ids.extend([self.indexer(pad_token)] * n_pad)
        decoder_attention_mask = [1] * len(tokens_b)
        decoder_attention_mask.extend([0] * n_pad)
        
        
        labels.extend([-100] * n_pad)

#         if self.num_qkv > 1:
#             mask_qkv = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
#             mask_qkv.extend([0] * n_pad)
#         else:
#             mask_qkv = None
        return (input_ids, input_entity_relation_ids, subword_mask, word_mask, word_subword, decoder_input_ids, decoder_attention_mask, labels)


class Preprocess4Seq2seq(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, vocab_words, indexer, new_segment_ids=False, truncate_config={},
                 mask_source_words=False, mode="s2s", num_qkv=0, s2s_special_token=False,
                 s2s_add_segment=False, s2s_share_segment=False, pos_shift=False):
        super().__init__()
        
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        
        self.new_segment_ids = new_segment_ids
        self.always_truncate_tail = truncate_config.get(
            'always_truncate_tail', False)
        self.max_len_a = truncate_config.get('max_len_a', None)
        self.max_len_b = truncate_config.get('max_len_b', None)
        self.trunc_seg = truncate_config.get('trunc_seg', None)
        
#         self.max_len = max(self.max_len_a, self.max_len_b)
#         self._tril_matrix = torch.tril(torch.ones(
#             (max_len, max_len), dtype=torch.long))
        
        self.task_idx = 3  # relax projection layer for different tasks
        self.mask_source_words = mask_source_words
        assert mode in ("s2s", "l2r")
        self.mode = mode
        self.num_qkv = num_qkv
        self.s2s_special_token = s2s_special_token
        self.s2s_add_segment = s2s_add_segment
        self.s2s_share_segment = s2s_share_segment
        self.pos_shift = pos_shift

#         src_tk, tgt_tk, src_entity_relation_id, word_subword
    def __call__(self, instance):
        tokens_a, tokens_b, entity_id_a, word_subword = instance
        sep_token = "</s>"
        cls_token = "<s>"
        pad_token = "<pad>"
        mask_token = "<mask>"
        
        entity_id_a = [item for sublist in entity_id_a for item in sublist]
        word_subword = [item for sublist in word_subword for item in sublist]
        

        n_pad = self.max_len_a - len(word_subword)
        word_subword.extend([0] * n_pad)

        labels = self.indexer(tokens_b)

        input_entity_relation_ids = self.indexer([cls_token]) + entity_id_a + self.indexer([sep_token])
        input_ids = self.indexer(tokens_a)  # [self.indexer(tokens) for tokens in tokens_a]
        decoder_input_ids = self.indexer(['</s>']+tokens_b[:-1])

        # Zero Padding
        n_pad = self.max_len_a - len(input_ids)
        subword_mask = [1] * len(input_ids)
        subword_mask.extend([0] * n_pad)
        input_ids.extend([self.indexer(pad_token)] * n_pad)

        n_pad = self.max_len_a - len(input_entity_relation_ids)
        word_mask = [1] * len(input_entity_relation_ids)
        word_mask = word_mask + [0] * n_pad
        input_entity_relation_ids.extend([self.indexer(pad_token)] * n_pad)
        

        n_pad = self.max_len_b - len(decoder_input_ids)
        decoder_input_ids.extend([self.indexer(pad_token)] * n_pad)
        decoder_attention_mask = [1] * len(tokens_b)
        decoder_attention_mask.extend([0] * n_pad)
        
        labels.extend([-100] * n_pad)

        if self.num_qkv > 1:
            mask_qkv = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
            mask_qkv.extend([0] * n_pad)
        else:
            mask_qkv = None

        return (input_ids, word_subword, input_entity_relation_ids, subword_mask, word_mask, decoder_input_ids, decoder_attention_mask, labels)


# class Preprocess4Seq2seqDecoder(Pipeline):
#     """ Pre-processing steps for pretraining transformer """

#     def __init__(self, vocab_words, indexer, max_len=512, max_tgt_length=128, new_segment_ids=False, mode="s2s",
#                  num_qkv=0, s2s_special_token=False, s2s_add_segment=False, s2s_share_segment=False, pos_shift=False):
#         super().__init__()
#         self.max_len = max_len
#         self.vocab_words = vocab_words  # vocabulary (sub)words
#         self.indexer = indexer  # function from token to token index
#         self.max_len = max_len
#         self._tril_matrix = torch.tril(torch.ones(
#             (max_len, max_len), dtype=torch.long))
#         self.new_segment_ids = new_segment_ids
#         assert mode in ("s2s", "l2r")
#         self.mode = mode
#         self.max_tgt_length = max_tgt_length
#         self.num_qkv = num_qkv
#         self.s2s_special_token = s2s_special_token
#         self.s2s_add_segment = s2s_add_segment
#         self.s2s_share_segment = s2s_share_segment
#         self.pos_shift = pos_shift

#     def __call__(self, instance):
#         tokens_a, entity_id_a, word_subword, concept_entity_expand, concept_relation_expand = instance
#         sep_token = "</s>"
#         cls_token = "<s>"
#         pad_token = "<pad>"
#         mask_token = "<mask>"

#         input_entity_relation_ids = self.indexer([cls_token]) + entity_id_a + self.indexer([sep_token])

#         tokens_a = [cls_token] + tokens_a + [sep_token]

#         n_pad = self.max_len - len(input_entity_relation_ids)
#         word_mask = [1] * len(input_entity_relation_ids)
#         word_mask.extend([0] * n_pad)
#         input_entity_relation_ids.extend([self.indexer(pad_token)] * n_pad)

#         n_pad = self.max_len - len(word_subword)
#         word_subword.extend([0] * n_pad)

#         tokens = tokens_a

#         # Token Indexing
#         input_ids = [self.indexer(t) for t in tokens]

#         if len(input_ids) >= self.max_len:
#             print(tokens)
#         input_mask = [1] * len(tokens)

#         n_pad = self.max_len - len(input_ids)
#         subword_mask = [1] * len(input_ids)
#         subword_mask.extend([0] * n_pad)
#         input_ids.extend([self.indexer(pad_token)] * n_pad)
#         input_mask.extend([0] * n_pad)
        
#         assert len(input_ids) == self.max_len
#         assert len(input_mask) == self.max_len
#         assert len(input_entity_relation_ids) == self.max_len
#         return (input_ids, input_entity_relation_ids, subword_mask, word_mask, word_subword, concept_entity_expand,
#                 concept_relation_expand)
