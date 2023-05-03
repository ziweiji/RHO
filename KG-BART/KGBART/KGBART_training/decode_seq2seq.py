"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import glob
import argparse
import math
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import random
import pickle
import os
import json

import re
import sys
sys.path.append('~/RHO/KG-BART/KGBART')
from nn.data_parallel import DataParallelImbalance
import KGBART_training.seq2seq_loader as seq2seq_loader
from KGBART_model.tokenization_bart import MBartTokenizer, BartTokenizer
from KGBART_model.modeling_kgbart import KGBartForConditionalGeneration

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


def ascii_print(text):
    text = text.encode("ascii", "ignore")
    print(text)


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--entity_id_path", default=None, type=str)
    parser.add_argument("--entity_relation_embedding_path", default=None, type=str)
    parser.add_argument("--relation_id_path", default=None, type=str)
    parser.add_argument("--max_len_src", default=64, type=int,)
    parser.add_argument("--max_len_tgt", default=64, type=int,)
    
    parser.add_argument("--eval_triple_path", default=None, type=str)
    parser.add_argument("--eval_hist_path", default=None, type=str)
    parser.add_argument("--eval_respone_path", default=None, type=str)
    parser.add_argument("--num_workers", default=5, type=int,
                        help="Number of workers for the data loader.")
    
    parser.add_argument("--bart_model", type=str)
    parser.add_argument("--model_recover_path", default="../../../output/train_unilm_newinital/model.15.bin", type=str,
                        help="The file of fine-tuned pretraining model.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument('--ffn_type', default=0, type=int,
                        help="0: default mlp; 1: W((Wx+b) elem_prod x);")
    parser.add_argument('--num_qkv', default=0, type=int,
                        help="Number of different <Q,K,V>.")
    parser.add_argument('--seg_emb', action='store_true',
                        help="Using segment embedding for self-attention.")
    # decoding parameters
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--amp', action='store_true',
                        help="Whether to use amp for fp16")
    
    parser.add_argument('--subset', type=int, default=0,
                        help="Decode a subset of the input dataset.")
    parser.add_argument("--output_dir", type=str, default="../../../output/train_unilm_newinital/Gen/test",
                        help="output dir")  # "../../../../output/unilm/Gen/model_base.5.bin.test"
    parser.add_argument("--output_file", type=str, default="model.15",
                        help="output file")  # "../../../../output/unilm/Gen/model_base.5.bin.test"
    parser.add_argument('--tokenized_input', action='store_true',
                        help="Whether the input is tokenized.")
    parser.add_argument('--seed', type=int, default=123,
                        help="random seed for initialization")
    parser.add_argument("--do_lower_case", action='store_true',)
    parser.add_argument('--new_segment_ids', default=True, type=bool,
                        help="Use new segment ids for bi-uni-directional LM.")
    parser.add_argument('--new_pos_ids', action='store_true',
                        help="Use new position ids for LMs.")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for decoding.")
    parser.add_argument('--t', type=float, default=None,
                        help='Temperature for sampling')
    parser.add_argument('--p', type=float, default=None,
                        help='p for Nucleus (top-p) sampling')
    parser.add_argument('--beam_size', type=int, default=1,
                        help="Beam size for searching")
    parser.add_argument('--need_score_traces', default=False, type=bool)
    parser.add_argument('--forbid_duplicate_ngrams', default=True, type=bool, )
    parser.add_argument('--forbid_ignore_word', type=str, default=".",
                        help="Forbid the word during forbid_duplicate_ngrams")
    parser.add_argument("--min_len", default=None, type=int)
    parser.add_argument('--ngram_size', type=int, default=3)
    parser.add_argument('--mode', default="s2s",
                        choices=["s2s", "l2r", "both"])
    parser.add_argument('--max_tgt_length', type=int, default=64,
                        help="maximum length of target sequence")
    parser.add_argument('--max_src_length', type=int, default=64,
                        help="maximum length of target sequence")
    parser.add_argument('--s2s_special_token', action='store_true',
                        help="New special tokens ([S2S_SEP]/[S2S_CLS]) of S2S.")
    parser.add_argument('--s2s_add_segment', action='store_true',
                        help="Additional segmental for the encoder of S2S.")
    parser.add_argument('--s2s_share_segment', action='store_true',
                        help="Sharing segment embeddings for the encoder of S2S (used with --s2s_add_segment).")
    parser.add_argument('--pos_shift', action='store_true',
                        help="Using position shift for fine-tuning.")
    parser.add_argument('--length_penalty', type=int, default=2.0)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=3)
    parser.add_argument('--num_return_sequences', type=int, default=1)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    args.output_file = args.output_dir + "/" + args.output_file

    if args.need_score_traces and args.beam_size <= 1:
        raise ValueError(
            "Score trace is only available for beam search with beam size > 1.")
    if args.max_tgt_length >= args.max_seq_length - 2:
        raise ValueError("Maximum tgt length exceeds max seq length - 2.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = BartTokenizer.from_pretrained(
        args.bart_model, do_lower_case=args.do_lower_case)

    # tokenizer.max_len = args.max_seq_length

    pair_num_relation = 0
    bi_uni_pipeline = [
        seq2seq_loader.Preprocess4Seq2seq(list(tokenizer.encoder.keys()), tokenizer.convert_tokens_to_ids,
                                          new_segment_ids=args.new_segment_ids,
                                          truncate_config={'max_len_a': args.max_len_src,
                                                           'max_len_b': args.max_len_tgt,},
                                          mode="s2s", num_qkv=args.num_qkv,
                                          s2s_special_token=args.s2s_special_token,
                                          s2s_add_segment=args.s2s_add_segment,
                                          s2s_share_segment=args.s2s_share_segment,
                                          pos_shift=args.pos_shift)]

    amp_handle = None
    if args.fp16 and args.amp:
        from apex import amp
        amp_handle = amp.init(enable_caching=True)
        logger.info("enable fp16 with amp")

    # Prepare model
    
    sep_token = "</s>"
    cls_token = "<s>"
    pad_token = "<pad>"
    mask_token = "<mask>"
    mask_word_id, eos_word_ids, sos_word_id = tokenizer.convert_tokens_to_ids(
        [mask_token, sep_token, "[S2S_SOS]"])
    
    entity_relation_embedding = np.array(pickle.load(open(args.entity_relation_embedding_path, "rb")))
    emb_depth = entity_relation_embedding.shape[1]
    entity_relation_embedding = np.array(list(np.zeros((4, emb_depth))) + list(entity_relation_embedding))

    model_recover_path = args.model_recover_path.strip()
    logger.info("***** Recover model: %s *****", model_recover_path)
    model_recover = torch.load(model_recover_path)

    model = KGBartForConditionalGeneration.from_pretrained(args.bart_model,
                                                           entity_relation_weight=entity_relation_embedding,
                                                           state_dict=model_recover)

    del model_recover

    if args.fp16:
        model.half()
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    torch.cuda.empty_cache()
    model.eval()
    if "Neural-Path-Hunter" in args.entity_relation_embedding_path:
        strict = False
    else:
        strict = True
        
    dev_dataset = seq2seq_loader.Seq2SeqDataset(
            args.eval_triple_path, args.eval_hist_path, args.eval_respone_path, args.entity_id_path, args.relation_id_path,
            batch_size=args.batch_size, tokenizer=tokenizer,
            max_len_src=args.max_len_src, max_len_tgt=args.max_len_tgt,
            bi_uni_pipeline=bi_uni_pipeline,
        strict=strict)

    
        
        
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size,
                                                     num_workers=args.num_workers,
                                                     collate_fn=seq2seq_loader.batch_list_to_batch_tensors,
                                                     pin_memory=False)
    
    
    print("len(dev_dataset)", len(dev_dataset))
    output_lines = []
    score_trace_list = []
    beam_search_text = []
    total_batch = math.ceil(len(dev_dataset) / args.batch_size)
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Generating", position=0, leave=True)):
            if args.subset and step > args.subset:
                break
                
            batch = [t.to(device) if t is not None else None for t in batch]

            input_ids, input_entity_relation_ids, subword_mask, word_mask, word_subword, decoder_input_ids, decoder_attention_mask, labels = batch
#             print(input_entity_relation_ids)
            
            traces = model.generate(
                input_ids=input_ids,
                input_entity_relation_ids=input_entity_relation_ids,
                attention_mask=subword_mask,
                word_mask=word_mask, word_subword=word_subword,
                max_length=args.max_tgt_length + len(input_ids[0]),
                temperature=args.t,
                top_p=args.p,
                do_sample=False,
                num_return_sequences=1,
                num_beams=args.beam_size,
                no_repeat_ngram_size=args.no_repeat_ngram_size
            )

            # if args.beam_size > 1:
            #     traces = {k: v.tolist() for k, v in traces.items()}
            #     output_ids = traces['pred_seq']
            # else:
                
            output_ids = traces.tolist()
            for w_ids in output_ids:
                w_ids = w_ids[2:]
                output_buf = tokenizer.convert_ids_to_tokens(w_ids)
                output_tokens = []
                for t in output_buf:
                    if t in (sep_token, pad_token):
                        break
                    output_tokens.append(t)
                output_sequence = tokenizer.convert_tokens_to_string(output_tokens)
                output_lines.append(output_sequence)
                
#                 if args.need_score_traces:
#                     score_trace_list[buf_id[i]] = {
#                         'scores': traces['scores'][i], 'wids': traces['wids'][i], 'ptrs': traces['ptrs'][i]}

#                     for b_index in range(len(buf)):
#                         beam_search_trace = traces['wids'][b_index]
#                         beam_search_trace = list(np.transpose(np.array(beam_search_trace)))
#                         each_sentence = []
#                         for ind in range(len(beam_search_trace)):
#                             output_buf = tokenizer.convert_ids_to_tokens(list(beam_search_trace[ind]))
#                             output_tokens = []
#                             for t in output_buf:
#                                 if t in (sep_token, pad_token):
#                                     break
#                                 output_tokens.append(t)
#                             output_sequence = ' '.join(detokenize(output_tokens))
#                             each_sentence.append(output_sequence)
#                         beam_search_text[buf_id[b_index]] = each_sentence

            
       
    with open(args.output_file, "w", encoding="utf-8") as fout:
        for l in output_lines:
            l = re.sub('<s>', '', l)
            l = re.sub('\s+', ' ', l)
            fout.write(l.strip()+"\n")

    if args.need_score_traces:
        with open(fn_out + ".beam.pickle", "wb") as fout_trace:
            pickle.dump(beam_search_text, fout_trace)
        with open(fn_out + ".trace.pickle", "wb") as fout_trace:
            pickle.dump(
                {"version": 0.0, "num_samples": len(dev_dataset)}, fout_trace)
            for x in score_trace_list:
                pickle.dump(x, fout_trace)


if __name__ == "__main__":
    main()
