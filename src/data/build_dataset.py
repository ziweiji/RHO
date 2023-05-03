import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import json
import torch
import argparse
import sys
import re
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
import csv
import math
import json
from argparse import ArgumentParser
import ast

from sentence_transformers import SentenceTransformer, util
from flair.data import Sentence
from flair.models import SequenceTagger
from data_utils import *

def build_dataset(tripleid, in_path, out_path, process_rel, mod, max_hist_len, filter_no_english=False):
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
    tokenizer.add_tokens(['<sep>','<triple>','<user>','<assistant>','<response>'])
    embedder = SentenceTransformer('paraphrase-distilroberta-base-v2')
    tagger = SequenceTagger.load('ner')
    
    
    
    with open(in_path) as f_in, open(out_path,"w") as fout:
        writer = csv.writer(fout)
        writer.writerow(['entity_relation_ids', 'memory_bank', 'history','response'])
        pre = 'Given the knowledge:'
        pre_len = len(tokenizer.tokenize(pre))
        
        
        for idx, l_in in tqdm(enumerate(f_in.readlines())):
            if idx%500 == 0:
                print('idx', idx)
            l_in = json.loads(l_in)
            # print(l_in)
            
            entity_relation_ids = [0]*pre_len
            memory_bank = []
            triples = []
            
            if_only_english = True
            all_entities = set()
            for triple in l_in["knowledge_base"]:
                sub, rel, obj = triple
                all_entities.add(sub)
                all_entities.add(obj)
                if process_rel:
                    if '~' in rel:
                        rel = re.sub('~','',rel)
                        sub, obj = obj, sub
                        
                sub_id = tripleid.get_triple_id(sub, False)
                rel_id = tripleid.get_triple_id(rel, True)
                rel = re.sub("[-_]",' ',rel)
                obj_id = tripleid.get_triple_id(obj, False)
                
                memory_bank.append([sub_id, rel_id, obj_id])
                
                sub_len = len(tokenizer.tokenize(' '+sub))
                rel_len = len(tokenizer.tokenize(' '+rel))
                obj_len = len(tokenizer.tokenize(' '+obj))
                if mod == 'all':
                    entity_relation_ids += [sub_id]*sub_len+[0]+[rel_id]*rel_len+[0]+[obj_id]*obj_len+[0]#多加是为了给<triple>留空
                elif mod=='first':
                    entity_relation_ids += [sub_id]+[0]*sub_len+[rel_id]+[0]*rel_len+[obj_id]+[0]*obj_len
                else:
                    assert False
                if filter_no_english:
                    if_only_english = if_only_english and only_english(sub) and only_english(rel) and only_english(obj)
                              
                triple2 = sub+'<sep> '+rel+'<sep> '+obj
                triples.append(triple2)
                
            entity_relation_ids = entity_relation_ids[:-1]
            # print('entity_relation_ids', entity_relation_ids)
                
            
            input_text = pre+' '+'<triple> '.join(triples)
            if not if_only_english:#test data 不能直接删去
                print('no only', idx, input_text)
                continue
                    
            if max_hist_len:            
                min_turn_i = len(l_in["history"]) - max_hist_len
            else:
                min_turn_i = 0   
            l_hist = ""
            for turn_i, (speaker, turn) in enumerate(l_in["history"]):
                if turn_i < min_turn_i:
                    continue
                else:
                    assert speaker in ["user", "assistant"]
                    speaker = '<'+speaker+'> '
                    l_hist += speaker + turn
                              
                        
            speaker, response = l_in["response"]
            assert speaker in ["user", "assistant"]
            speaker = '<'+speaker+'> '
            l_hist_response = l_hist+"<response>"+speaker+response
            # print("l_hist_response", l_hist_response)
            l_hist_response, all_results = match_entities(all_entities, l_hist_response, embedder, tagger, if_replace=True)
            # print("l_hist_response2", l_hist_response)
            # print(all_results)
            
            
            hist_response_entity_relation_ids = get_tokenized_idx(l_hist_response, all_results, tripleid, tokenizer, mod)
            # print('hist_response_entity_relation_ids', hist_response_entity_relation_ids)
            
            l_hist, l_response = l_hist_response.split("<response>")
            hist_entity_relation_ids = delete_response_part(hist_response_entity_relation_ids, l_hist, l_response, tokenizer)           
            # print('hist_entity_relation_ids', hist_entity_relation_ids)
            
            input_text += l_hist
            entity_relation_ids += hist_entity_relation_ids
            entity_relation_ids = [0]+entity_relation_ids+[0]#<s>和</s>
            
            # print("final entity_relation_ids", entity_relation_ids)
            # print("final input_text", input_text)
            # print("final input_ids", tokenizer.tokenize(input_text))
            
            # assert len(entity_relation_ids) == len(tokenizer.encode(input_text))
            if len(entity_relation_ids) != len(tokenizer.encode(input_text)):
                print(len(entity_relation_ids), len(tokenizer.encode(input_text)))
                print("entity_relation_ids", entity_relation_ids)
                print("input_ids", tokenizer.tokenize(input_text))
                print("input_text", input_text)
                assert False
                
                
            
            writer.writerow([entity_relation_ids,
                             memory_bank,
                             input_text,
                             l_response])
            
            

def check_entity_id(path):
    with open(path) as f:
        reader = csv.reader(f)
        next(reader, None)
        for i, row in enumerate(reader):
            entity_ids = set(ast.literal_eval(row[0]))
            
            entity_ids2 = set([0])
            for memory_bank in ast.literal_eval(row[1]):
                entity_ids2 |= set(memory_bank)
                
            if entity_ids != entity_ids2:
                print(i)
                print(entity_ids)
                print(entity_ids2)
                
            # if i > 10:
            #     break
            # break
            assert entity_ids == entity_ids2
            
            
def main():
    parser = ArgumentParser()
    parser.add_argument("--file_entity_id", type=str, required=True, help="Path to entity_id")
    parser.add_argument("--file_relation_id", type=str, required=True, help="Path to relation_id")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input file")
    parser.add_argument("--out_file", type=str, required=True, help="Path to the output csv file")
    args = parser.parse_args()
    tripleid = Triple_id(args.file_entity_id, args.file_relation_id)
    
    
    max_hist_len = 3
    build_dataset(tripleid=tripleid,
                  in_path=args.input_file,
                  out_path=args.out_file,
                 process_rel='process_rel', 
                  mod="all",
                  max_hist_len=max_hist_len)
    check_entity_id(args.out_file)

    
if __name__ == "__main__":
    main()

