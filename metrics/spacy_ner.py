# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT
# https://github.com/amazon-research/fact-check-summarization
import spacy
import math
import multiprocessing
from tqdm import tqdm
from collections import defaultdict
import json
import re
from spacy.lang.en.stop_words import STOP_WORDS
import random
import argparse
import en_core_web_lg
import numpy as np
    
TRACKING_ENTITY_LIST = ['PERSON', 'FAC', 'GPE', 'ORG', 'NORP', 'LOC', 'EVENT']

def entity_match(ent, source, level=2):
    if level == 0:
        # case sensitive match
        if ent in source:
            return [ent,]
        else:
            return []
    elif level == 1:
        # case insensitive match
        if re.search(re.escape(ent), source, re.IGNORECASE):
            return [ent,]
        else:
            return []
    elif level == 2:
        # split entity and match non-stop words
        ent_split = ent.split()
        result = []
        for l in range(len(ent_split), 1, -1):
            for start_i in range(len(ent_split) - l + 1):
                sub_ent = " ".join(ent_split[start_i:start_i+l])
                if re.search(re.escape(sub_ent), source, re.IGNORECASE):
                    result.append(sub_ent)
            if result:
                break
        if result:
            return result
        else:
            for token in ent_split:
                if token.lower() not in STOP_WORDS or token == "US":
                    if re.search(re.escape(token), source, re.IGNORECASE):
                        result.append(token)
            return result
    return []


def ent_count_match(nlp, base, parent, is_scispacy=False):
    # perform NER on base, then match in parant:
    doc = nlp(base)
#     print(doc.ents)
    ent_count_base = 0
    en_count_in_base_parent = 0
    if is_scispacy:
        for e in doc.ents:
            ent_count_base += 1
            # if e.text in source:
            match_result = entity_match(e.text, parent, 1)
            if match_result:
                en_count_in_base_parent += 1
    else:
        for e in doc.ents:
#             print(e[0].ent_type_)
            if e[0].ent_type_ in TRACKING_ENTITY_LIST:
#                 print("e.text",e.text)
                ent_count_base += 1
                # if e.text in source:
                match_result = entity_match(e.text, parent, 2)
#                 print("match_result", match_result)
                if match_result:
                    en_count_in_base_parent += 1
    return ent_count_base, en_count_in_base_parent




def get_ner_result(nlp, source_file, target_file, generated_file):
    
    
    result = {}
    metric_names = ['precision_source', 'precision_target', 'recall_source', 'recall_target', 'f1_source', 'f1_target']
    for m in metric_names:
        result[m] = []

    with open(source_file) as f_source, open(target_file) as f_target, open(generated_file) as f_generate:
        source_lines = list(f_source.readlines())
        target_lines = list(f_target.readlines())
        generate_lines = list(f_generate.readlines())
        print('total lines:', len(source_lines), len(target_lines), len(generate_lines))
        assert len(source_lines)==len(target_lines)==len(generate_lines)
        
        
        for sline, tline, gline in tqdm(zip(source_lines, target_lines, generate_lines)):
            ent_count_generate, ent_count_generate_source = ent_count_match(nlp, gline, sline)
            ent_count_source, ent_count_source_generate = ent_count_match(nlp, sline, gline)
#             print(ent_count_source_generate, ent_count_generate_source)
#             assert ent_count_source_generate == ent_count_generate_source
            
            ent_count_generate2, ent_count_generate_target = ent_count_match(nlp, gline, tline)
            assert ent_count_generate == ent_count_generate2
            ent_count_target, ent_count_target_generate = ent_count_match(nlp, tline, gline)
#             print(ent_count_target_generate, ent_count_target_generate)
#             assert ent_count_target_generate == ent_count_target_generate

            precision_source = precision_target = recall_target = recall_source = -1
            if ent_count_generate != 0:
                precision_source = ent_count_generate_source * 1.0 / ent_count_generate
                result['precision_source'].append(precision_source)
                precision_target = ent_count_generate_target * 1.0 / ent_count_generate
                result['precision_target'].append(precision_target)

                
            if ent_count_target != 0:
                recall_target = ent_count_target_generate * 1.0 / ent_count_target
                result['recall_target'].append(recall_target)
                
            if ent_count_source != 0:
                recall_source = ent_count_source_generate * 1.0 / ent_count_source
                result['recall_source'].append(recall_source)
                
            if precision_source != -1 and recall_source != -1:
                if precision_source or recall_source:
                    f1_source = 2 * (precision_source * recall_source) / (precision_source + recall_source)
                else:
                    f1_source = 0
                result['f1_source'].append(f1_source)
                
            if precision_target != -1 and recall_target != -1:
                if precision_target or recall_target:
                    f1_target = 2 * (precision_target * recall_target) / (precision_target + recall_target)
                else:
                    f1_target = 0
                result['f1_target'].append(f1_target)
                
                
        
    for m in result.keys():
        result[m] = np.mean(result[m])
        
        
    return result
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_file", type=str, )
    parser.add_argument("--target_file", type=str, )
    parser.add_argument("--generated_file", type=str, )
    args = parser.parse_args()
    
    nlp = en_core_web_lg.load()
    source_result = get_ner_result(nlp, args.source_file, args.target_file, args.generated_file)
    print("source_result")
    print(source_result)
    