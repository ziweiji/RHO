import os
from tqdm import tqdm
import json
import re
import string
from sentence_transformers import SentenceTransformer, util
from flair.data import Sentence
from flair.models import SequenceTagger
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# embedder = SentenceTransformer('paraphrase-distilroberta-base-v2')
# tagger = SequenceTagger.load('ner')
def re_trans(text, ignore=False):
    text = re.sub(r"([.^$*+\\\[\]])", r"\\\1", text)
    text = fr"\W{text}\W"#不能是\b因为entity结尾可能为非字母 那么|b后面要接字母
    
    if ignore:
        return re.compile(text, re.IGNORECASE)
    else:
        return re.compile(text)
    
    
def re_sub_trans(before, after, text):
    before = re.sub(r"([.^$*+\\\[\]])", r"\\\1", before)
    before = fr"(\W)({before})(\W)"#不能是\b因为entity结尾可能为非字母 那么|b后面要接字母\
    try:
        r =  re.compile(before)
    except:
        print(before, after, text)
    # print(r)
    
    if re.match('\d', after):
        pass
        # with open("shuzi.txt", 'a') as f:
        #     f.write(before+'\t'+after+'\t'+text+'\n')
        
        # print("before", before)
        # print("after", after)
        # print(text)
        # after = fr"\1%{after}\3"
        # text = re.sub(r, after, text)
        # text = re.sub("%", "", text)
    else:
        after = fr"\1{after}\3"
        text = re.sub(r, after, text)
        
    
    return text

    
def match_one_entity(entity, text, embedder, tagger):
    results = []
    for m in re.finditer(re_trans(entity), text):#不会重叠
        results.append([m.group()[1:-1], m.start()+1, m.end()-1])
        # print('first')
    if results:
        return results, []
    
    for m in re.finditer(re_trans(entity, ignore=True), text):
        results.append([m.group()[1:-1], m.start()+1, m.end()-1])
    if results:
        return results, []
    
    buchong = re.search("\s*\(.*?\)\s*",entity)
    if buchong:#有括号
        buchong = re.sub("\s*\(", "",buchong.group())
        buchong = re.sub("\)\s*", "",buchong)
        entity2 = re.sub("\s*\(.*?\)\s*", "", entity).strip()
        results = kuohao(entity2, buchong, text)
        if results:
            return results, []

    replace_list = use_NER(entity, text, embedder, tagger)
    return [], replace_list

def kuohao(entity, buchong, text):
    results = []
    # '''['"]*'''+entity+'''['"]*'''
    for m in re.finditer(re_trans(entity), text):
        results.append([m.group()[1:-1], m.start()+1, m.end()-1])
    if not results:
        for m in re.finditer(re_trans(entity, ignore=True), text):
            results.append([m.group()[1:-1], m.start()+1, m.end()-1])
            
    if results:#如果entity有匹配 看括号里的配不配
        # print("results", results)
        results_buchong = []
        for m in re.finditer(re_trans(buchong, ignore=True), text):
            results_buchong.append([m.group()[1:-1], m.start()+1, m.end()-1])
            
        return merge_kuohao(results, results_buchong, text)
    
    return results




def merge_kuohao(results, results_buchong, text):
    # print('merge_kuohao')
    if not results_buchong:
        return results
    # print(results, results_buchong)
    results2 = []
    for r in results:
        flag = False
        for rb in results_buchong:
            if -1<rb[1]-r[2]<4:#buchong在后
                flag = True
                r2 = [text[r[1]:rb[2]], r[1], rb[2]]
                results2.append(r2)
                break#不会那么离谱好多补充都能配得上吧。。。
            elif -1<r[1]-rb[2]<4:#buchong在前
                flag = True
                r2 = [text[rb[1]:r[2]], rb[1], r[2]]
                results2.append(r2)
                break
                
        if not flag:#都没配上
            results2.append(r)
                
    return results2
                
            
        
def match_entities(entities, text, embedder, tagger, if_replace=True):
    all_replace_list = []
    all_results = {}
    for i, entity in enumerate(entities):
        results, replace_list = match_one_entity(entity, text, embedder, tagger)
        if results:
            all_results.update({entity: results})
        all_replace_list += replace_list
            
    all_replace_list = set(all_replace_list)
    if if_replace and all_replace_list:#改的话 重新匹配
        for replace_list in all_replace_list:
            # print(replace_list)
            text = re_sub_trans(replace_list[0], replace_list[1], text)

        all_results = {}
        for i, entity in enumerate(entities):#重新匹配 第一种情况
            results, _ = match_one_entity(entity, text, embedder, tagger)
            if results:
                all_results.update({entity: results})
                
        return text, all_results
    else:#不改的话 直接输出
        return text, all_results
        






SIMILARITY = 0.75
def use_NER(entity, text, embedder, tagger):
    entity_embed = embedder.encode(entity)
    entity_len = len(entity.split())
    
    replace_list = []
    for turn in re.split("<.*?>", text):
        if turn.strip():
            sentence = Sentence(turn)
            tagger.predict(sentence)
            for entity_text in sentence.get_spans('ner'):
                score = util.pytorch_cos_sim(entity_embed, embedder.encode(entity_text.text))
                # print("score", score, entity_text.text)
                if score > SIMILARITY:#match
                    # print("similar")
                    # print(entity_len, len(entity_text.text.split()))
                    # if abs(len(entity_text.text.split())-entity_len)/entity_len<0.5:#长度不要差太多
                    replace_list.append((entity_text.text, entity))
    return replace_list



    
def get_tokenized_idx(text, all_results, tripleid, tokenizer, mod):
    original = tokenizer.encode(text)[1:-1]
    # print(original, len(original))
    
    all_results = sorted(all_results.items(), key=lambda items: len(tokenizer.encode(items[0])), reverse=True)
    history = []
    all_idxs = {}
    for entity, results in all_results:
        entity_id = tripleid.get_triple_id(entity, False)
        # print(entity, entity_id)
        total = [0]*len(original)
        for res in results:
            s = res[1]
            e = res[2]
            
            if text[s-1] == " ":
                s2 = s-1
            elif text[s-2] == " ":
                s2 = s-2
            elif text[s-3] == " ":
                s2 = s-3
            else:
                s2 = s
                
                
            if text[e] in [" ", "<"]:
                e2 = e
            elif len(tokenizer.encode(text[s2:e]))==len(tokenizer.encode(text[s2:e+1])):
                e2 = e+1
            else:
                e2 = e
                
            if justify_overlap(s2, e2, history):
                continue
                
            history.append((s2, e2))
            front = tokenizer.encode(text[:s2])[1:-1]#entity前面有“”号
            entity_idxs = tokenizer.encode(text[s2:e2])[1:-1]
            behind = tokenizer.encode(text[e2:])[1:-1]
                
                
            if original!= front+entity_idxs+behind:
                print(text)
                print(all_results)
                print("original", original)
                print("front", front)
                print("entity_idxs", entity_idxs)
                print("behind", behind)
                print("res", res)
                print("s2, e2", s2, e2)
                
                
            assert original == front+entity_idxs+behind
            
            if mod=='all':
                total[len(front):len(front)+len(entity_idxs)] = [entity_id]*len(entity_idxs)
            elif mod=='first':
                total[len(front)] = entity_id
            else:
                assert False
            # print("total", total)
        all_idxs.update({entity: total})
            
    return all_idxs

    
def justify_overlap(s, e, history):
    assert e > s
    for s1, e1 in history:
        if not (e1<s or s1>e):
            return True
    return False    

remove_nota = u'[’·°–!"#$%&\'()*+,-./:;<=>?@，。?★、…【】（）《》？“”‘’！[\\]^_`{|}~]+'
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
def filter_str(sentence):
    sentence = re.sub(remove_nota, '', sentence)
    sentence = sentence.translate(remove_punctuation_map)
    return sentence.strip()
    
    
def only_english(s):
    s = filter_str(s)
    result = []
    s = re.sub('[0-9]', '', s).strip()
    if not s:
        return True
    
    other_words = re.compile(u"[\u4e00-\u9fa5\uac00-\ud7ff\u30a0-\u30ff\u3040-\u309f]+")
    if re.findall(other_words, s):
        return False
    
    # unicode english
    # res = re.sub('[a-zA-Z]', '', s).strip()
    re_words = re.compile(u"[a-zA-Z]")
    res = re.findall(re_words, s)  # 查询出所有的匹配字符串
    if res:
        return True
    else:
        return False
    
    
class Triple_id():
    def __init__(self, file_entity_id, file_relation_id):
        with open(file_entity_id, "r", encoding='utf-8') as f_entity, \
        open(file_relation_id, "r", encoding='utf-8') as f_relation:
            entity_id = {}
            id_entity = {}
            add_tokens = ["<pad>"]
            for i, token in enumerate(add_tokens):
                entity_id[token] = i
                id_entity[i] = token

            entity_line = f_entity.readlines()
            for i, entityid in enumerate(entity_line):
                if i == 0: 
                    num_entity = int(entityid.strip())+len(add_tokens)
                    continue
                items = entityid.split('\t')
                assert len(items) == 2
                assert items[0] not in entity_id.keys()
                entity_id[items[0]] = int(items[1])+len(add_tokens)
                id_entity[int(items[1])+len(add_tokens)] = items[0]

            assert num_entity == len(id_entity.keys())
            self.num_entity = num_entity
            self.entity_id = entity_id
            self.id_entity = id_entity

            relation_id = {}
            id_relation = {}
            relation_line = f_relation.readlines()
            for i, relationid in enumerate(relation_line):
                if i == 0: 
                    num_relation = int(relationid.strip())
                    continue
                items = relationid.split('\t')
                assert len(items) == 2
                assert items[0] not in relation_id.keys()
                relation_id[items[0]] = int(items[1])
                id_relation[int(items[1])] = items[0]
                
            assert num_relation == len(id_relation.keys())
            self.id_relation = id_relation
            self.relation_id = relation_id
            
            #不能合并 因为relation和entity有重叠


    def get_triple_id(self, word, relation=True):
        if relation:
            return self.relation_id[word.lower()] + self.num_entity
        else:
            return self.entity_id[word.lower()]
        
        
def delete_response_part(hist_response_ids, hist, response, tokenizer):
    if not hist_response_ids:
        hist_ids = tokenizer.encode(hist)[1:-1]
        return [0]*len(hist_ids)
    
    all_entityids = []
    for entity, entityids in hist_response_ids.items():
        all_entityids.append(np.array(entityids))
    hist_response_ids = list(np.sum(all_entityids, axis = 0))
    
    hist_ids = tokenizer.encode(hist)[1:-1]
    response_ids = tokenizer.encode(response)[1:-1]
    
    # print(len(hist_response_ids), len(hist_ids), len(response_ids))
    assert len(hist_response_ids) == len(hist_ids)+1+len(response_ids)
    return hist_response_ids[:len(hist_ids)]
    