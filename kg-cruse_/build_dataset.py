import csv
import re
import pickle
import os
import numpy as np
import torch
from argparse import ArgumentParser

# from ~/RHO/src/data/data_utils.py
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
            if word == 'Suzanne Collin':
                word = 'Suzanne Collins'
            return self.entity_id[word.lower()]
        
        
def main():
    parser = ArgumentParser()
    parser.add_argument("--file_entity_id", type=str, required=True)
    parser.add_argument("--file_relation_id", type=str, required=True)
    parser.add_argument("--kg_embedding_path", type=str, required=True, help="Path to the input file")
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    tripleid = Triple_id(args.file_entity_id, args.file_relation_id)
    
    
    with open(args.data_dir+'/sent_entity_embeddings.pkl', 'rb') as f:
        sent_entity_embeddings = pickle.load(f)
    
    with open(args.data_dir+'/entity2entityId.pkl', 'rb') as f:
        entity2entityId = pickle.load(f)

    with open(args.data_dir+'/entityId2entity.pkl', 'rb') as f:
        entityId2entity = pickle.load(f)

#     print(len(sent_entity_embeddings), len(entity2entityId), len(entityId2entity))
    
    with open(args.kg_embedding_path, 'rb') as f:
        ent_rel_embeddings = pickle.load(f)
        
        
    all_add_embeddings = [torch.Tensor([0]*768)]
    for idx, (entity, entityId) in enumerate(entity2entityId.items()):
        entity_id = tripleid.get_triple_id(entity, False)-1#去掉<pad>
        txtual = sent_entity_embeddings[idx+1]
        kg = torch.Tensor(ent_rel_embeddings[entity_id])
        all_add_embeddings.append(txtual+kg)

    all_add_embeddings = torch.stack(all_add_embeddings, 0)
    with open(args.data_dir+"/sent+kg_entity_embeddings.pkl", 'wb') as f:
        pickle.dump(all_add_embeddings, f)
        
        
        
    with open(args.data_dir+'/sent_relation_embeddings.pkl', 'rb') as f:
        sent_relation_embeddings = pickle.load(f)

    with open(args.data_dir+'/relation2relationId.pkl', 'rb') as f:
        relation2relationId = pickle.load(f)

    with open(args.data_dir+'/relationId2relation.pkl', 'rb') as f:
        relationId2relation = pickle.load(f)

#     print(len(sent_relation_embeddings), len(relation2relationId), len(relationId2relation))

    all_merged_embeddings = [torch.Tensor([0]*768)]
    for idx, (relation, relationId) in enumerate(relation2relationId.items()):
        txtual = sent_relation_embeddings[idx+1]
        if relation == 'self loop':
            kg = torch.Tensor([0]*768)
        else:   
            relation_id = tripleid.get_triple_id(relation, True)-1#去掉<pad>
            kg = torch.Tensor(ent_rel_embeddings[relation_id])
        all_merged_embeddings.append(txtual+kg)

    all_merged_embeddings = torch.stack(all_merged_embeddings, 0)
    with open(args.data_dir+"/sent+kg_relation_embeddings.pkl", 'wb') as f:
        pickle.dump(all_merged_embeddings, f)

            
if __name__ == "__main__":
    main()