from __future__ import print_function, division
import sys
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import torch

from dataset_ODKG import ConvKGDataset, ToTensor, ConvKG_collate
from trainer import ConvKGTrainer

import argparse

from skimage import io, transform
import numpy as np
import pickle
import glob
from tqdm import tqdm

from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils import load_pickle_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    data_directory = args.data_directory
    if args.mode == 'sentence':
        entity_embeddings_file = "sent_entity_embeddings.pkl"
        relation_embeddings_file = "sent_relation_embeddings.pkl"
    elif args.mode == 'kg':
        entity_embeddings_file = "kg_entity_embeddings.pkl"
        relation_embeddings_file = "kg_relation_embeddings.pkl"
    elif args.mode == 'add':
        entity_embeddings_file = "sent+kg_entity_embeddings.pkl"
        relation_embeddings_file = "sent+kg_relation_embeddings.pkl"
    elif args.mode == 'average':
        entity_embeddings_file = "aver_sent+kg_entity_embeddings.pkl"
        relation_embeddings_file = "aver_sent+kg_relation_embeddings.pkl"
        
        
        
    opt_dataset_train = {"entity2entityId": data_directory+"entity2entityId.pkl", 
                         "relation2relationId": data_directory+"relation2relationId.pkl",
                         "entity_embeddings": data_directory+entity_embeddings_file, 
                         "relation_embeddings": data_directory+relation_embeddings_file,
                         "dialogue2dialogueId": data_directory+"dialogue2dialogueId.pkl", 
                         "dialogueId2AlbertRep": data_directory+"dialogueId2SBertRep.pkl",
                         "dataset": data_directory+"splits/split_1/dataset_train.pkl", 
                         "knowledge_graph": data_directory+"opendialkg_triples.txt", "device": device,
                         "max_dialogue_history": 3, "test": False}

    # Dataset Preparation
    ConvKG_dataset_train = ConvKGDataset(opt=opt_dataset_train, transform=transforms.Compose([ToTensor(opt_dataset_train)]))
    self_loop_id = ConvKG_dataset_train.relation2relationId["self loop"]
    
    os.makedirs(args.model_directory, exist_ok=True)

    opt_model = {"n_entity": len(ConvKG_dataset_train.entity2entityId)+1, 
                "n_relation": len(ConvKG_dataset_train.relation2relationId)+1, 
                "graph": ConvKG_dataset_train.graph,
                "entity2entityId": opt_dataset_train["entity2entityId"], 
                "entity_embedding_path": opt_dataset_train["entity_embeddings"], 
                "self_loop_id": self_loop_id,
                "entity_embeddings": ConvKG_dataset_train.entity_embeddings, 
                "relation_embeddings": ConvKG_dataset_train.relation_embeddings, 
                "device": device, 
                "entity_dim": 768, 
                "relation_dim": 768, 
                "dialogue_dim": 768, 
                "model_directory":args.model_directory, 
                "model_name": args.model_name,
                "batch_size":args.batch_size, 
                "epochs": args.epochs, 
                "max_path_length": 3, 
                "state_dim": 60, 
                "gamma": 0.99, 
                "lr": args.lr, 
                "clip": 5, 
                "max_acts": args.max_acts,
                "test": False
                }

    ConvKGDatasetLoaderTrain = DataLoader(ConvKG_dataset_train, batch_size=opt_model["batch_size"], shuffle=True, num_workers=0, collate_fn=ConvKG_collate)
    
    trainer = ConvKGTrainer(opt_model)
    trainer.to(device)    
    if args.load_model_path:
        model_sd = trainer.state_dict()
        pretrain_sd = torch.load(args.load_model_path)
        model_sd.update(pretrain_sd)
        trainer.load_state_dict(model_sd)
        trainer.to(device)
        
    trainer.train_model(ConvKGDatasetLoaderTrain)    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', 
                        type=str, 
                        required=True,
                        help='name of the model to be saved')
    parser.add_argument('--data_directory', 
                        type=str, 
                        required=True,
                        help='Dataset Directory')
    parser.add_argument('--mode', 
                        type=str, 
                        required=True,)
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--max_acts', 
                        type=int, 
                        required=True,
                        help='Maximum number of utgoing edges to a node. Set it to 100000 for OpendialKG dataset')
    parser.add_argument('--batch_size', type=int,)
    parser.add_argument('--epochs', type=int,)
    parser.add_argument('--model_directory', type=str,  default='models')
    parser.add_argument('--lr', type=float, default=1e-4)
    
    args = parser.parse_args()
    main(args)
