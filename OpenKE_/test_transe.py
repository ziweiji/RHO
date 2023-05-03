import argparse
import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import os
import torch
import pickle
import re
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--ent_tot", type=int, default=4835821)
    parser.add_argument("--rel_tot", type=int, default=2063)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=300)
    args = parser.parse_args()
    
    print('start TestDataLoader')
    # dataloader for test
    test_dataloader = TestDataLoader(
        in_path=args.datadir,
        sampling_mode="link")
    
    print("define the model")

    # # define the model
    transe = TransE(
        ent_tot = args.ent_tot,
        rel_tot = args.rel_tot,
        dim = args.dim,
        p_norm = 1,
        norm_flag = True)
    
    
    model = torch.load(args.model_path, map_location=torch.device('cpu'))
    
    if args.save_path:
        for key in ['model.ent_embeddings.weight', 'model.rel_embeddings.weight']:
            n_key = re.sub('model.', '', key)
            model[n_key] = model.pop(key)

        for key in ['loss.zero_const', 'loss.pi_const', 'loss.margin', "model.pi_const", "model.zero_const"]:
            del model[key]
        torch.save(model, args.save_path)
        
        
    transe.load_state_dict(model)
    torch.cuda.empty_cache()
    transe.eval()
    
    print("test the model")
    tester = Tester(model=transe, 
                    data_loader=test_dataloader,
                    use_gpu = True)
    tester.run_link_prediction(type_constrain = False)
    # tester.run_triple_classification()

    
    
if __name__ == "__main__":
    main()