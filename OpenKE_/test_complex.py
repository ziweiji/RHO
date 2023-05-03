import argparse
import openke
from openke.config import Trainer, Tester
from openke.module.model import ComplEx
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import os
import torch
import pickle
import re
import numpy as np


def main():
    datadir = "opendialkg/"
    model_path = "ComplEx_768_result/949.ckpt"
    save_path = ""

    print('start TestDataLoader')
    # dataloader for test
    test_dataloader = TestDataLoader(
        in_path=datadir,
        sampling_mode="link")

    print("define the model")

    # # define the model
    complEx = ComplEx(
        ent_tot = 100813,
        rel_tot = 1358,
        dim = 768,)



    model = torch.load(model_path, map_location=torch.device('cpu'))
    if save_path:
        for key in ['model.ent_re_embeddings.weight', 'model.ent_im_embeddings.weight', 'model.rel_re_embeddings.weight', 'model.rel_im_embeddings.weight']:
            n_key = re.sub('model.', '', key)
            model[n_key] = model.pop(key)

        for key in ['loss.zero_const', 'loss.pi_const', "model.pi_const", "model.zero_const"]:
            del model[key]
        torch.save(model, save_path)


    complEx.load_state_dict(model)
    torch.cuda.empty_cache()
    complEx.eval()

    print("test the model")
    tester = Tester(model=complEx, 
                    data_loader=test_dataloader,
                    use_gpu = True)
    tester.run_link_prediction(type_constrain = False)
    
if __name__ == "__main__":
    main()