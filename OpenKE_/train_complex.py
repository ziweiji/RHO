import openke
from openke.config import Trainer, Tester
from openke.module.model import ComplEx
from openke.module.loss import SoftplusLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import torch
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=300)
    parser.add_argument("--patient", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--outdir", type=str)
    parser.add_argument("--datadir", type=str)
    parser.add_argument("--model_path", type=str, default="")
    
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    
    print('start TrainDataLoader')
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path = args.datadir,
        batch_size = args.batch_size,
        threads = 8,
        sampling_mode = "normal",
        bern_flag = 0,
        filter_flag = 1,
        neg_ent = 64,
        neg_rel = 0)
    
    print('batch_size: ', train_dataloader.get_batch_size())


    print('start TestDataLoader')
    # dataloader for test
    test_dataloader = TestDataLoader(
        in_path=args.datadir,
        sampling_mode="link")
    
    print("define the model")

    # define the model
    complEx = ComplEx(
        ent_tot = train_dataloader.get_ent_tot(),
        rel_tot = train_dataloader.get_rel_tot(),
        dim = args.dim,
    )

    if args.model_path:
        print('load model from ', args.model_path)
        complEx.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
        
        
    # define the loss function
    model = NegativeSampling(
        model = complEx, 
        loss = SoftplusLoss(),
        batch_size = train_dataloader.get_batch_size(), 
        regul_rate = 1.0
    )
    
    if args.patient == -1:
        args.patient = float('inf')
        
        
    trainer = Trainer(model = model, 
                      data_loader = train_dataloader, 
                      train_times = args.epoch, #epoch
                      alpha = args.lr, #learning rate
                      use_gpu = True,
                      opt_method = "adagrad",
                     save_steps = args.save_steps,
                     checkpoint_dir = args.outdir,
                     patient = args.patient,
                     save_num = 1)
    
    trainer.run()
    complEx.save_checkpoint(args.outdir+'/complEx.ckpt')

    # test the model
    complEx.load_checkpoint(args.outdir+'/complEx.ckpt')
    tester = Tester(model = complEx, data_loader = test_dataloader, use_gpu = True)
    tester.run_link_prediction(type_constrain = False)
    
        
if __name__ == "__main__":
    main()