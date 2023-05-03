import pickle
import numpy as np
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input folder")
    args = parser.parse_args()
    
    with open(args.input_dir+'/rel_embeddings', 'rb') as f:
        rel_embeddings = pickle.load(f)
    with open(args.input_dir+'/ent_embeddings', 'rb') as f:
        ent_embeddings = pickle.load(f)
        
        
    all_list = np.concatenate((ent_embeddings, rel_embeddings), axis=0)
    with open(args.input_dir+'/ent_rel_embeddings', 'wb') as f:
        pickle.dump(all_list, f)
    
if __name__ == "__main__":
    main()