from sklearn.model_selection import train_test_split
import csv
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input folder")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output folder")
    args = parser.parse_args()
    
    
    with open(args.input_dir+'/opendialkg_triples.txt') as f:
        X = list(f.readlines())

    X_train, X_test_val, y_train, y_test_val = train_test_split(X, ['']*len(X), test_size=0.2)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, ['']*len(X_test_val), test_size=0.5)
    
    with open(args.output_dir+'/opendialkg_triples.train', 'w') as fout:
        for l in X_train:
            fout.write(l)
            
    with open(args.output_dir+'/opendialkg_triples.test', 'w') as fout:
        for l in X_test:
            fout.write(l)
    
    with open(args.output_dir+'/opendialkg_triples.valid', 'w') as fout:
        for l in X_val:
            fout.write(l)
        
        
        
    entity2dix, dix2entity = {}, {}
    with open(args.input_dir+'/opendialkg_entities.txt') as f:
        for idx, line in enumerate(f.readlines()):
            line = line.lower()
            assert line not in entity2dix
            entity2dix.update({line.strip(): idx})
            dix2entity.update({idx: line.strip()})
            
    with open(args.output_dir+'/entity2id.txt', 'w') as fout:
        fout.write(str(len(entity2dix))+'\n')
        for key, value in entity2dix.items():
            fout.write(key+'\t'+str(value)+'\n')
            
            
    relation2dix, dix2relation = {}, {}
    with open(args.input_dir+'/opendialkg_relations.txt') as f:
        for idx, line in enumerate(f.readlines()):
            line = line.lower()
            assert line not in relation2dix
            relation2dix.update({line.strip(): idx})
            dix2relation.update({idx: line.strip()})
            
    with open(args.output_dir+'/relation2id.txt', 'w') as fout:
        fout.write(str(len(relation2dix))+'\n')
        for key, value in relation2dix.items():
            fout.write(key+'\t'+str(value)+'\n')
            
            
    for split in ['train', 'valid', 'test']:
        total_num = 0
        outlines = []
        
        with open(args.output_dir+f'/opendialkg_triples.{split}') as f, \
        open(args.output_dir+f'/{split}2id.txt', 'w') as fout:
            for l in f.readlines():
                items = l.lower().strip().split('\t')
                if len(items) == 3:
                    total_num += 1
                    entity1 = str(entity2dix[items[0]])
                    relation = str(relation2dix[items[1]])
                    entity2 = str(entity2dix[items[2]])
                    outlines.append(entity1+'\t'+entity2+'\t'+relation+'\n')
                else:
                    pass
                
            fout.write(str(total_num)+'\n')
            for l in outlines:
                fout.write(l)
                
            
if __name__ == "__main__":
    main()