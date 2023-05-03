from sklearn.model_selection import train_test_split
import csv
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input file")
    args = parser.parse_args()

    with open(args.input_file) as f:
        lines = list(csv.reader(f))
        head = lines[0]
        X = lines[1:]

    X_train, X_test_val, y_train, y_test_val = train_test_split(X, ['']*len(X), test_size=0.2)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, ['']*len(X_test_val), test_size=0.5)

    
    with open(f'train.csv', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerow(head)
        for l in X_train:
            writer.writerow(l)

    with open(f'test.csv', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerow(head)
        for l in X_test:
            writer.writerow(l)

    with open(f'val.csv', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerow(head)
        for l in X_val:
            writer.writerow(l)
            
if __name__ == "__main__":
    main()