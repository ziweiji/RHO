import json
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input file")
    parser.add_argument("--out_file", type=str, required=True, help="Path to the output file")
    args = parser.parse_args()

    with open(args.input_file) as f, open(args.out_file,'w') as fout:
        for i, l in enumerate(f.readlines()):
            json_dialog = json.loads(l)
            knowledge_base = json_dialog["knowledge_base"]
            if knowledge_base:
                flag = True
                for tripes in knowledge_base:
                    assert len(tripes) == 3
                    if not (tripes[0] and tripes[1] and tripes[2]):
                        flag = False
                        # print(i, knowledge_base)

                if flag:
                    fout.write(l)



if __name__ == "__main__":
    main()