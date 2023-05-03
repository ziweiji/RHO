# from Neural-Path-Hunter
import csv
import json
from argparse import ArgumentParser
from typing import Any, Dict, Iterable, Tuple

from tqdm import tqdm
import spacy
import re

def read_csv(data_file: str) -> Iterable[Tuple[str, int]]:
    with open(data_file, "r") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)  # skip header row
        dialog_id = 0
        for i, row in enumerate(reader):
            dialog_id += 1
            dialogue, _, _ = row[0].strip(), row[1].strip(), row[2].strip()

            yield dialogue, dialog_id


def parse_message(dialogue: str, dialog_id: int) -> Iterable[Dict[str, Any]]:
    json_dialog = json.loads(dialogue)
    history = []
    metadata = {}
    for i, turn in enumerate(json_dialog):
        if i == 0:
            if "message" in turn:
                history.append((turn["sender"], process(turn["message"])))
        else:
            if "metadata" in turn:
                if "path" in turn["metadata"]:
                    metadata = turn["metadata"]["path"][1]
            else:
                response = process(turn["message"])
                yield {
                    "history": history,
                    "response": [turn["sender"], response],
                    "knowledge_base": metadata,
                    "dialogue_id": dialog_id,
                }

                metadata = {}
                history.append((turn["sender"], response))

def process(text):
    text = re.sub("\s+", " ", text)
    text = re.sub(r"(\b)(D|d)(o)(es)?(nt)(\b)", r"\1\2\3\4n't\6", text)
    text = re.sub(r"(\b)(D|d)(idnt)(\b)", r"\1\2idn't\4", text)
    
    text = re.sub(r"(\b)(C|c)(ant)(\b)", r"\1\2an't\4", text)
    text = re.sub(r"(\b)(A|a)(rent)(\b)", r"\1\2ren't\4", text)
    
    text = re.sub(r"(\b)(i)(\b)", r"\1I\3", text)
    text = re.sub(r"(\b)(I|i)(snt)(\b)", r"\1\2sn't\4", text)
    text = re.sub(r"(\b)(w|W)(asnt)(\b)", r"\1\2asn't\4", text)
    text = re.sub(r"(\b)(w|W)(erent)(\b)", r"\1\2eren't\4", text)
    
    text = re.sub(r"(\b)(I|i)(m)(\b)", r"\1I'm\4", text)
    text = re.sub(r"(\b)(Ill)(\b)", r"\1I'll\3", text)
    text = re.sub(r"(\b)(I|i)(ve)(\b)", r"\1I've\4", text)
    text = re.sub(r"(\b)(ha)((ve)|s)(nt)(\b)", r"\1ha\3n't\6", text)
    
    text = re.sub(r"(\b)(DId)(\b)", r"\1Did\3", text)
    text = re.sub(r"(\b)(Iknow)(\b)", r"\1I know\3", text)
    
    return text.strip()
    
    
def convert(data_file: str, out_file: str):
    with open(out_file, "w") as out:
        for dialogue, dialog_id in tqdm(read_csv(data_file)):
            for utterance in parse_message(dialogue, dialog_id):
                out.write(json.dumps(utterance) + "\n")


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input file")
    parser.add_argument("--out_file", type=str, required=True, help="Path to the output file")
    args = parser.parse_args()

    convert(args.input_file, args.out_file)


if __name__ == "__main__":
    main()
