
# Environment settings
```
conda create --name rho python=3.7
conda activate rho
pip install -r requirements.txt
(you need to change the cuda version if necessary)
```
# Dataset
Download the raw data [OpenDialKG](https://github.com/facebookresearch/opendialkg) to local folder ~/RHO/
```
git clone git@github.com:facebookresearch/opendialkg.git
```


# KG Embedding via TransE
Install [OpenKE](https://github.com/thunlp/OpenKE)
```
git clone -b OpenKE-PyTorch https://github.com/thunlp/OpenKE --depth 1
cd OpenKE
cd openke
bash make.sh
```
move all files and folders in ~/RHO/OpenKE_ into ~/RHO/OpenKE

##  Data processing
```
cd ~/RHO/OpenKE

python process_data.py \
--input_dir ~/RHO/opendialkg/data \
--output_dir opendialkg

python n-n.py \
--input_dir "opendialkg/" \
--output_dir "opendialkg" \
--test_path "opendialkg/test2id.txt"
```

## train and test TransE
```
CUDA_VISIBLE_DEVICES=0 python train_transe.py \
--dim 768 \
--lr 0.5 \
--margin 5.0 \
--outdir "TransE_768_result/" \
--datadir ~/RHO/OpenKE/opendialkg \
--save_steps 10 \
--batch_size 4096 \
--epoch 1000 \
--patient 10


CUDA_VISIBLE_DEVICES=0 python test_transe.py \
--dim 768 \
--datadir ~/RHO/OpenKE/opendialkg \
--ent_tot 100813 \
--rel_tot 1358 \
--model_path TransE_768_result/CommonGen_transe.ckpt
```
The commands for other models (such as TransH, ComplEx) are similar.

## concat relation and entity embeddings
```
python merge.py \
--input_dir TransE_768_result
```


# Train RHO

## Data processing
**We offer the processed data files in ~/RHO/src/data.**

run the following commands to process the data:
```
cd  ~/RHO/src/data

python convert_opendialkg.py \
--input_file ~/RHO/opendialkg/data/opendialkg.csv \
--out_file processed_opendialkg.txt

python filter.py \
--input_file processed_opendialkg.txt \
--out_file only_path.txt
 ```

## Build dataset
```
CUDA_VISIBLE_DEVICES=0 python build_dataset.py \
--input_file only_path.txt \
--out_file all_3.csv \
--file_entity_id ~/RHO/OpenKE/opendialkg/entity2id.txt \
--file_relation_id ~/RHO/OpenKE/opendialkg/relation2id.txt

python split.py \
--input_file all_3.csv
```


## Train RHO
```
MODE="input"
DATAMODE="all"
STEP=100
DIM=768
EMBEDPATH=~/RHO/OpenKE/TransE_768_result/ent_rel_embeddings

# IFMEMORY="no_memory"
IFMEMORY='entity_memory'
# need to change --use_memory_bank

IFKG="_no_KG"
# IFKG=''
# need to change --use_kg_embedding

OUTDIR=$MODE"_"$IFMEMORY$IFKG"_output"

CUDA_VISIBLE_DEVICES=0 python \
run_summarization.py \
--use_memory_bank True \
--memory_bank_mode "entity" \
--use_kg_embedding False \
--model_name_or_path "facebook/bart-base" \
--text_column history \
--summary_column response \
--train_file "data/train.csv" \
--validation_file "data/val.csv" \
--entity_relation_embedding_path $EMBEDPATH \
--pad_to_max_length True \
--mode $MODE \
--output_dir $OUTDIR \
--learning_rate 3.5e-5 \
--do_train \
--do_eval \
--evaluation_strategy steps \
--eval_steps $STEP \
--logging_strategy steps \
--logging_steps $STEP \
--logging_first_step \
--logging_dir $OUTDIR \
--max_source_length 800 \
--max_target_length 64 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 8 \
--eval_accumulation_steps 1 \
--early_stop True \
--early_stopping_patience 3 \
--save_total_limit 1 \
--load_best_model_at_end \
--overwrite_cache \
--overwrite_output_dir \
--num_train_epochs 50
```

## Generate

generate candidates for reranking
num_return_sequences > 1
```
CUDA_VISIBLE_DEVICES=0 python generation.py \
--use_memory_bank True \
--memory_bank_mode "entity" \
--use_kg_embedding True \
--model_name_or_path $OUTDIR \
--text_column history \
--summary_column response \
--test_file "data/test.csv" \
--entity_relation_embedding_path $EMBEDPATH \
--pad_to_max_length True \
--mode $MODE \
--num_beams 4 \
--num_return_sequences 4 \
--predict_with_generate True \
--max_source_length 800 \
--max_target_length 64 \
--per_device_eval_batch_size 8 \
--generation_path "rerank/generation_"$GENERTYPE".json"
```

without re-ranking
num_return_sequences=1
```
GENERTYPE="B4"
CUDA_VISIBLE_DEVICES=0 python \
run_summarization.py \
--use_memory_bank True \
--memory_bank_mode "entity" \
--use_kg_embedding True \
--model_name_or_path $OUTDIR \
--text_column history \
--summary_column response \
--test_file "data/test.csv" \
--train_file "data/train.csv" \
--entity_relation_embedding_path $EMBEDPATH \
--pad_to_max_length True \
--mode $MODE \
--output_dir $OUTDIR \
--do_predict \
--num_beams 4 \
--predict_with_generate True \
--max_source_length 800 \
--max_target_length 64 \
--per_device_eval_batch_size 4 \
--generation_file $GENERTYPE"_generated_predictions.txt"
```


## Re-Ranking
Install [KG-CRuSE](https://github.com/rajbsk/kg-cruse) in ~/RHO/
```
git clone git@github.com:rajbsk/kg-cruse.git
```

In ~/RHO/kg-cruse_/codes/KGCruse/predict.py 
change the following path into your own path.
```
"model_directory": "~/RHO/kg-cruse/codes/KGCruse/models/"
```
move all files and folders in ~/RHO/kg-cruse_ into ~/RHO/kg-cruse

### Dataset
Download Sentence Embedding for OpenDialKG from this ["link"](https://drive.google.com/file/d/1pZlmqku2suO1xAlhiS8M2tBwzuF17f2t/view?usp=sharing)  and unzip.

Combine Sentence and KG embedding 
```
cd ~/RHO/kg-cruse/

python build_dataset.py \
--data_dir datasets \
--file_entity_id ~/RHO/OpenKE/opendialkg/entity2id.txt \
--file_relation_id ~/RHO/OpenKE/opendialkg/relation2id.txt \
--kg_embedding_path ~/RHO/OpenKE/TransE_768_result/ent_rel_embeddings
```

### Train and Test Re-ranker
```
cd ~/RHO/kg-cruse/codes/KGCruse

LR=1e-4
ACTS=50000
MODE='add'

CUDA_VISIBLE_DEVICES=0 python3 main.py \
--mode $MODE \
--max_acts $ACTS \
--model_name $MODE \
--data_directory ~/RHO/kg-cruse/datasets/ \
--batch_size 40 \
--lr $LR \
--model_directory models \
--epochs 50


CUDA_VISIBLE_DEVICES=0 python3 tester.py \
--split_id=1 \
--model_name models/<model_name> \
--mode $MODE \
--data_directory ~/RHO/kg-cruse/datasets/ \
--batch_size 1 
```

### Re-Rank (Predict)
```
cd ~/RHO/kg-cruse/codes/KGCruse

RERANKLR="1e-4"
RERANKFOLDER="rerank_"$RERANKLR

CUDA_VISIBLE_DEVICES=0 python predict.py \
--load_model_path models/<model_name> \
--generated_path "rerank/generation_"$GENERTYPE".json" \
--output_path rerank/best.txt \
--dataset_path ~/RHO/src/data/test.csv
```

# Train modified KG-BART (baseline)
The code modify [KG-BART](https://github.com/yeliu918/KG-BART) 
In ~/RHO/KG-BART/KG_grounding/entity_onehot.py
change ```commongend = "~/RHO/KG-BART/dataset/commongen/commongen.dev.src_alpha.txt"``` to your own path.

In ~/RHO/KG-BART/KGBART/KGBART_training/
decode_pretrain.py 
decode_seq2seq.py
pretrain_kgbart.py
run_seq2seq.py
seq2seq_loader.py
change ```sys.path.append('~/RHO/KG-BART/KGBART')``` to your own path.


## Pre-train
```
cd ~/RHO/KG-BART


OUTDIR="Pretraining_output"
CUDA_VISIBLE_DEVICES=1 \
python KGBART/KGBART_training/pretrain_kgbart.py \
--entity_id_path ~/RHO/OpenKE/opendialkg/entity2id.txt \
--entity_relation_embedding_path $EMBEDPATH \
--relation_id_path ~/RHO/OpenKE/opendialkg/relation2id.txt \
--pretrained_train_path opendialkg/train_input.txt \
--pretrained_eval_path opendialkg/val_input.txt \
--bart_model "facebook/bart-base \
--learning_rate 1e-5 \
--output_dir $OUTDIR  \
--log_dir $OUTDIR \
--max_seq_length 64 \
--max_position_embeddings 64 \
--max_len_a 64 \
--max_len_b 64 \
--max_pred 64 \
--train_batch_size 64 \
--eval_batch_size 8 \
--gradient_accumulation_steps 8 \
--warmup_proportion 0.1 \
--label_smoothing 0.1 \
--num_train_epochs 60
```

## fine-tune
```
OUTDIR="seq2seq_output"
CUDA_VISIBLE_DEVICES=0 python KGBART/KGBART_training/run_seq2seq.py \
--entity_id_path ~/RHO/OpenKE/opendialkg/entity2id.txt \
--entity_relation_embedding_path ~/RHO/OpenKE/TransE_768_result/ent_rel_embeddings \
--relation_id_path ~/RHO/OpenKE/opendialkg/relation2id.txt \
--train_triple_path opendialkg/train_input.txt \
--eval_triple_path opendialkg/val_input.txt \
--train_hist_path opendialkg/train_history.txt \
--eval_hist_path opendialkg/val_history.txt \
--train_respone_path opendialkg/train_response.txt \
--eval_respone_path opendialkg/val_response.txt \
--bart_model "facebook/bart-base" \
--model_recover_path "Pretraining_output/best_model/model.best.bin" \
--output_dir $OUTDIR \
--log_dir $OUTDIR \
--do_train --do_eval \
--learning_rate 1e-5 \
--max_len_src 250 --max_len_tgt 64 --max_pred 64 \
--train_batch_size 16 --eval_batch_size 2 \
--gradient_accumulation_steps 8 --learning_rate 0.00001 \
--warmup_proportion 0.1 --label_smoothing 0.1 \
--num_train_epochs 50
```
## Generate
```
MODEDIR="seq2seq_output"
BEAM=4
CUDA_VISIBLE_DEVICES=1 python KGBART/KGBART_training/decode_seq2seq.py \
--entity_id_path ~/RHO/OpenKE/opendialkg/entity2id.txt \
--entity_relation_embedding_path $EMBEDPATH \
--relation_id_path ~/RHO/OpenKE/opendialkg/relation2id.txt \
--eval_triple_path opendialkg/test_input.txt \
--eval_hist_path opendialkg/test_history.txt \
--eval_respone_path opendialkg/test_response.txt \
--max_len_src 250 --max_len_tgt 64 \
--model_recover_path $MODEDIR"/best_model/model.best.bin" \
--bart_model "facebook/bart-base \
--output_dir $MODEDIR \
--output_file "B"$BEAM"_test.txt" \
--beam_size $BEAM \
--batch_size 8 \
--forbid_duplicate_ngrams True

```

**We also offer our generated results in ~/RHO/src/generated_results.**



# Metrics

## FeQA
Install [FeQA](https://github.com/esdurmus/feqa) into local folder ~/RHO/metrics/
```
git clone git@github.com:esdurmus/feqa.git
```
Create a new conda environment for FeQA

## QuestEval
Install [QuestEval](https://github.com/ThomasScialom/QuestEval) into local folder ~/RHO/metrics/


## Entity Coverage
```
python ~/RHO/metrics/spacy_ner.py \
--source_file $FEQASOURCE \
--target_file $TGT \
--generated_file $GENER
```

## BLEU ROUGE
We use HuggingFace [Datasets](https://github.com/huggingface/datasets) to calculate the BLEU and ROUGE scores.


# Bib
```
@inproceedings{ji2023rho,
  title={RHO: Reducing Hallucination in Open-domain Dialogues with Knowledge Grounding},
  author={Ji, Ziwei and Liu, Zihan and Lee, Nayeon and Yu, Tiezheng and Wilie, Bryan and Zeng, Min and Fung, Pascale},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2023},
  pages={4504--4522},
  year={2023}
}
```

