import os
import torch
import csv
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
from dataset_ODKG import *
from trainer import ConvKGTrainer
from kg_env import BatchKGEnvironment
from model import Model
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
from functools import reduce
import argparse


def filter_paths(paths, probs):
    assert len(paths) == len(probs)
    last_entities = set()
    final_paths = []
    final_probs = []
    for path, prob in zip(paths, probs):
        if path[-1] not in last_entities:
            last_entities.add(path[-1])
            final_paths.append(path)
            final_probs.append(prob)
    return final_paths, final_probs

def remove_incorrect_paths(paths, probs):
    assert len(paths) == len(probs)
    final_paths = []
    final_probs = []
    for path, prob in zip(paths, probs):
        if 0 not in path:
            final_paths.append(path)
            final_probs.append(prob)
    return final_paths, final_probs

def calculate_metrics(paths, probs, true_path):
    # print('paths', paths)
    # print('true_path', true_path)
    paths, probs = remove_incorrect_paths(paths, probs)
    filtered_paths, filtered_probs = filter_paths(paths, probs)
    
    entities = [path[2] for path in filtered_paths]

    # gt = [entityId2entity[entityId] for entityId in true_path]
    recall_path_score = recall_entity_score = 0
    for k, (path, pro) in enumerate(zip(paths, probs)):
        k += 1
        if true_path == path:
            # recall_path_score = 1/k
            recall_path_score = pro
    return recall_path_score
    

def batch_beam_search(model, batch, device, topk=[5, 10, 1], opt={}):
    def batch_action_mask(batch_acts):
        """Return action masks of size [bs, actions]."""
        entity_candidates = batch_acts 
        entity_candidates = pad_sequence(entity_candidates, batch_first=True)
        batch_action_mask = entity_candidates != 0
        return batch_action_mask

    with torch.no_grad():
        batch_dialogue_history, batch_entity_history, batch_true_path = batch[0], batch[1], batch[2]
        true_entity_path = batch_true_path[:, :, 1].tolist()
        true_relation_path = batch_true_path[:, :, 0].tolist()
        env = BatchKGEnvironment(model.graph, opt)

        batch_state = env.reset(batch_entity_history, batch_dialogue_history)  # numpy of [bs, dim]
        batch_dialogue_history, batch_entity_history, batch_path = batch_state
        done = False
        time = 0
        
        dialogue_history = batch_dialogue_history.to(device).clone()
        path_pool = env._batch_path  # list of list, size=bs
        probs_pool = [[] for _ in range(len(path_pool[0]))]
        model.eval()
        for hop in range(3):
            if hop==0:
                batch_relation_actions, batch_entity_actions = env._batch_curr_actions
            else:
                batch_relation_actions, batch_entity_actions = env._batch_get_actions(path_pool, False)

            batch_relation_actions = pad_sequence(batch_relation_actions, batch_first=True).to(device)
            batch_entity_actions = pad_sequence(batch_entity_actions, batch_first=True).to(device)

            relation_path_history, entity_path_history = torch.tensor(path_pool[0], dtype=torch.int64).to(device), torch.tensor(path_pool[1], dtype=torch.int64).to(device)
            actor_logits, _ = model.model.actor(entity_path_history, relation_path_history, batch_entity_actions, batch_relation_actions, dialogue_history)
            action_mask = batch_entity_actions!=0
            actor_logits[~action_mask] = -1e6
            actor_probs = F.softmax(actor_logits, dim=-1)
            
            # print('len(actor_probs[0])', len(actor_probs[0]))
            k = min(topk[hop], len(actor_probs[0]))
            topk_probs, topk_actions = torch.topk(actor_probs, k=k)

            dialogue_history = dialogue_history.clone()
            dialogue_history_pool = []

            new_entity_path_pool = []
            new_relation_path_pool = []
            new_probs_pool = []
            predicted_entities = [batch_entity_actions[i][topk_actions[i]] for i in range(len(batch_entity_actions))]
            predicted_relations = [batch_relation_actions[i][topk_actions[i]] for i in range(len(batch_entity_actions))]
            for i in range(len(path_pool[1])):
                entity_path = path_pool[1][i]
                relation_path = path_pool[0][i]
                probs_path = probs_pool[i]
                predicted_instance_entities = predicted_entities[i]
                predicted_instance_relations = predicted_relations[i]
                instance_probs = topk_probs[i]
                for j in range(k):
                    new_entity_path_pool.append(entity_path+[predicted_instance_entities[j].item()])
                    new_relation_path_pool.append(relation_path+[predicted_instance_relations[j].item()])
                    new_probs_pool.append(probs_path+[instance_probs[j].item()])
                    dialogue_history_pool.append(dialogue_history[i])
            
            dialogue_history = torch.stack(dialogue_history_pool)
            probs_pool = new_probs_pool[:]
            path_pool = new_relation_path_pool[:], new_entity_path_pool[:]

        # new_probs_pool = [[-np.log(p+1e-60) for p in prob] for prob in new_probs_pool]
        rel_probs_pool = new_probs_pool[:]
        
        # new_probs_pool, new_entity_path_pool = fuse_edges(new_probs_pool, new_entity_path_pool)
        new_probs_pool = [reduce(lambda x, y: x*y, probs) for probs in new_probs_pool]
        probs_relation_entity = zip(new_probs_pool, new_entity_path_pool)
        probs_relation_entity = sorted(probs_relation_entity, key=lambda x:x[0], reverse=True)
        probs_relation_entity = zip(*probs_relation_entity)
        probs_relation_entity = [list(a) for a in probs_relation_entity]
        probs_pool , path_pool = probs_relation_entity[0], probs_relation_entity[1]
        true_path = true_entity_path[0]
        # true_path = (true_path[0], true_path[1], true_path[2])
        recall_path_score = calculate_metrics(path_pool, probs_pool, true_path)
        return recall_path_score
        
        
def main(args):
    device = torch.device("cuda")
    st_model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
    st_model.to(device)
    
    
    data_directory = args.data_directory
    opt_dataset = {"entity2entityId": data_directory+"entity2entityId.pkl", "relation2relationId": data_directory+"relation2relationId.pkl",
                        "entity_embeddings": data_directory+"entity_embeddings.pkl", "relation_embeddings": data_directory+"relation_embeddings.pkl",
                        "dialogue2dialogueId": data_directory+"dialogue2dialogueId.pkl", "dialogueId2AlbertRep": data_directory+"dialogueId2SBertRep.pkl",
                        "knowledge_graph": data_directory+"opendialkg_triples.txt",
                        "max_dialogue_history": 3, "triple2tripleId": data_directory+"triple2tripleId.pkl",
                        "test": True}

    
    
    triples = ["Windtalkers<sep> starred actors<sep> Roger Willie"]
    dialogue_history = "Could you recommend movies starring Noah Emmerich?<assistant> Sure, check out Windtalkers, Tumbleweeds, and Trust."
    responses = ["<user> Thanks."]
    ConvKG_dataset_predict = ConvKGDataset_predict(opt_dataset, st_model, triples, dialogue_history, responses)
    self_loop_id = ConvKG_dataset_predict.relation2relationId["self loop"]
    
    opt_model = {"n_entity": len(ConvKG_dataset_predict.entity2entityId)+1, 
                 "n_relation": len(ConvKG_dataset_predict.relation2relationId)+1, 
                 "graph": ConvKG_dataset_predict.graph,
                 "entity2entityId": opt_dataset["entity2entityId"], 
                 "entity_embedding_path": opt_dataset["entity_embeddings"], 
                 "self_loop_id": self_loop_id,
                "entity_embeddings": ConvKG_dataset_predict.entity_embeddings, 
                 "relation_embeddings": ConvKG_dataset_predict.relation_embeddings, 
                 "device": device, 
                 "entity_dim": 768, "relation_dim": 768, "dialogue_dim": 768, 
                 "model_directory": "~/RHO/kg-cruse/codes/KGCruse/models/",
                 "model_name": "Pretrained_Reward_Agent",
                 "batch_size":1, "epochs": 200, "max_path_length": 3, "state_dim": 60, "num_heads": 4, "entropy_weight": 0.1, "gamma": 0.99, "lr": 1e-4,
                 "pretrained_actor_model": "Cloning_Model_actor_400", "clip":5, "alpha": 0.1, 
                 "test": True,
                "pretrained_discriminator_model": "Cloning_Model_discriminator_100", 
                 "pretrained_critic_model": "Cloning_Model_critic_20", 
                 "max_acts": 100000}

    
    pretrain_sd = torch.load(args.load_model_path)
    model = ConvKGTrainer(opt_model).to(opt_model["device"])
    model_sd = model.state_dict()
    model_sd.update(pretrain_sd)
    model.load_state_dict(model_sd)
    model = model.to(opt_model["device"])
    
    
    with open(args.generated_path) as f:
        all_responses = json.load(f)

    
    with open(args.dataset_path) as f:
        reader = csv.reader(f)
        data_lines = list(reader)[1:]
        
    assert len(data_lines) == len(all_responses)

    fout = open(args.output_path, 'w')
    for data_line, responses in tqdm(zip(data_lines, all_responses), total=len(all_responses)):
        responses = list(set(responses))
        if len(responses) == 1:
            fout.write(responses[0].strip()+'\n')
            continue

        history = data_line[2]
        result = re.search('<((user)|(assistant))>', history)
        triples = history[21:result.span()[0]]
        triples = triples.split('<triple>')
        if len(triples) == 1:
            fout.write(responses[0].strip()+'\n')
            continue
            
        dialogue_history = history[result.span()[1]:]

        answer = data_line[3]
        # print(answer)
        # responses += [answer]
        ConvKG_dataset_predict = ConvKGDataset_predict(opt_dataset, st_model, triples, dialogue_history, responses)
        ConvKGDatasetLoaderPredict = DataLoader(ConvKG_dataset_predict, batch_size=1, shuffle=False, num_workers=0, collate_fn=ConvKG_collate)

        # K = [[2, 5, 5], [2, 5, 10], [2, 5, 25], [2, 5, 50], 
        #     [2, 10, 5], [2, 10, 10], [2, 10, 25], [2, 10, 50],
        #     [2, 25, 5], [2, 25, 10], [2, 25, 25], [2, 25, 50], 
        #     [2, 50, 5], [2, 50, 10], [2, 50, 25], [2, 50, 50], ]
        K = [[2, 25, 25]]
        for ks in K:
            # print(ks)
            recall_path_scores = []
            for batch in ConvKGDatasetLoaderPredict:
                recall_path_score = batch_beam_search(model, batch, opt_model["device"], topk = ks, opt=opt_model)
                # print(recall_path_score)
                recall_path_scores.append(recall_path_score)

        zipped = zip(responses, recall_path_scores)
        sort_zipped = sorted(zipped,key=lambda x:x[1], reverse=True)
        sort_responses = [pair[0] for pair in sort_zipped]
        # print(sort_responses)
        fout.write(sort_responses[0].strip()+'\n')

    fout.close()





if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated_path', type=str, required=True,)
    parser.add_argument('--output_path', type=str, required=True,)
    parser.add_argument('--dataset_path', 
                        type=str, 
                        required=True,
                        help='name of the model to be saved')
    parser.add_argument('--data_directory', 
                        type=str, 
                        default="kg-cruse/datasets/")
    parser.add_argument('--load_model_path', type=str)
    
    
    args = parser.parse_args()
    main(args)
