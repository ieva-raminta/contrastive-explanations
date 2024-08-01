import torch
import torch.nn as nn
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
    LayerConductance, 
    LayerIntegratedGradients,
)
import numpy as np
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup, BertModel, AutoModel, LongformerModel
import pickle
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import json 
from pprint import pprint
import random 
from scipy.stats import spearmanr
from scipy.stats import kendalltau

torch.manual_seed(123)
np.random.seed(123)

model_directory_path = "/home/irs38/Negative-Precedent-in-Legal-Outcome-Prediction/results/Outcome/joint_model/longformer/facts/be7001606980465a883b6119ced5fb0d/"
model_path = model_directory_path+"model.pt"
model = torch.load(model_path)
MODEL_NAME="allenai/longformer-base-4096"
model.eval()
model.zero_grad()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenized_dir = "/home/irs38/Negative-Precedent-in-Legal-Outcome-Prediction/ECHR/Outcome/longformer"

def flatten_list(l):
    return [item for sublist in l for item in sublist]

def make_loader(input, mask, labels, claims, train=True):
    labels = torch.tensor(labels)
    claims = torch.tensor(claims)
    data = TensorDataset(input, mask, labels, claims)
    if train:
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=1)
    return dataloader

def predict(b_input_ids, b_attn_mask=None, global_attention_mask=None, b_claims=None):
    logits, _ = model(b_input_ids.cuda(), b_attn_mask.cuda(), global_attention_mask, b_claims) #predictor.predict_json(e)
    return logits

def forward_func(b_input_ids, b_attn_mask=None, global_attention_mask=None, b_claims=None, article_id=0):
    logits = predict(b_input_ids, b_attn_mask, global_attention_mask, b_claims)
    logits = logits.reshape(b_input_ids.shape[0], -1, 3)
    logits = logits[:,article_id,:]
    out = logits.max(1).values
    return out

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions

with open(tokenized_dir + "/tokenized_dev.pkl", "rb") as f:
            val_facts, val_masks, val_arguments, \
            val_masks_arguments, val_ids, val_claims, val_outcomes, _ = pickle.load(f)

with open("/home/irs38/contrastive-explanations/data/ecthr/Chalkidis/train.jsonl", "r") as f:
    train_Chalkidis_data = [json.loads(line) for line in f]
with open("/home/irs38/contrastive-explanations/data/ecthr/Chalkidis/dev.jsonl", "r") as f:
    dev_Chalkidis_data = [json.loads(line) for line in f]
with open("/home/irs38/contrastive-explanations/data/ecthr/Chalkidis/test.jsonl", "r") as f:
    test_Chalkidis_data = [json.loads(line) for line in f]
ids = [item["case_no"] for item in train_Chalkidis_data+dev_Chalkidis_data+test_Chalkidis_data]
exs = train_Chalkidis_data+dev_Chalkidis_data+test_Chalkidis_data

max_len=512
test_size = 100000
val_inputs = val_facts
val_inputs, val_masks = val_inputs[:test_size, :, :max_len], val_masks[:test_size, :, :max_len]
neg_val_labels = val_claims[:test_size, :] - val_outcomes[:test_size, :]
pos_val_labels = val_outcomes[:test_size, :]
pos_val_labels[pos_val_labels < 0] = 0
neg_val_labels[neg_val_labels < 0] = 0
val_labels = np.concatenate((pos_val_labels, neg_val_labels), axis=1)
claim_val_labels = val_claims[:test_size, :]

val_dataloader = make_loader(val_inputs, val_masks, val_labels, claim_val_labels, train=False)

dev_data = []
for step, batch in enumerate(val_dataloader):
    b_input_ids, b_attn_mask, b_labels, b_claims = tuple(t.to("cuda") for t in batch)
    b_input_ids = b_input_ids.squeeze(1)
    b_attn_mask = b_attn_mask.squeeze(1)
    global_attention_mask = torch.zeros(b_input_ids.shape, dtype=torch.long, device="cuda")
    global_attention_mask[:, [0]] = 1
    dev_data.append([b_input_ids, b_attn_mask, b_labels, b_claims, global_attention_mask])

train_gold_rationales = []
with open("/home/irs38/contrastive-explanations/data/ecthr/outcome/train_gold_rationales.txt", "r") as f:
    for line in f:
        train_gold_rationales.append(line.strip())
dev_gold_rationales = []
with open("/home/irs38/contrastive-explanations/data/ecthr/outcome/dev_gold_rationales.txt", "r") as f:
    for line in f:
        dev_gold_rationales.append(line.strip())
test_gold_rationales = []
with open("/home/irs38/contrastive-explanations/data/ecthr/outcome/test_gold_rationales.txt", "r") as f:
    for line in f:
        test_gold_rationales.append(line.strip())
gold_rationales = train_gold_rationales + dev_gold_rationales + test_gold_rationales

train_silver_rationales = []
with open("/home/irs38/contrastive-explanations/data/ecthr/outcome/train_silver_rationales.txt", "r") as f:
    for line in f:
        train_silver_rationales.append(line.strip())
dev_silver_rationales = []
with open("/home/irs38/contrastive-explanations/data/ecthr/outcome/dev_silver_rationales.txt", "r") as f:
    for line in f:
        dev_silver_rationales.append(line.strip())
test_silver_rationales = []
with open("/home/irs38/contrastive-explanations/data/ecthr/outcome/test_silver_rationales.txt", "r") as f:
    for line in f:
        test_silver_rationales.append(line.strip())
silver_rationales = train_silver_rationales + dev_silver_rationales + test_silver_rationales

ids_to_rationales = {}
for id,rationale in zip(ids, silver_rationales):
    ids_to_rationales[id] = rationale
ids_to_ex = {}
for id,ex in zip(ids, exs):
    ids_to_ex[id] = ex
ids_to_gold_rationales = {}
for id,rationale in zip(ids, gold_rationales):
    ids_to_gold_rationales[id] = rationale

interesting_label_options = ["claimed_and_violated", "claimed_not_violated"]
index2label = {0: "not_claimed", 1: "claimed_and_violated", 2: "claimed_not_violated"}

dev_rationales = []
dev_attributions_per_sentence = []
for i,item in enumerate(dev_data): 
    b_input_ids, b_attn_mask, b_labels, b_claims, global_attention_mask = item
    claims = b_claims
    gold = b_labels 
    if ";" not in val_ids[i]:
        gold_id = val_ids[i]
    else:
         for id in val_ids[i].split(";"):
            if id in ids:
                gold_id = id
                break

    ex = ids_to_ex[gold_id]
    facts = ex["facts"]
    sentence_lengths = [len(tokenizer.tokenize(sentence)) for sentence in facts]

    rationale = ids_to_gold_rationales[gold_id]

    if rationale not in ["[]", []]: 
            
        rationales = [int(num) for num in rationale.lstrip("[").rstrip("]").split(",")]
        rationale_binary = [1 if num in rationales else 0 for num in range(len(facts))]
        if any([r for r in rationales if r >= len(facts)]):
            import pdb; pdb.set_trace()

        D_out = int(b_labels.shape[1] / 2)
        y = torch.zeros(b_labels.shape[0], D_out).long().to("cuda")
        y[b_labels[:, :D_out].bool()] = 1
        y[b_labels[:, D_out:].bool()] = 2
        y = y.squeeze(1).squeeze().tolist()

        ref_token_id = tokenizer.pad_token_id
        cls_token_id = tokenizer.cls_token_id
        sep_token_id = tokenizer.sep_token_id
        ref_input_ids = [cls_token_id] + [ref_token_id] * (b_input_ids.shape[-1]-2) + [sep_token_id]
        ref = torch.tensor([ref_input_ids], device="cuda")

        article_id = y.index(2) if 2 in y else y.index(1) if 1 in y else 0

        lig = LayerIntegratedGradients(forward_func, model._modules["model"].embeddings)
        attr, delta = lig.attribute(inputs=b_input_ids,
                                    baselines=ref,
                                    additional_forward_args=(b_attn_mask, global_attention_mask, b_claims, article_id),
                                    return_convergence_delta=True,
                                    )
        #print("article id", article_id)
        #print(attr)
        #print(delta)
        attr_summary = summarize_attributions(attr)
        attr_per_sentence = []
        for i,sentence_length in enumerate(sentence_lengths):
            attr_per_sentence.append(attr_summary[:sentence_length].sum().item())
            attr_summary = attr_summary[sentence_length:]
        #print(attr_per_sentence)

        dev_attributions_per_sentence.append(attr_per_sentence)
        dev_rationales.append(rationale_binary)

import pdb; pdb.set_trace()
print("ALL:")
print(len(dev_attributions_per_sentence))
# find pearson and spearman correlation between item_distances and item_rationales
coef, p = spearmanr(flatten_list(dev_attributions_per_sentence), flatten_list(dev_rationales))
print("spearman")
print(coef, p)
# calculate kendall correlation between item_distances and item_rationales
coef, p = kendalltau(flatten_list(dev_attributions_per_sentence), flatten_list(dev_rationales))
print("kendall")
print(coef, p)
print("0:")



