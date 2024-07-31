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

torch.manual_seed(123)
np.random.seed(123)

model_path = "/home/irs38/Negative-Precedent-in-Legal-Outcome-Prediction/results/Outcome/joint_model/legal_bert/facts/ccc660d6049c4d1782bc6c81f2f30b12/model.pt"
model = torch.load(model_path)
MODEL_NAME="nlpaueb/legal-bert-base-uncased"
model.eval()
model.zero_grad()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenized_dir = "/home/irs38/Negative-Precedent-in-Legal-Outcome-Prediction/ECHR/Outcome/legal_bert"

def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def preprocessing_for_bert(data, tokenizer, max=512):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """

    # For every sentence...
    input_ids = []
    attention_masks = []

    for sent in data:
        sent = " ".join(sent)
        sent = sent[:500000] # Speeds the process up for documents with a lot of precedent we would truncate anyway.
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=max,  # Max length to truncate/pad
            pad_to_max_length=True,  # Pad sentence to max length
            # return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True,  # Return attention mask
            truncation=True,
        )

        # Add the outputs to the lists
        input_ids.append([encoded_sent.get('input_ids')])
        attention_masks.append([encoded_sent.get('attention_mask')])

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

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

def forward_func(b_input_ids, b_attn_mask=None, global_attention_mask=None, b_claims=None):
    logits = predict(b_input_ids, b_attn_mask, global_attention_mask, b_claims)
    logits = logits.reshape(b_input_ids.shape[0], -1, 3)
    out = torch.argmax(logits, dim=2).squeeze(1)
    return out

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


for i,item in enumerate(dev_data): 
    b_input_ids, b_attn_mask, b_labels, b_claims, global_attention_mask = item
    claims = b_claims

    """
    logits, last_hidden_state_cls = model(b_input_ids.cuda(), b_attn_mask.cuda(), global_attention_mask, b_claims) #predictor.predict_json(e)
    logits = logits.reshape(b_input_ids.shape[0], -1, 3)
    gold = b_labels
    out = torch.argmax(logits, dim=2).squeeze(1)
    """
    D_out = int(b_labels.shape[1] / 2)
    y = torch.zeros(b_labels.shape[0], D_out).long().to("cuda")
    y[b_labels[:, :D_out].bool()] = 1
    y[b_labels[:, D_out:].bool()] = 2
    y = y.squeeze(1)

    if ";" not in val_ids[i]:
        gold_id = val_ids[i]
    else:
         for id in val_ids[i].split(";"):
            if id in ids:
                gold_id = id
                break

    #silver_rat = ids_to_rationales[gold_id]
    #ex = ids_to_ex[gold_id]
    #facts = ex["facts"]
    #preprocessed_ex, preprocessed_ex_masks = preprocessing_for_bert([facts], tokenizer, max=512)
    #encoded = model(preprocessed_ex.squeeze(1).cuda(), preprocessed_ex_masks.squeeze(1).cuda(), global_attention_mask.squeeze(1), claims)[1]
    #baseline = torch.zeros_like(encoded) 

    ref_token_id = tokenizer.pad_token_id
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    ref_input_ids = [cls_token_id] + [ref_token_id] * (len(b_input_ids)-2) + [sep_token_id]
    ref = torch.tensor([ref_input_ids], device="cuda")


    print("EXAMPLE", i)
    lig = LayerIntegratedGradients(forward_func, model._modules["model"].embeddings)
    attr, delta = lig.attribute(inputs=b_input_ids,
                                  baselines=ref,
                                  additional_forward_args=(b_attn_mask, global_attention_mask, b_claims),
                                  return_convergence_delta=True)
    #attr, delta = lig.attribute(inputs=encoded, baselines=baseline, return_convergence_delta=True)
    print(attr)
    print(delta)
