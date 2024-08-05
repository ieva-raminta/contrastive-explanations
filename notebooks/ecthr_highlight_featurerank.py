import json
import random
from transformers import AutoTokenizer, AdamW, BertModel, AutoModel, LongformerModel
import numpy as np
from scipy.spatial import distance
from nltk.tokenize import sent_tokenize
import nltk
import torch
import tqdm
import re
nltk.download('punkt')
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import sys
import os
sys.path.append(os.path.abspath('..'))
from scipy.special import softmax
from transformers import BertConfig

DATASET="ecthr"
MODEL_NAME="allenai/longformer-base-4096"
model_directory_path = "/home/irs38/Negative-Precedent-in-Legal-Outcome-Prediction/results/Outcome/joint_model/longformer/facts/be7001606980465a883b6119ced5fb0d/"
model_path = model_directory_path+"model.pt"
model = torch.load(model_path)
model_state_dict = model.state_dict()

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

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

label2index = {"not_claimed":0, "claimed_and_violated":1, "claimed_not_violated":2}

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

def all_masks(tokenized_text):
    # https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
    # WITHOUT empty and full sets!
    s = list(range(len(tokenized_text)))
    x = len(s)
    masks = [1 << i for i in range(x)]
    #     for i in range(1 << x):  # empty and full sets included here
    for i in range(1, 1 << x - 1):
        yield [ss for mask, ss in zip(masks, s) if i & mask]
        
def all_consecutive_masks(tokenized_text, max_length = -1):
    # WITHOUT empty and full sets!
    s = list(range(len(tokenized_text)))
    x = len(s)
    for i in range(x):
        for j in range(i+1, x):
            mask = s[:i] + s[j:]
            if max_length > 0:
                if j - i >= max_length:
                    yield mask
            else:
                yield mask
                
def all_consecutive_masks2(tokenized_text, max_length = -1):
    # WITHOUT empty and full sets!
    s = list(range(len(tokenized_text)))
    x = len(s)
    for i in range(x+1):
        for j in range(i+1, x+1):
            mask = s[i:j]
            if max_length > 0:
                if j - i <= max_length:
                    yield mask
            else:
                yield mask

def precisionAtK(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(k)
    return result

def recallAtK(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(len(act_set))
    return result

def meanPrecisionAtK(actual, predicted, k):
    return np.mean([precisionAtK(a, p, k) for a, p in zip(actual, predicted)])

def meanRecallAtK(actual, predicted, k):
    return np.mean([recallAtK(a, p, k) for a, p in zip(actual, predicted)])

def flatten_list(l):
    return [item for sublist in l for item in sublist]

import pickle

tokenized_dir = "/home/irs38/Negative-Precedent-in-Legal-Outcome-Prediction/ECHR/Outcome/longformer"

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

articles = ['10', '11', '13', '14', '18', '2', '3', '4', '5', '6', '7', '8', '9', 'P1-1', 'P4-2', 'P7-1', 'P7-4']

#ex = {"facts": "5.  The applicant was born in 1983 and is detained in Sztum. 6.  At the time of the events in question, the applicant was serving a prison sentence in the Barczewo prison. 7.  On 8 January 2011 the applicant\u2019s grandmother died. On 10 January 2011 the applicant lodged a request with the Director of Prison and the Penitentiary judge for leave to attend her funeral which was to take place on 12 January 2011. Together with his application he submitted a statement from his sister E.K. who confirmed that she would personally collect the applicant from prison and bring him back after the funeral. 8.  On 11 January 2011 the Penitentiary judge of the Olsztyn Regional Court (S\u0119dzia Penitencjarny S\u0105du Okr\u0119gowego w Olsztynie) allowed the applicant to attend the funeral under prison officers\u2019 escort. The reasoning of the decision read as follows:\n\u201cIn view of [the applicant\u2019s] multiple convictions and his long term of imprisonment there is no guarantee that he will return to prison\u201d 9.  The applicant refused to attend the funeral, since he believed his appearance under escort of uniformed officers would create a disturbance during the ceremony. 10.  On the same day the applicant lodged an appeal with the Olsztyn Regional Court (S\u0105d Okr\u0119gowy) complaining that the compassionate leave was granted under escort and also that he was only allowed to participate in the funeral (not the preceding church service). 11.  On 3 February 2011 the Olsztyn Regional Court upheld the Penitentiary judge\u2019s decision and dismissed the appeal. The court stressed that the applicant had been allowed to participate in the funeral under prison officers\u2019 escort. It further noted that the applicant was a habitual offender sentenced to a long term of imprisonment therefore there was no positive criminological prognosis and no guarantee that he would have returned to prison after the ceremony.", "claims": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], "outcomes": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "case_no": "20488/11"}
#ex = {"facts": "4.  The applicant was born in 1960 and lives in Oleksandrivka, Kirovograd Region. 5.  On 3 February 2007 the applicant was assaulted. According to the subsequent findings of medical experts, she sustained haematomas on her jaw, shoulder and hip, a bruise under her right eye, concussion, and a displaced rib fracture. The applicant alleges that her assailants were Mr and Mrs K., her daughter\u2019s former parents-in-law, whereas the domestic authorities found that it was only Mrs K. who had assaulted the applicant. The incident occurred in front of the applicant\u2019s two-year-old granddaughter. 6.  On 4 February 2007 the applicant lodged a complaint with the police. 7.  On 5 February 2007 a forensic medical expert examined the applicant. He found that she had haematomas which he classified as \u201cminor bodily injuries\u201d. 8.  On 14 February 2007 the Oleksandrivka District Police Department (\u201cthe Oleksandrivka police\u201d) refused to institute criminal proceedings in connection with the incident. 9.  On 22 February 2007 a forensic medical examination of the applicant was carried out. The expert found that in addition to the previously noted haematomas, the applicant had also suffered concussion and a displaced rib fracture. The expert classified the injuries as \u201cbodily harm of medium severity\u201d. 10.  On 20 March 2007 the Oleksandrivka prosecutor overruled the decision of 14 February 2007 as premature and on 21 March 2007 instituted criminal proceedings in connection with the infliction of bodily harm of medium severity on the applicant. 11.  On 20 May 2007 the investigator suspended the investigation for failure to identify the perpetrator. 12.  On 29 August and 3 October 2007 the Oleksandrivka prosecutor\u2019s office issued two decisions in which it overruled the investigator\u2019s decision of 20 May 2007 as premature. 13.  On 6 October 2007 the investigator questioned Mr and Mrs K. 14.  On 1 December 2007 the investigator again suspended the investigation for failure to identify the perpetrator. 15.  On 10 December 2007 the Oleksandrivka prosecutor\u2019s office, in response to the applicant\u2019s complaint about the progress of the investigation, asked the Kirovograd Regional Police Department to have the police officers in charge of the investigation disciplined. 16.  On 21 January 2008 the Kirovograd Regional Police Department instructed the Oleksandrivka police to immediately resume the investigation. 17.  On 7 April 2008 the investigator decided to ask a forensic medical expert to determine the degree of gravity of the applicant\u2019s injuries. On 22 September 2008 the expert drew up a report generally confirming the findings of 22 February 2007. 18.  On 15 May 2008 the Kirovograd Regional Police Department informed the applicant that the police officers in charge of the case had been disciplined for omissions in the investigation. 19.  On 23 October 2008 the Oleksandrivka Court absolved Mrs K. from criminal liability under an amnesty law, on the grounds that she had an elderly mother who was dependent on her. On 24 February 2009 the Kirovograd Regional Court of Appeal (\u201cthe Court of Appeal\u201d) quashed that judgment, finding no evidence that Mrs K.\u2019s mother was dependent on her. 20.  On 1 July 2009 the investigator refused to institute criminal proceedings against Mr K. 21.  On 7 July 2009 the Novomyrgorod prosecutor issued a bill of indictment against Mrs K. 22.  On 24 July 2009 the Oleksandrivka Court remitted the case against Mrs K. for further investigation, holding that the applicant had not been informed about the completion of the investigation until 3 July 2009 and had therefore not been given enough time to study the case file. It also held that the refusal to institute criminal proceedings against Mr K. had contravened the law. 23.  On 13 November 2009 the Novomyrgorod prosecutor quashed the decision of 1 July 2009 not to institute criminal proceedings against Mr K. Subsequently the investigator again refused to institute criminal proceedings against Mr K. 24.  On 21 December 2009 the new round of pre-trial investigation in the case against Mrs K. was completed and another bill of indictment was issued by the Novomyrgorod prosecutor. 25.  On 29 March 2010 the Oleksandrivka Court remitted the case against Mrs K. for further investigation, holding in particular that the decision not to institute criminal proceedings against Mr K. had been premature, since his role in the incident had not been sufficiently clarified. 26.  On 13 July 2010 the Novomyrgorod prosecutor quashed the decision not to institute criminal proceedings against Mr K. On 26 May 2011 the investigator again refused to institute criminal proceedings against Mr K. 27.  On 20 December 2011 the Znamyanka Court convicted Mrs K. of inflicting bodily harm of medium severity on the applicant, sentencing her to restriction of liberty for two years, suspended for a one-year probationary period. The court found that the decision not to institute criminal proceedings against Mr K. in connection with the same incident had been correct. Mrs K., the prosecutor and the applicant appealed. 28.  On 6 March 2012 the Court of Appeal quashed the judgment and discontinued the criminal proceedings against Mrs K. as time-barred.", "claims": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "outcomes": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "case_no": "27454/11"}

silver_interesting_items = []
gold_interesting_items = []

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

model.eval()

non_zero = 0
for i,item in enumerate(dev_data): 
    b_input_ids, b_attn_mask, b_labels, b_claims, global_attention_mask = item
    logits, last_hidden_state_cls = model(b_input_ids.cuda(), b_attn_mask.cuda(), global_attention_mask, b_claims) #predictor.predict_json(e)
    logits = logits.reshape(b_input_ids.shape[0], -1, 3)
    claims = b_claims
    D_out = int(b_labels.shape[1] / 2)
    y = torch.zeros(b_labels.shape[0], D_out).long().to("cuda")
    y[b_labels[:, :D_out].bool()] = 1
    y[b_labels[:, D_out:].bool()] = 2
    y = y.squeeze(1)
    out = torch.argmax(logits, dim=2).squeeze(1)
    if ";" not in val_ids[i]:
        gold_id = val_ids[i]
    else:
         for id in val_ids[i].split(";"):
            if id in ids:
                gold_id = id
                break
    silver_rat = ids_to_rationales[gold_id]
    gold_rat = ids_to_gold_rationales[gold_id]
    ex = ids_to_ex[gold_id]

    if silver_rat != [] and silver_rat != "[]": # out.sum != 0 and torch.equal(out, y)
        rats = [int(num) for num in silver_rat.lstrip("[").rstrip("]").split(",")]
        silver_interesting_items.append({"out":out, "ex":ex, "claims":claims, "y":y, "rationales":rats})
        #break
    if gold_rat != [] and gold_rat != "[]":
        gold_rats = [int(num) for num in gold_rat.lstrip("[").rstrip("]").split(",")]
        gold_interesting_items.append({"out":out, "ex":ex, "claims":claims, "y":y, "rationales":gold_rats})
        #break

masking_sentences = True # otherwise masking tokens
using_gold = False # otherwise using silver rationales
using_all = True # otherwise using only the postiive subset

results = []

index2label = {0: "not_claimed", 1: "claimed_and_violated", 2: "claimed_not_violated"}

positive = []
negative = []
neutral = []

if using_gold:
    print("using gold rationales")
    interesting_items = gold_interesting_items.copy()
else:
    print("using silver rationales")
    interesting_items = silver_interesting_items.copy()

for interesting_item in interesting_items: 
    out = interesting_item["out"]
    claims = interesting_item["claims"]
    ex = interesting_item["ex"]
    y = interesting_item["y"]
    facts = ex["facts"]
    rationales = interesting_item["rationales"]

    preprocessed_ex, preprocessed_ex_masks = preprocessing_for_bert([facts], tokenizer, max=512)
    global_attention_mask = torch.zeros(preprocessed_ex.shape, dtype=torch.long, device="cuda")
    global_attention_mask[:, [0]] = 1

    encoded_orig = model(preprocessed_ex.squeeze(1).cuda(), preprocessed_ex_masks.squeeze(1).cuda(), global_attention_mask.squeeze(1), claims)[1]

    facts_sentences = facts
    sentence_lengths = [len(tokenizer.tokenize(sentence)) for sentence in facts_sentences]
    tokenized_facts = tokenizer.tokenize(" ".join(facts_sentences))
    number_of_sentences = len(facts_sentences)

    masks1 = [[]]  # change this if you also want to mask out parts of the premise.
    if masking_sentences: 
        masks2 = list(all_consecutive_masks2(facts_sentences, max_length=1))
    else: 
        masks2 = list(all_consecutive_masks2(tokenizer.tokenize(" ".join(facts_sentences))))
    
    encoded = []
    mask_mapping = []

    for m1_i, m1 in enumerate(masks1):
        masked1 = []
        for i in m1:
            masked1[i] = '<mask>'
        masked1 = ' '.join(masked1)
        masked_sentence = []
        for m2_i, m2 in enumerate(masks2):
            if masking_sentences: 
                masked2 = facts_sentences.copy()
                for i in m2:
                    masked_sentence.append(masked2[i])
                    sentence_length = len(tokenizer.tokenize(masked2[i]))
                    masked2[i] = '<mask> '*sentence_length
                masked2 = tokenizer.tokenize(' '.join(masked2))
            else: 
                masked2 = tokenized_facts.copy()
                for i in m2: 
                    masked2[i] = '<mask>'
                masked2 = masked2

            masked_ex = {
                "facts": masked2,
                "claims": claims,
                "case_no": ex['case_no']
            }
            
            preprocessed_masked_ex, preprocessed_masked_ex_masks = preprocessing_for_bert([masked_ex["facts"]], tokenizer, max=512)
            global_attention_mask = torch.zeros(preprocessed_masked_ex.shape, dtype=torch.long, device="cuda")
            global_attention_mask[:, [0]] = 1
            last_hidden_state_cls = model(preprocessed_masked_ex.squeeze(1).cuda(), preprocessed_masked_ex_masks.squeeze(1).cuda(), global_attention_mask.squeeze(1), claims)[1]
            #masked_out, last_hidden_state_cls = model(preprocessed_masked_ex.squeeze(1).cuda(), preprocessed_masked_ex_masks.squeeze(1).cuda(), global_attention_mask.squeeze(1), None)[0] #predictor.predict_json(masked_ex)

            #print("indices", m1_i, m2_i)
            #print("case facts with masks in them", f"{masked1}\n{masked2}")
            #print("gold labels", masked_out['labels'])
            #print("masked out sentence", masked_sentence)
            encoded.append(last_hidden_state_cls.cpu().detach())
            mask_mapping.append((m1_i, m2_i))
            
            #print("====")
            
    # make a tensor out of a list of tensors
    encoded = torch.cat(encoded, dim=0)
    encoded = np.array(encoded)

    encoded_orig = np.array(encoded_orig.cpu().detach())

    # replace some random f in the following list with another option from
    # ["not_claimed", "claimed_and_violated", "claimed_not_violated"] at random
    label_options = ["not_claimed", "claimed_and_violated", "claimed_not_violated"]
    interesting_label_options = ["claimed_and_violated", "claimed_not_violated"]
    # choosing an article for which either the predicted or the gold label is one of the interesting ones (not 0)
    article_id = y[0].tolist().index(2) if 2 in y[0] else random.choice([i for i in range(len(out[0])) if index2label[out[0][i].item()] in interesting_label_options or index2label[y[0][i].item()] in interesting_label_options])
    # choosing a foil randomly that is not the fact
    for label in [0,1,2]:
        if label in out[0]:
            article_id = out[0].tolist().index(label)

            fact_id = out[0][article_id].item()
            foil_id = random.choice([label2index[i] for i in interesting_label_options if label2index[i] != fact_id])

            fact_idx = article_id * len(label_options) + fact_id
            foil_idx = article_id * len(label_options) + foil_id

            classifier_w = model_state_dict["classifier_positive.0.weight"].cpu().numpy()
            classifier_b = model_state_dict["classifier_positive.0.bias"].cpu().numpy()

            u = classifier_w[fact_idx] - classifier_w[foil_idx]
            contrastive_projection = np.outer(u, u) / np.dot(u, u)

            z_all = encoded_orig 
            z_h = encoded 
            z_all_row = encoded_orig @ contrastive_projection
            z_h_row = encoded @ contrastive_projection

            prediction_probabilities = softmax(z_all_row @ classifier_w.T + classifier_b)
            prediction_probabilities = np.tile(prediction_probabilities, (z_h_row.shape[0], 1))

            prediction_probabilities_del = softmax(z_h_row @ classifier_w.T + classifier_b, axis=1)

            p = prediction_probabilities[:, [fact_idx, foil_idx]]
            q = prediction_probabilities_del[:, [fact_idx, foil_idx]]

            p = p / p.sum(axis=1).reshape(-1, 1)
            q = q / q.sum(axis=1).reshape(-1, 1)
            distances = (p[:, 0] - q[:, 0])

            highlight_rankings = np.argsort(-distances)
            explained_indices = []
            explained_distances = []

            if masking_sentences: 
                for i in range(len(facts_sentences)):
                    rank = highlight_rankings[i]
                    m1_i, m2_i = mask_mapping[rank]
                    
                    masked_sentence = []
                    masked2 = facts_sentences.copy()
                    for k in masks2[m2_i]:
                        masked_sentence.append(masked2[k])
                        masked2[k] = '<mask>'
                    explained_indices.append(k)
                    explained_distances.append(distances[rank])
                    masked2 = ' '.join(masked2)
            else: 
                for i in range(len(tokenized_facts)):
                    rank = highlight_rankings[i]
                    m1_i, m2_i = mask_mapping[rank]
                    masked_sentence = []
                    masked2 = tokenized_facts.copy()
                    for k in masks2[m2_i]:
                        masked_sentence.append(masked2[k])
                        masked2[k] = '<mask>'
                    explained_indices.append(k)
                    explained_distances.append(distances[rank])
                    masked2 = ' '.join(masked2)

            ex_dict = {"ex":ex, "rationales":rationales, "explained_indices":explained_indices, "explained_distances":explained_distances, "number_of_sentences":number_of_sentences, "article_id": article_id, "fact": index2label[fact_id], "foil": index2label[foil_id], "y": y, "predicted": out}
            
            if fact_id == 1:
                positive.append(ex_dict)
            elif fact_id == 2:
                negative.append(ex_dict)
            else: 
                neutral.append(ex_dict)

            results.append(ex_dict)
            print(ex_dict)

from scipy.stats import spearmanr
from scipy.stats import kendalltau

for label in [0,1,2]:
    results = [neutral, positive, negative][label]
    print("LABEL:", label)

    indices = [a["explained_indices"] for a in results]
    distances = [a["explained_distances"] for a in results]
    rationales = [p["rationales"] for p in results]
    number_of_sentences = [n["number_of_sentences"] for n in results]
    ys = [g["y"] for g in results]
    article_ids = [a["article_id"] for a in results]
    predicteds = [a["predicted"] for a in results]
    foil_ids = [label2index[a["foil"]] for a in results]

    all_distances = []
    all_rationales = []
    distances_correct = []
    rationales_correct = []
    ids_correct = []

    for rationale, distance, number, index, y, article_id, predicted, foil_id in zip(rationales, distances, number_of_sentences, indices, ys, article_ids, predicteds, foil_ids):
        gold_label = y[0][article_id]
        item_distances = []
        item_rationales = []
        for i in range(number):
            item_distances.append(distance[index.index(i)])
            if i in rationale: 
                item_rationales.append(1)
            else:
                item_rationales.append(0)
        if torch.equal(torch.tensor(predicted[0][article_id]), torch.tensor(y[0][article_id])):
            ids_correct.append(index)
            distances_correct.append(item_distances)
            rationales_correct.append(item_rationales)
        all_distances.append(item_distances)
        all_rationales.append(item_rationales)

    print(len(all_distances))
    # find pearson and spearman correlation between item_distances and item_rationales
    coef, p = spearmanr(flatten_list(all_distances), flatten_list(all_rationales))
    print("spearman")
    print(coef, p)
    # calculate kendall correlation between item_distances and item_rationales
    coef, p = kendalltau(flatten_list(all_distances), flatten_list(all_rationales))
    print("kendall")
    print(coef, p)

    print("CORRECT:")
    print(len(distances_correct))
    coef, p = spearmanr(flatten_list(distances_correct), flatten_list(rationales_correct))
    print("spearman")
    print(coef, p)
    coef, p = kendalltau(flatten_list(distances_correct), flatten_list(rationales_correct))
    print("kendall")
    print(coef, p)

