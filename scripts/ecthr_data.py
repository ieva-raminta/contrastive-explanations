from datasets import load_dataset
import json

ecthr_alleged = load_dataset("ecthr_cases", "alleged-violation-prediction")
ecthr = load_dataset("ecthr_cases", "violation-prediction")

ecthr_joined = {}
for split in ["train", "validation", "test"]:
    ecthr_joined[split] = []
    for i, item in enumerate(ecthr[split]):
        new_item = item
        new_item["alleged_labels"] = ecthr_alleged[split][i]["labels"]
        new_item["gold_rationales"] = ecthr_alleged[split][i]["gold_rationales"]
        ecthr_joined[split].append(new_item)

data_path = "~/contrastive-explanations/data/ecthr/"

for split in ["train", "validation", "test"]:
    ecthr_split = ecthr_joined[split]
    # save data to jsonl files
    with open(data_path + f"ecthr_{split}.jsonl", "w+") as f:
        for item in ecthr_split:
            f.write(json.dumps(item) + "\n")
