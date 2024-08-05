import json
import os

file_number_to_case_number = {}

# read in every filename from Negative-Precedent-in-Legal-Outcome-Prediction/ECHR/Outcome/dev/
for root, dirs, files in os.walk("/home/irs38/Negative-Precedent-in-Legal-Outcome-Prediction/ECHR/Outcome/dev/"):
    for file in files:
        if file.endswith(".json"):
            with open(os.path.join(root, file)) as f:
                data = json.load(f)
                case_number = data['case_no']
                file_number = file.split(".")[0]
                case_numbers = case_number.split(";")
                file_number_to_case_number[file_number] = case_numbers
for root, dirs, files in os.walk("/home/irs38/Negative-Precedent-in-Legal-Outcome-Prediction/ECHR/Outcome/train/"):
    for file in files:
        if file.endswith(".json"):
            with open(os.path.join(root, file)) as f:
                data = json.load(f)
                case_number = data['case_no']
                case_numbers = case_number.split(";")
                file_number = file.split(".")[0]
                file_number_to_case_number[file_number] = case_numbers
for root, dirs, files in os.walk("/home/irs38/Negative-Precedent-in-Legal-Outcome-Prediction/ECHR/Outcome/test/"):
    for file in files:
        if file.endswith(".json"):
            with open(os.path.join(root, file)) as f:
                data = json.load(f)
                case_number = data['case_no']
                case_numbers = case_number.split(";")
                file_number = file.split(".")[0]
                file_number_to_case_number[file_number] = case_numbers

# save the dictionary to a file
with open("/home/irs38/Negative-Precedent-in-Legal-Outcome-Prediction/ECHR/Outcome/numbers.json", "w") as f:
    json.dump(file_number_to_case_number, f)