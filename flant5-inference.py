import json
import pandas as pd
import torch
import tqdm

from collections import defaultdict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


option2label = {
    '(A)': 0,
    'A': 0,
    '(B)': 1,
    'B': 1,
    '(C)': 2,
    'C': 2,
    '(D)': 3,
    'D': 3
}

file_path = "data/SP-train.json"
with open(file_path) as f:
    data = json.load(f)
print("number of puzzles: ", len(data))

# file_path = "SP-eval.json"
# with open(file_path) as f:
#     data = json.load(f)
# print(len(data))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

model_paths = {
    'flan-t5-large-no-finetune': 'google/flan-t5-large',
    'flan-t5-large-rs-finetune': '/usr1/data/devanshj/brainteaser/checkpoints/flan-t5-large_full_finetune',
    'flan-t5-large-adv-finetune': '/usr1/data/devanshj/brainteaser/checkpoints/flan-t5-large_adversarial_finetune',
}

for model_name, model_path in model_paths.items():
    print("-" * 50)
    print("model name:", model_name)
    MODEL_SIZE = model_name.split('-')[2]
    BASE_MODEL = f"google/flan-t5-{MODEL_SIZE}"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    prediction_data = defaultdict(list)

    for mcq in tqdm.tqdm(data):
        prediction_data['id'].append(mcq['id'])
        prediction_data['question'].append(mcq['question'])
        prediction_data['answer'].append(mcq['answer'])
        prediction_data['label'].append(mcq['label'])
        prediction_data['choice_0'].append(mcq['choice_list'][0])
        prediction_data['choice_1'].append(mcq['choice_list'][1])
        prediction_data['choice_2'].append(mcq['choice_list'][2])
        prediction_data['choice_3'].append(mcq['choice_list'][3])
        if 'SR' in mcq['id']:
            prediction_data['group'].append('SR')
        elif 'CR' in mcq['id']:
            prediction_data['group'].append('CR')
        else:
            prediction_data['group'].append('OG')

        prompt = f"""Question: {mcq['question']}

        What is the correct answer to the question from the following choices?
        (A) {mcq['choice_list'][0]}
        (B) {mcq['choice_list'][1]}
        (C) {mcq['choice_list'][2]}
        (D) {mcq['choice_list'][3]}"""

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs)
        predicted_option = tokenizer.batch_decode(
            outputs, skip_special_tokens=True)[0]
        for k, v in option2label.items():
            if k in predicted_option:
                predicted_label = v

        prediction_data['prediction'].append(predicted_label)
        if int(mcq['label']) == int(predicted_label):
            prediction_data['correct'].append(1)
        else:
            prediction_data['correct'].append(0)

    prediction_df = pd.DataFrame(prediction_data)
    accuracy = prediction_df['correct'].sum() / prediction_df.shape[0]
    print("overall accuracy: ", accuracy * 100)

    print("storing results...")
    prediction_df.to_csv(f"results/a4/{model_name}.csv")
