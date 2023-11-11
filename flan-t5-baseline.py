import json
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

option2label = {
    '(A)': 0,
    '(B)': 1,
    '(C)': 2,
    '(D)': 3,
}

prediction_data = {
    'id': [],
    'question': [],
    'answer': [],
    'label': [],
    'prediction': [],
    'correct': [],
    'group': [],
    'choice_0': [],
    'choice_1': [],
    'choice_2': [],
    'choice_3': [],
}

file_path = "SP-train.json"
with open(file_path) as f:
    data = json.load(f)
print(len(data))

# file_path = "SP-eval.json"
# with open(file_path) as f:
#     data = json.load(f)
# print(len(data))

for mcq in data:
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

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs)
    predicted_option = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    # predicted_label = option2label[predicted_option]
    for k, v in option2label.items():
        if k in predicted_option:
            predicted_label = v
    print(predicted_label)

    prediction_data['prediction'].append(predicted_label)
    if int(mcq['label']) == int(predicted_label):
        prediction_data['correct'].append(1)
    else:
        prediction_data['correct'].append(0)

# print(prediction_data)
prediction_df = pd.DataFrame(prediction_data)
print(prediction_df)
print(prediction_df.shape)

prediction_df_og = prediction_df[prediction_df['group'] == 'OG']
accuracy = prediction_df_og['correct'].sum() / prediction_df_og.shape[0]
print(accuracy)
# prediction_df.to_csv("SP-train-flant5-780M.csv")