import torch

from poe_utils import *
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

batch_size = 4

print("loading model...")
model_path = 'google/flan-t5-small'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

print("assigning compute and preprocess functions...")
compute_func = compute_conditional_score_seq2seq
preprocess_func = preprocess_function_seq2seq
remove_columns = ['header_input_ids',
                  'header_attention_mask',
                  'ending_input_ids',
                  'ending_attention_mask', ]

print("-" * 50)
print("creating dataset...")
multiple_choice_prompt = ""
# create transformers dataset
ending_names, header_name, raw_dataset, n_shot_dataset = load_data(
    multiple_choice_prompt)
fn_kwargs = {"ending_names": ending_names,
             "header_name": header_name,
             "tokenizer": tokenizer, }
num_of_options = len(ending_names)
tokenized_dataset = raw_dataset.map(
    preprocess_func, fn_kwargs=fn_kwargs, batched=True, batch_size=batch_size)
eval_dataloader = DataLoader(
    tokenized_dataset, batch_size=batch_size, shuffle=False)

print("-" * 50)
print("scoring method...")
scoring_method = "multiple_choice_prompt"
# raw_mcp_dataset is same as raw_dataset since mcp = "", line 109
avg_log_probs, _, _ = inference_language_modeling(
    model, eval_dataloader, device, compute_func, tokenizer.pad_token_id)

print("-" * 50)
print("computing masks...")
mask_strategy = "below_average"
mask_kwargs = {}
masks = compute_mask_process_of_elimination(
    avg_log_probs, mask_strategy, **mask_kwargs)
masks = masks.to(torch.float32)
# compute mask accuracy, i.e., check whether mask that correspond to labels is 1
mask_result = masks[torch.arange(masks.size(0)), tokenized_dataset["label"]]
mask_accuracy = torch.sum(mask_result) / mask_result.size(0)
print("mask_accuracy:", mask_accuracy)

print("-" * 50)
print("creating masked dataset...")
masked_dataset = tokenized_dataset.map(lambda example, idx: {"mask": masks[idx]},
                                       with_indices=True,
                                       batched=True,
                                       remove_columns=remove_columns)

prompting_method = "multiple_choice_prompt"
process_of_elimination_prompt = "Select the most suitable option to answer the question. Ignore [MASK] options."
mask_token = None
mcp_kwargs = {"multiple_choice_prompt": process_of_elimination_prompt,
              "scoring_method": scoring_method,
              "num_of_options": num_of_options,
              "mask_token": mask_token, }

print("creating MCP prompt with masks...")
mcp_dataset = masked_dataset.map(
    create_multiple_choice_prompt, fn_kwargs=mcp_kwargs)

print("-" * 50)
print("creating tokenized dataset for second step inference...")
mcp_dataset = mcp_dataset.map(
    preprocess_func, fn_kwargs=fn_kwargs, batched=True, batch_size=batch_size)
eval_mcp_dataloader = DataLoader(
    mcp_dataset, batch_size=batch_size, shuffle=False)

print("PoE inference...")
lm_predictions,  lm_accuracy, _ = inference_process_of_elimination(
    model, eval_mcp_dataloader, device, compute_func, tokenizer.pad_token_id)

print("-" * 50)
print("storing results...")
print("lm accuracy:", lm_accuracy)
results_path = f"results/poe/{model_path[7:]}-no-finetune.csv"
write_to_csv(results_path, mcp_dataset, lm_predictions)
