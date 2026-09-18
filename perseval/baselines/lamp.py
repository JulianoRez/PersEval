import sys
import os

import pandas as pd
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, \
    Trainer, set_seed, TrainingArguments, pipeline, AutoModel
from datasets import Dataset
import torch
from sklearn.utils import compute_class_weight
import numpy as np
from tqdm import tqdm
from .personalized_llms import PrepareData
import json
import datasets

from .. import config

class PerspectivistLaMP():
    def __init__(self, model_identifier, persp_dataset, label, context,dataset_name):
        self.model_id = model_identifier
        self.label = label
        self.named = persp_dataset.named
        self.all_dataset = persp_dataset
        self.dataset = persp_dataset.name 
        self.context = context
        self.num_profiles = 5
        self.max_length = 1024
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.dataset_name = dataset_name
        if self.model_id == "mistralai/Mixtral-8x7B-Instruct-v0.1":
            self.output_path = config.prediction_dir_mixtral
        elif self.model_id == "meta-llama/Meta-Llama-3.1-8B-Instruct":
            self.output_path = config.prediction_dir_llama
        else:
            print("LaMP requires mistralai/Mixtral-8x7B-Instruct-v0.1 or meta-llama/Meta-Llama-3.1-8B-Instruct")
            exit()
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.float16, device_map="auto")
    
    def create_preprocessor(self):
        def preprocess(data):
            inputs = []
            targets = []
            for d in data:
                inputs.append(d["source"])
                targets.append(str(d["target"]))
            model_inputs = self.tokenizer(inputs, text_target=targets,return_tensors="pt", max_length=self.max_length, padding=True)
            return model_inputs
        return preprocess

    def to_hugging_dataset(dataset):
        def generator():
            for element in dataset:
                yield element
        return datasets.Dataset.from_generator(generator)
        
    def classification_query_corpus_maker(self, inp, profile):
        corpus = [f'{x["comment"]}' for x in profile]
        idx = inp.find('Input:')
        if idx == -1:
            return corpus, None
        query = inp[idx+len('Input:'):].strip()
        return corpus, query

    def create_prompt(self,input,profile,max_length,tokenizer,label):    
        inputs=input.split("Input:")
        if len(profile) == 0:
            print("No profiles found")
            return f'{inputs[0]}\nExample to label:\n {inputs[1]}\nYour output:'
        per_p_max_length = (max_length - 1 - 2 * (len(profile) - 1)) // len(profile)
        saved_tokens = 0
        prompts = []
        i = 1
        for p in profile:
            value_label = label if p[label] == 1 else "not " + label
            needed_part_len = len(tokenizer(f'Output: {value_label}')['input_ids'])
            tokens = tokenizer(p["comment"], max_length=per_p_max_length + saved_tokens - needed_part_len, truncation=True)
            saved_tokens += per_p_max_length - len(tokens['input_ids']) - needed_part_len
            new_text = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
            new_text = new_text.replace('post :', "- Post:").replace(" reply :","\n- Reply:")
            prompt = f'Example {i}:\nInput:\n{new_text} \nOutput: {value_label}\n'
            i=i+1
            prompts.append(prompt)
        return f'{inputs[0]}\n{"".join(prompts)}\nExample to label:\n {inputs[1]}\nYour output:'

    def create_prompt_generator(self,num,label):
        tokenizer = AutoTokenizer.from_pretrained('facebook/contriever', padding=True)
        tokenizer.pad_token = tokenizer.eos_token
        def prompt(input,profile):
            selected_profs_cont = profile[:num]
            selected_profs = selected_profs_cont
            max_len_prompt = self.max_length - min(len(tokenizer(input)['input_ids']), int(0.6 * self.max_length))
            out = self.create_prompt(input,selected_profs,max_len_prompt,tokenizer,label)
            return out
        return prompt

    def evaluate_dataset(self):
        # Load the tokenizer and model
        self.tokenizer.pad_token = self.tokenizer.eos_token
        prompt_generator = self.create_prompt_generator(self.num_profiles, self.label)
        files = [file for file in os.listdir(config.data_lamp_dir) if file.lower().startswith(self.dataset.lower()) and "merged" in file and str(self.named) in file]
        if len(files) == 0:
            print("No merged files found. Please generate them using PrepareData.")
            exit()
        
        preprocessor = self.create_preprocessor()
        for file in files:
            with open(f'{config.data_lamp_dir}/{file}') as f:
                data = json.load(f)
                print(f'Processing {file}')
            if not os.path.exists(f'{config.data_lamp_dir}/output/'):
                os.makedirs(f'{config.data_lamp_dir}/output/')
            with open(f'{config.data_lamp_dir}/output/{file.replace("json","csv").replace("_merged","")}', 'w', newline='') as out_csv:
                writer = csv.writer(out_csv)
                field=["user_id","id","predictions"]
                writer.writerow(field)
                outputs = []
                with torch.no_grad():
                    for d in data:
                        print("Processing user: ", d)
                        for element in tqdm(data[d]):
                            profiles=[{
                                "id": element['id'],
                                "source": prompt_generator(element['input'], element['profile']),
                                "target": element[self.label]
                            }]
                            preprocessed_data = preprocessor(profiles)
                            inputs = {key: value.to("cuda") for key, value in preprocessed_data.items()}
                            for i in range(len(inputs["input_ids"])):
                                input_ids = inputs["input_ids"][i].unsqueeze(0)
                                attention_mask = inputs["attention_mask"][i].unsqueeze(0)
                                output = self.model.generate(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    max_new_tokens=100,
                                    do_sample=False
                                )
                                outputs.append(output)
                            generated_ids = output[:, inputs["input_ids"].shape[-1]:]
                            writer.writerow([d, element['id'],self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)])


    def merge_data(self, input, output, ranks,label):
        for data in input:
            for inp in input[data]:
                for id in output:
                    for o in output[id]:
                        if o['id'] == inp['id']:
                            outs = o[label]
                            break
                new_profile = []
                for x in ranks[data][inp['id']]:
                    for y in inp['profile']:
                        if y['id'] == x:
                            new_profile.append(y)
                            break
                inp['profile'] = new_profile
                inp[label] = outs
        return input

    def rank_profile(self,prompt):
        PrepareData(persp_dataset=self.all_dataset, dataset_config=prompt, named=self.named, context=self.context)
        contriver = AutoModel.from_pretrained("facebook/contriever").to("cuda:0")
        tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")

        contriver.eval()
        
        files = [file for file in os.listdir(config.data_lamp_dir) if file.lower().startswith(self.dataset.lower()) and "input" in file and str(self.named) in file]
        
        if len(files)==0:
            print("No input files found, you need to generate them with PrepareData")
            exit()
        
        for file in files:
            rank_dict={}
            with open(f'{config.data_lamp_dir}/{file}') as f:
                data = json.load(f)
                for user in tqdm(data):
                    rank_dict[user]={}
                    for comment in data[user]:
                        corpus,query = self.classification_query_corpus_maker(comment['input'],comment['profile'])
                        ranked_profile = self.retrieve_top_k_with_contriver(contriver, tokenizer, corpus, comment['profile'], query, len(comment['profile']), 16)
                        comment['profile'] = ranked_profile
                        rank_dict[user][comment['id']] = [x['id'] for x in ranked_profile]
            with open(f'{config.data_lamp_dir}/{file.replace("input","ranked")}', "w") as out:
                json.dump(rank_dict, out)
                
    def merge_profile(self):
        files = [file for file in os.listdir(config.data_lamp_dir) if file.lower().startswith(self.dataset.lower()) and "input" in file and str(self.named) in file]
        with open(f'{config.data_lamp_dir}/{self.dataset_name}_{self.named}_output.json') as output_file:
            out =json.load(output_file)
        if len(files)==0:
            print("No input files found, you need to generate them with Rank Profile")
            exit()
        for file in files:
            with open(f'{config.data_lamp_dir}/{file}') as f:
                data = json.load(f)
            if not os.path.isfile(f'{config.data_lamp_dir}/{file.replace("input","ranked")}'):
                print(f'File not found: {config.data_lamp_dir}/{file.replace("input","ranked")}')
                exit()
            with open(f'{config.data_lamp_dir}/{file.replace("input","ranked")}') as f:
                ranked_data = json.load(f)
            with open(f'{config.data_lamp_dir}/{file.replace("input","merged")}',"w") as merge_file:
                merged = self.merge_data(data,out, ranked_data,self.label)
                json.dump(merged,merge_file,indent=4)
                
    # LaMP rank_profiles methods
    def mean_pooling(self,token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings
    
    def batchify(self,lst, batch_size):
        return [lst[i:i+batch_size] for i in range(0, len(lst), batch_size)]

    def retrieve_top_k_with_contriver(self,contriver, tokenizer, corpus, profile, query, k, batch_size = 16):
        query_tokens = tokenizer([query], padding=True, truncation=True, return_tensors='pt').to("cuda:0")
        output_query = contriver(**query_tokens)
        output_query = self.mean_pooling(output_query.last_hidden_state, query_tokens['attention_mask'])
        scores = []
        batched_corpus = self.batchify(corpus, batch_size)
        for batch in batched_corpus:
            tokens_batch = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to("cuda:0")
            outputs_batch = contriver(**tokens_batch)
            outputs_batch = self.mean_pooling(outputs_batch.last_hidden_state, tokens_batch['attention_mask'])
            temp_scores = output_query.squeeze() @ outputs_batch.T
            scores.extend(temp_scores.tolist())
        _, topk_indices = torch.topk(torch.tensor(scores), k)
        return [profile[m] for m in topk_indices.tolist()]