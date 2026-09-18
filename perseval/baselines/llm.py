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
import json
import datasets

from .. import config

class PerspectivistLLM():
    def __init__(self, model_identifier, persp_dataset, label):
        self.model_id = model_identifier
        self.training_split = persp_dataset.training_set
        self.adaptation_split = persp_dataset.adaptation_set        
        self.test_split = persp_dataset.test_set
        self.label = label
        self.traits = persp_dataset.traits
        self.named = persp_dataset.named
        self.user_adaptation = persp_dataset.user_adaptation
        self.extended = persp_dataset.extended
        self.dataset = persp_dataset.name 
    
        if self.model_id == "mistralai/Mixtral-8x7B-Instruct-v0.1":
            self.mixtral = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.float16, device_map="auto")
            self.mixtral_tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.output_path = config.prediction_dir_mixtral
        elif self.model_id == "meta-llama/Meta-Llama-3.1-8B-Instruct":
            self.output_path = config.prediction_dir_llama
            self.llama = pipeline(
                "text-generation",
                model=self.model_id,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device="cuda")


    def predict(self):
        #prep output path
        if not os.path.exists(self.output_path): 
            os.makedirs(self.output_path)
        
        #prep experiment
        all_user_traits = {}
        if self.named:
            settings = config.prompts[self.dataset]["traits"]
            all_user_traits = self.get_user_traits(settings)
            
        #inference for each perspective
        csv_files = {}
        try:
            for sample in tqdm(self.test_split, desc="Processing samples"):
                text_id = sample.instance_id
                user_id = sample.user.id
                text = sample.instance_text

                if self.named:
                    if user_id not in all_user_traits:
                        continue
                    # Get the traits for the current user
                    traits = all_user_traits[user_id]
                else:
                    traits = {"zero": "zero"}

                # Write data for each trait of the user
                for trait, profile in traits.items():
                    filename = "/predictions_%s_%s_%s_%s_%s.csv" % (self.dataset, self.named, self.user_adaptation, self.extended, trait)

                    # Open or retrieve the CSV file for this trait
                    if trait not in csv_files:
                        fo = open(self.output_path+filename, "a")
                        writer = csv.DictWriter(fo, fieldnames=["user_id", "text_id", "predictions", "thoughts", "prompt"])
                        writer.writeheader()
                        csv_files[trait] = (fo, writer)

                    fo, writer = csv_files[trait]

                    prompt = self.create_prompt(text, profile)
                    messages = [{"role": "user", "content": prompt}]


                    if self.model_id == "mistralai/Mixtral-8x7B-Instruct-v0.1":
                        #prepare
                        model_inputs = self.mixtral_tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
                        #infer
                        generated_ids = self.mixtral.generate(model_inputs, max_new_tokens=100, do_sample=False)
                        llm_response = self.mixtral_tokenizer.batch_decode(generated_ids)[0]
                    elif self.model_id == "meta-llama/Meta-Llama-3.1-8B-Instruct":
                        #infer
                        outputs = self.llama(messages, max_new_tokens=100, do_sample=False)
                        llm_response = outputs[0]["generated_text"][-1]["content"]

                    prediction = self.parse_output(llm_response)
                    
                    writer.writerow({
                        "user_id": user_id,
                        "text_id": text_id,
                        "predictions": prediction,
                        "thoughts": llm_response,
                        "prompt": prompt
                    })
        finally:
            for file, _ in csv_files.values():
                file.close()



    def create_prompt(self, text, trait):
        prompt_options = config.prompts[self.dataset]
      
        #add perspective if it is set
        if self.named:
            prompt = f"You are {trait}.\n"
        else:
            prompt = ""

        #add intructions and explanations
        prompt = prompt + f"{prompt_options['prelude']} {prompt_options['task']} {prompt_options['instr_pre']}" 
        
        #dynamically add the options
        pred_option_count = len(prompt_options["pred_opt"])
        for i in range(pred_option_count-1): 
            prompt = prompt + f" '{prompt_options['pred_opt'][i]}'"
        prompt = prompt + f" or '{prompt_options['pred_opt'][-1]}' {prompt_options['instr_post']}.\n"

        #add the context part
        prompt = prompt + f"{prompt_options['context_pre']}\n"
        for key, value in text.items():
            prompt = prompt + f"{key}:\n {value}\n"
        prompt = prompt + f"{prompt_options['context_post']}"

        return prompt



    def parse_output(self, llm_response):
        #prep label strings
        pred_labels = config.label_map[self.label+"_pred"]

        #parse llm predictions
        if self.model_id == "mistralai/Mixtral-8x7B-Instruct-v0.1":
            llm_response = llm_response.split("[/INST]")[-1]
        pred_str = llm_response.split("{")[-1].split("}")[0].strip().lower()

        #filtering
        pred_str = pred_str.strip("\'")

        #map the predictions
        if pred_str in pred_labels:
            pred_int = pred_labels[pred_str]
        else:
            pred_int = -1

        return pred_int



    def get_user_traits(self, settings):
        all_user_traits = {}
        for user, user_class in self.test_split.users.items():
            traits = user_class.traits 
            store_traits = {}

            for trait,value in traits.items():
                trait_value = None

                for k,v in settings.items():
                    if value[0] == k:
                        trait_value = v
                        break
                if trait_value is None: 
                    trait_value = f"from an unknown {trait}"
                
                store_traits[trait] = trait_value
            all_user_traits[user] = store_traits

        return all_user_traits