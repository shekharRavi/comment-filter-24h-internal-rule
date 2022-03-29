# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import re

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader,  SequentialSampler


from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments,DataCollatorWithPadding


import pandas as pd

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

from langdetect import detect
from lingua import Language, LanguageDetectorBuilder

from tqdm import tqdm

lang_detector = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
valid_langs= ['BOSNIAN', 'SERBIAN','CROATIAN','MONTENEGRIN', 'SLOVENE','SLOVAK', 'hr','sr','bs','cnr','sl','slv']


map_24h_internal_rule={"0":"0","1":"5","2":"1","3":"2","4":"3","5":"4","6":"6","7":"7","8":"8"}

class ModelLoad():

    def __init__(self):
        # configuration
        self.ROOT_FOLDER = os.path.dirname(__file__)
        # self.ROOT_FOLDER = '.'

        print('ROOT_FOLDER',self.ROOT_FOLDER)

        # Load a trained model that you have fine-tuned
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model = 'EMBEDDIA/crosloengual-bert' #EMBEDDIA Model

        DIRECTORIES = {
            'ml_hate_speech_path': os.path.join(self.ROOT_FOLDER, 'models/ml_hate_speech_classifier', self.bert_model)
        }
        self.model_dir = DIRECTORIES['ml_hate_speech_path']
        self.model_file = os.path.join(DIRECTORIES['ml_hate_speech_path'], 'pytorch_model.bin')
        self.model_file_nb = os.path.join(DIRECTORIES['ml_hate_speech_path'], 'nb_model_bigram.sav')

        print('model_file',self.model_file)
        print('model_dir',os.listdir(os.path.join(self.ROOT_FOLDER, 'models')))
        print('model_dir_s',os.listdir(DIRECTORIES['ml_hate_speech_path']))
        print(os.path.isfile(self.model_file))
        if not os.path.isfile(self.model_file):
            print('Please Download the model ...')
            exit(0)


        # config_file = DIRECTORIES['ml_hate_speech_path']+'/config.json'
        # token_file = DIRECTORIES['ml_hate_speech_path']+'/vocab.txt'

        # config = BertConfig.from_pretrained(config_file, num_labels=9)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        print('Loading model ...', self.model_file)

        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)

        self.model.eval()
        self.model.to(self.device)
        self.softmax = torch.nn.Softmax(dim=1)

        r2_file = '/2_possible_words.csv'
        r3_file = '/3_possible_words.csv'
        r4_file = '/4_possible_words.csv'
        r5_file = '/5_possible_words.csv'
        r6_file = '/6_possible_words.csv'
        all_file = '/possible_words.csv'

        self.R2 = pd.read_csv(self.model_dir + r2_file).rule_word.tolist()
        self.R3 = pd.read_csv(self.model_dir + r3_file).rule_word.tolist()
        self.R4 = pd.read_csv(self.model_dir + r4_file).rule_word.tolist()
        self.R5 = pd.read_csv(self.model_dir + r5_file).rule_word.tolist()
        self.R6 = pd.read_csv(self.model_dir + r6_file).rule_word.tolist()
        self.all_words = pd.read_csv(self.model_dir + all_file).rule_word.tolist()


        self.not_allowed_words = ['Adolf', 'Hitler', 'jebem', 'jebemti', 'pička', 'lezba', 'majmun',
                             'mater', 'majku', 'kurac', 'seri', 'seres', 'nabijem', 'glup', 'idijot', 'idiot', 'budala',
                             'kreten', 'pizda', 'klad', 'kladi', 'ZDS', 'klaunski', 'mater', 'smece', 'tenkre', 'bilde',
                             'debil', 'jebi', 'cigan', 'govn', 'seljac', 'drolj', 'kozojeb', 'musliman', 'klaunski',
                             'derpe', 'maloumni', 'hebe', 'racku', 'spodob', 'kita', 'stoka', 'crkn', 'debel', 'krmaca',
                             'dubre', 'djubre', 'retard', 'barbika', 'miss piggy', 'srb', 'kme', 'novina', 'pick',
                             'nepism', 'sipt', 'ptar', 'za dom', 'srps', 'tuka', 'jeb, peni', 'udba, čas', 'šiptar',
                             'šupak', 'ustaša', 'smrad', 'budal', 'kopile', 'imbecil', 'guba', 'stoko', 'icka', 'ubre',
                             'gnjida', 'ljubičice', 'štraca', 'šljam', 'pupavac', 'jado', 'bilde', 'straca', 'sereš',
                             'đubre', 'čedo', 'pička', 'jadnič', 'četn', 'kurc', 'krmača', 'lomač', 'metak', 'čelo',
                             'jebo', 'ubit', 'asenovac', 'cetni', 'gamad', 'kurva', 'peder', 'kurvetina', 'kurv',
                             'kobilo', 'degenerik', 'panglu', 'tenkre', 'smeće', 'sme', 'smece', 'sponzoruša',
                             'sponzorusa',
                             'sponz', 'konju', 'krivousti', 'krivou', 'hanzek', 'hanžek', 'lešinari', 'lesinari',
                             'ološ', 'olos', 'papcino', 'papak', 'papčino', 'bosanko', 'bosanđeros', 'bosanceros',
                             'bosancina', 'hercegovance', 'šupak', 'šupci', 'supak', 'supci', 'bosančeros', 'kuja',
                             'kujica', 'dementra', 'dementna', 'nakaza', 'katolibani', 'talibani', 'papčina', 'kuraba',
                             'ganci', 'ljadro', 'retard', 'paksu', 'droca', 'express', 'srba', 'srbi', 'expres',
                             'šuft', 'suft', 'ćifut', 'katoliban', 'kolje', 'klati', 'kolj', 'jambrusic', 'jambrušić',
                             'tolusic', 'tolušić', 'siptar', 'balija', 'droca', 'acab', 'a.c.a.b.', 'radman',
                             'selekcija', 'sjajna zvijezdo', 'sjajna zvjezdo', 'celofanka', 'kravo', 'kobila',
                             'samoprozvani', 'doktor za ljubav', 'drkolinda', 'poturica', 'poturico', 'isprdak']

    def load_models(self):
        #Return Model
        return self.model,self.tokenizer, self.data_collator

    def get_model(self):
        # Return Model
        return self.model

    def get_tokenizer(self):
        return self.tokenizer
    def get_data_collator(self):
        return self.data_collator

    def get_device(self):
        return self.device
    def get_softmax(self):
        return self.softmax
    

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


class HRDataset(torch.utils.data.Dataset):
    # 24Sata Dataset Processing
    def __init__(self, encodings,labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]) #TODO make a test Data Class
        # item['idx'] = torch.tensor(self.idxs[idx])
        return item

    def __len__(self):
        return len(self.labels)

def check_blocked_words(text,not_allowed_words=[]):
    # Simply based on the texonomy
    rule_flag = False
    words = text.split(' ')
    c=0
    for word in words:
        if word in not_allowed_words:
            rule_flag = True
            # found_word = word  # TODO: This misses if multiple words are present
            break
    return rule_flag


def upper_check(text):
    rule_flag = False

    # print(text)
    if text.isupper():
        rule_flag = True
        # print('All upper')
    else:
        words = text.split(' ')
        c_w = len(words)
        up_count = 0
        for word in words:
            if word.isupper():
                up_count += 1
                ratio = up_count / c_w
                diff = c_w - up_count
                if diff < 3:
                    rule_flag = True
                    break
                elif ratio > 0.9:
                    rule_flag = True
                    break
    return rule_flag

def check_rule_seven(text):
    # To check based on the language and upper case
    text = " ".join(re.findall("[a-zA-Z]+", text))
    rule_flag = upper_check(text)
    if not rule_flag:
        words = text.split(' ')
        c_w = len(words)
        if c_w > 0:
            rule_flag = True
            if len(words)>4:
                # Check in two instances
                val1 = " ".join(words[:int(c_w/2)])
                val2 = " ".join(words[int(c_w/2):])
                values = [val1,val2, text]
            else:
                values = [text]
            for val in values:
                try:
                    lang2 = detect(val) #Easy to run
                except:
                    lang2 = 'hr'
                # print(lang2)
                if lang2 not in valid_langs:
                    try:
                        lang1 = str(lang_detector.detect_language_of(val))[9:]
                    except:
                        lang1 = 'CROATIAN'
                    # print(lang1)
                    if lang1 in valid_langs: #Found valid language, so not breaking Rule 7
                        rule_flag = False  # Found valid language, so not breaking Rule 7
                        break
                else:
                    rule_flag = False #Found valid language, so not breaking Rule 7
                    break

    return rule_flag

def keyword_to_rule(text,rule_words,threshold=1):
    # print('keyword_to_rule', text)
    words = text.split(' ')
    c = 0
    rule_flag = False
    for word in words:
        if word in rule_words:
            c +=1
            if c > threshold:
                # print('Found',c, threshold)
                rule_flag = True #TODO: use overall ratio to determine instead of count only
                break
    # print('keyword_to_rule',c)
    return rule_flag


def keyword_based_classification(text, model_load):
    # print('keyword_based_classification', text)
    # TODO: we coud include keyword based for the Rule 1 and 8 also.
    rule = 0
    conf = 1
    # Check blacklisted words
    rule_flag = check_blocked_words(text, not_allowed_words=model_load.not_allowed_words)
    if rule_flag:
        rule = 3
        conf = 0.7
    else:
        # Check rule 7 based on language or all Caps
        rule_flag = check_rule_seven(text)
        if rule_flag:
            rule = 7
            conf = 0.9
        else:
            # Check based on keywords for Rule 2,3,4,5,6,8
            rule_words = [model_load.R3, model_load.R4, model_load.R5, model_load.R2,model_load.R6, model_load.all_words]
            thresholds = [3, 3, 4, 2, 2, 2]  # For Major rule higher threshold
            rules = [3, 4, 5, 2, 6, 8]
            for rule_word, threshold, rule_key in zip(rule_words, thresholds, rules):
                rule_flag = keyword_to_rule(text, rule_word, threshold=threshold)
                # print('Check R' + str(rule_key), rule_flag)
                if rule_flag:
                    rule = rule_key
                    conf = 0.7
                    break
    return rule, conf

def all_keyword_based_classification(data, model_load):
    all_preds = []
    all_certainities = []
    all_details = []

    # print(len(data))

    for text in data:
        rule, conf = keyword_based_classification(text, model_load)
        #TODO: Better way to find assiging Keywords based Confidence
        details = {}
        for idx in range(0, 9):
            if idx == rule:
                details[idx] = conf
            else:
                details[idx] = 0
        all_preds.append(rule)
        all_certainities.append(conf)
        all_details.append(details)

    return all_preds, all_certainities, all_details

def predict_ml_hs(data, model_load):
    tokenizer = model_load.get_tokenizer()
    model = model_load.get_model()
    data_collator= model_load.get_data_collator()
    softmax = model_load.get_softmax()
    device = model_load.get_device()

    #local_rank = -1
    max_seq_length = 256
    batch_size = 1


    text_encodings = tokenizer(data, truncation=True, padding=True, max_length=max_seq_length)

    text_labels = [0] * len(text_encodings.input_ids) #This is dummay label for now

    # print(len(data),len(text_encodings),len(text_labels))

    text_dataset = HRDataset(text_encodings,text_labels)
    sampler = SequentialSampler(text_dataset)
    test_dataloader = DataLoader(text_dataset, collate_fn=data_collator, batch_size=batch_size, sampler=sampler)

    # data_iterator = tqdm(test_dataloader, desc="Iteration")

    all_preds = []
    all_certainities = []
    all_details = []


    for step, batch  in enumerate(test_dataloader):
        batch = batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits

        logits = softmax(logits) #This is probability
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).tolist()
        certainities = np.amax(logits, axis=1).tolist()

        details = {}

        for idx in range(0, 9):
            logit = logits[:, idx][0]
            details[idx] = logit

        all_preds.extend(preds)
        all_certainities.extend(certainities)
        all_details.append(details)


    return all_preds, all_certainities, all_details

def final_prediction(data, model_load):
    # print(len(data))
    label=[]
    confidence=[]
    details =[]
    #TODO: Could reduce time by doing ML prediction only when Keyword say pass
    key_preds, key_certainities, key_details = all_keyword_based_classification(data, model_load)

    ml_data = []

    for key_pred, text in zip(key_preds,data):
        if key_pred == 0:
            ml_data.append(text)

    if len(ml_data)>0:
        ml_preds, ml_certainities, ml_details = predict_ml_hs(ml_data, model_load)

    # print(len(key_preds))
    
    ml_idx = 0
    for i in range(len(data)):
        key_pred = key_preds[i]
        if key_pred !=0:
            label.append(key_preds[i])
            confidence.append(key_certainities[i])
            details.append(key_details[i])
        else:
            label.append(ml_preds[ml_idx])
            confidence.append(ml_certainities[ml_idx])
            details.append(ml_details[ml_idx])
            ml_idx +=1

    preds_class = []
    preds_class_details=[]
    preds_rule = []
    for i in range(len(label)):
        tmp_pred = label[i]
        if str(tmp_pred) == '0':
            pred = 'PASS'
            rule = 'PASS'
        else:
            pred = 'FAIL'
            rule = "RULE-" + map_24h_internal_rule[str(tmp_pred)]
        preds_class.append(pred)
        preds_rule.append(rule)
        logits = details[i]
        detail = {}
        for idx in range(0, 9):
            logit = logits[idx]
            if idx == 0:
                detail["PASS"] = logit
            else:
                rule = "RULE-" + map_24h_internal_rule[str(idx)]
                detail[rule] = logit
        detail = dict(sorted(detail.items()))
        preds_class_details.append(detail)
    return preds_class, confidence,preds_rule,preds_class_details
model_load = None
        
def predict(data):
    global model_load
    if model_load is None:
        model_load = ModelLoad()
    return final_prediction(data, model_load)

# if __name__ == "__main__":
#
#     data = pd.read_csv('/home/mladen/Desktop/data/24sata/test_mod_rule_selected_24sata.csv')
#
#     data = data.sample(frac=1)
#     data = data.head(100)
#     data = data.reset_index(drop=True)
#
#     # data
#     text = []
#     for i, t in enumerate(data.content.values):
#         text.append(t)
#         # if i > 100:
#         #     break
#     print(len(text))
#     # predict(text)
#     preds_class, confidence,preds_rule,detail = predict(text)
#     print(len(preds_class))
#     print(len(confidence))
#     print(len(detail))
