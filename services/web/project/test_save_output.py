import torch
import pandas as pd
import argparse
import logging
import numpy as np
import os
from tqdm import tqdm, trange

from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments,DataCollatorWithPadding
from transformers import Trainer
logger = logging.getLogger(__name__)

# import datasets
from datasets import load_metric
from datasets import load_from_disk

import re
from lingua import Language, LanguageDetectorBuilder

from langdetect import detect

lang_detector = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
valid_langs= ['BOSNIAN', 'SERBIAN','CROATIAN','MONTENEGRIN','ENGLISH', 'SLOVENE','SLOVAK', 'hr','sr','bs','cnr','en','sl','slv']

not_allowed_words = ['24sata', 'admin', 'Adolf', 'Hitler', 'jebem', 'jebemti', 'pička', 'lezba', 'majmun', 
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
                     'kobilo', 'degenerik', 'panglu', 'tenkre', 'smeće', 'sme', 'smece', 'sponzoruša', 'sponzorusa', 
                     'sponz', 'konju', 'krivousti', 'krivou', 'hanzek', 'hanžek', 'lešinari', 'lesinari', 
                     'ološ', 'olos', 'papcino', 'papak', 'papčino', 'bosanko', 'bosanđeros', 'bosanceros',
                     'bosancina', 'hercegovance', 'šupak', 'šupci', 'supak', 'supci', 'bosančeros', 'kuja', 
                     'kujica', 'dementra', 'dementna', 'nakaza', 'katolibani', 'talibani', 'papčina', 'kuraba',
                     'ganci', 'ljadro', 'retard', 'paksu', 'droca', 'express', 'srba', 'srbi', 'expres' ,
                     'šuft', 'suft', 'ćifut', 'katoliban', 'kolje', 'klati', 'kolj', 'jambrusic', 'jambrušić',
                     'tolusic', 'tolušić', 'siptar', 'balija', 'droca', 'acab', 'a.c.a.b.', 'radman', 
                     'selekcija', 'sjajna zvijezdo', 'sjajna zvjezdo', 'celofanka', 'kravo', 'kobila', 
                     'samoprozvani doktor za ljubav', 'drkolinda', 'poturica', 'poturico', 'isprdak']

# ' fasist',
#  ' fasista',
#  ' fasiste',
#  ' fasisti',
#  ' fasizam',
#  ' fasizma',
#  ' fasizmu',


# admin
# admin01
# admin24
# admin_1
# admin_24sata
# admin_7
# admina
# adminama
# admincek
# admincic
# admincice
# admincicu
# admincino
# admine
# adminee
# admineee
# admineeee
# admineeeee
# admineeeeeee
# admineeeeeeee
# admineeeeeeeee
# admineeeeeeeeeeee
# admineeeeeeeeeeeeeeeeeeeeee
# adminejo
# adminhe
# admini
# adminica

# adolf
# adolfa
# adolfe
# adolfeee
# adolfeeeee
# adolfima
# adolfinjo
# adolfland
# adolfom
# adolfov
# adolfova
# adolfu

# 24casa
# 24casova
# 24cetnicki
# 24cetnickoga
# 24dobilo
# 24glupost
# 24guzice
# 24h
# 24kurac
# 24kurca
# 24kurira
# 24pisamponjimasata
# 24politka
# 24pošto
# 24s4ta
# 24sat
# 24sata
# 24satajebovampasmater
# 24satara
# 24satni
# 24satno
# 24satnog
# 24smrada
# 24srbija
# 24sta
# 24trollsata
# 24ur
# 24ura
# 24ure
# 24v
# 24vsata
# 24yure
# 24zonu
# 24časa
# 24časnom
# 24časova
# 24časovci
# 24časovni
# 24časta
# 24čuke
# 25sata


word_to_rule = {'24sata':1}
def check_blocked_words(text):
    # Simply based on the texonomy 
    rule = -1
    conf = -1
    rule_flag = False

    for word in not_allowed_words:
        match = re.search(word,text)
        if match:
            rule_flag = True
            found_word = word #TODO: This misses if multiple words are present
            break
    
    if rule_flag:
        try:
            rule = word_to_rule[found_word]
        except:
            rule = 3
        conf = 0.9
    
    return rule, conf


def check_rule_seven(text):
    # To check based on the language and upper case
    rule = -1
    conf = -1
    rule_flag = False
    if text.isupper():
        rule_flag = True
    else:
        text = " ".join(re.findall("[a-zA-Z]+", text))
        words = text.split(' ')
        c_w = len(words)
        if c_w > 3:
            # Check in two instances 
            val1 = " ".join(words[:int(c_w/2)])
            val2 = " ".join(words[int(c_w/2):])
            for val in [val1, val2,value]:
                try:
                    lang2 = detect(val) #Easy to run 
                except:
                    lang2 = 'hr'
                if lang2 not in valid_langs:
                    try:
                        lang1 = str(detector.detect_language_of(val))[9:]
                    except:
                        lang1 = 'CROATIAN'

                    if lang1 in valid_langs:
                       rule_flag = True
                       break
                else:
                    rule_flag = True
                    break
    if rule_flag:
        rule = 7
        conf = 0.9
    return rule, conf

def compute_metrics(eval_preds):

    metric_acc = load_metric("accuracy")
    metric_f1 = load_metric("f1")

    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    acc = metric_acc.compute(predictions=predictions, references=labels)
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="macro")

    return {"acc": acc, "f1": f1}

def fix_metrics(metrics):
    # Fix metrics in dict format for logging purpose
    for key in metrics.keys():
        if isinstance(metrics[key], dict):
            for key1 in metrics[key].keys():
                print(metrics[key][key1])
                metrics[key] = metrics[key][key1]
    return metrics

def read_data(file_name, small_dataset=False):
    #Reading CSV File

    df = pd.read_csv(file_name, lineterminator='\n')
    df.label = df.label.astype(int)
    if small_dataset:
        df = df.sample(frac=1)
        df = df.head(1000)
    print('Processing', file_name, df.shape)
    
    texts= df.content.tolist()
    labels = df.label.tolist()

    return texts, labels


class HRDataset(torch.utils.data.Dataset):
    #24Sata Dataset Processing
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        # item['idx'] = torch.tensor(self.idxs[idx])
        return item

    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #Dataset
    parser.add_argument("--test_file", type=str, default='/import/cogsci/ravi/datasets/24sata/test_rule_selected_24sata.csv', help='Test File')
    parser.add_argument("--encode_data", type=bool, default=False, help='Encode data')
    parser.add_argument("--batch_size", type=int, default=24, help='batch size')

    #Model
    parser.add_argument("--model_dir", type=str, default='./results/claasify/csebert_rule/', help='The model directory checkpoint for weights initialization.')
   
    parser.add_argument("--out_file", type=str, default='test_outputs.csv',help='output file')
    parser.add_argument("-small_dataset", action='store_true', help='To Train on small dataset')
    
    args = parser.parse_args()

    small_dataset=args.small_dataset
    test_file = args.test_file
    model_dir= args.model_dir
    batch_size = args.batch_size
    out_file = model_dir + args.out_file
    max_seq_length = 256

    #Only done once because it assumes main directory will have same pre-processing part
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # TODO: Maybe want to save the dataset, so that processing is less
    test_texts, test_labels = read_data(test_file,small_dataset=small_dataset)
    test_encodings = tokenizer(test_texts,truncation=True, padding=True,max_length=max_seq_length)
    test_dataset = HRDataset(test_encodings, test_labels)

    sampler = SequentialSampler(test_dataset)
    
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=batch_size, sampler=sampler)

    data_iterator = tqdm(test_dataloader, desc="Iteration")
    softmax = torch.nn.Softmax(dim=-1)
    
    device = torch.cuda.current_device()

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    model.to(device)

    all_idx = []
    all_preds = []
    all_probs = []
    all_probs_rules = []
    for step, batch in enumerate(data_iterator):

        batch = batch.to(device)
        outputs = model(**batch)

        # batch_idx = batch['idx'].tolist()
        logits = outputs.logits
        logits = softmax(logits)
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).tolist()
        probs = np.amax(logits, axis=1).tolist()

        for logit in logits:
            all_probs_rules.append(logit.tolist())
        # all_idx.append(batch_idx)
        all_preds.extend(preds)
        all_probs.extend(probs)

    result = pd.DataFrame([all_preds, all_probs,all_probs_rules])
    result = result.transpose()
    result.columns = ['embeddia_rule', 'result', 'all_result']
    result.head()
    result.embeddia_rule = result.embeddia_rule.astype(int)

    result.to_csv(out_file, index=False)
    print('Output saved to ', out_file)


