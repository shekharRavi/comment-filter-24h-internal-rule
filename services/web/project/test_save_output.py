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
    if small_dataset:
        df = df.sample(frac=1)
        df = df.head(1000)
    print('Processing', file_name, df.shape)
    idx= df.index.tolist()
    texts= df.content.tolist()
    labels = df.label.tolist()

    return idx, texts, labels


class HRDataset(torch.utils.data.Dataset):
    #24Sata Dataset Processing
    def __init__(self, encodings, labels,idxs):
        self.encodings = encodings
        self.labels = labels
        self.idxs = idxs

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['idx'] = torch.tensor(self.idxs[idx])
        return item

    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #Dataset
    parser.add_argument("--test_file", type=str, default='/import/cogsci/ravi/datasets/24sata/test_rule_selected_24sata.csv', help='Test File')
    parser.add_argument("--encode_data", type=bool, default=False, help='Encode data')
    parser.add_argument("--batch_size", type=int, default=64, help='batch size')

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

    #Only done once because it assumes main directory will have same pre-processing part
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # TODO: Maybe want to save the dataset, so that processing is less
    test_idxs, test_texts, test_labels = read_data(test_file,small_dataset=small_dataset)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)
    test_dataset = HRDataset(test_encodings, test_labels,test_idxs)

    sampler = SequentialSampler(test_dataset)
    
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=batch_size, sampler=sampler)

    data_iterator = tqdm(test_dataloader, desc="Iteration")
    softmax = torch.nn.Softmax(dim=-1)
    
    true_labels = []
    all_mean0 = []
    all_std0  = []
    all_mean1 = []
    all_std1 = []

    device = torch.cuda.current_device()

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    model.to(device)

    all_idx = []
    all_preds = []
    all_probs = []
    for step, batch in enumerate(data_iterator):

        batch = batch.to(device)
        outputs = model(**batch)

        batch_idx = batch['idx'].tolist()

        logits = softmax(logits)
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).tolist()
        probs = np.amax(logits, axis=1).tolist()

        all_idx.append(batch_idx)
        all_preds.append(preds)
        all_probs.append(probs)

    result = pd.DataFrame([all_idx, all_preds, all_probs])
    result = result.transpose()
    result.columns = ['idx', 'embeddia_rule', 'result']
    result.head()
    
    result.to_csv(out_file, index=False)
    print('Output saved to ', out_file)


