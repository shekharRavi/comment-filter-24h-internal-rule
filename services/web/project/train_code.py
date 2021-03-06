import torch
import pandas as pd
import argparse
import logging
import numpy as np
import os

from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments,DataCollatorWithPadding
from transformers import Trainer
logger = logging.getLogger(__name__)

# import datasets
from datasets import load_metric
from datasets import load_from_disk

import wandb
wandb.init(project="train-classification", entity="hahackathon")
# os.environ["WANDB_DISABLED"] = "true"

from sklearn.utils import class_weight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_cards = {
"mbert":'bert-base-multilingual-cased',
"csebert":'EMBEDDIA/crosloengual-bert',
}

datasets = {
'rule':{
        'train_file':'/import/cogsci/ravi/datasets/24sata/train_rule_selected_24sata.csv',
        'val_file':'/import/cogsci/ravi/datasets/24sata/val_rule_selected_24sata.csv',
        'test_file':'/import/cogsci/ravi/datasets/24sata/test_rule_selected_24sata.csv',
    },
'rule_mod':{ #This is new data by adding 2022 tested on the EMBEDDIA system 
        'train_file':'/import/cogsci/ravi/datasets/24sata/train_mod_rule_selected_24sata.csv',
        'val_file':'/import/cogsci/ravi/datasets/24sata/val_mod_rule_selected_24sata.csv',
        'test_file':'/import/cogsci/ravi/datasets/24sata/test_mod_rule_selected_24sata.csv',
    },

}

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
        df = df.head(10000)
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

        # for key in item.keys():
        #     print(key, item[key].shape,item[key])
        return item

    def __len__(self):
        return len(self.labels)

class CustomTrainer(Trainer):
    def __init__(self, class_weights=torch.FloatTensor([1.0, 1.0, 1.0,1.0, 1.0, 1.0,1.0, 1.0, 1.0]), *args, **kwargs):
        super().__init__(*args, **kwargs)
        # You pass the class weights when instantiating the Trainer
        self.class_weights = class_weights
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Training Parameters
    parser.add_argument("--output_dir", type=str, default="./results/claasify/", help='Output Directory')
    parser.add_argument("--logging_dir", type=str, default="./logs/claasify/", help='Logging Directory')
    parser.add_argument("--num_train_epochs", type=int, default=10, help='Number of training Epochs')
    parser.add_argument("--per_device_train_batch_size", type=int, default=24, help='Traiing Batch Size')
    parser.add_argument("--per_device_eval_batch_size", type=int, default=24, help='Evaluation Batch Size')
    parser.add_argument("--warmup_steps", type=int, default=500, help='Warmup Steps')
    parser.add_argument("--weight_decay", type=int, default=0.001, help='Weight Decay Rate')
    parser.add_argument("--logging_steps", type=int, default=1000, help='Logging Steps')
    parser.add_argument("--save_steps", type=int, default=5000, help='Number of updates steps before two checkpoint saves')
    parser.add_argument("--save_total_limit", type=int, default=50, help='If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints')
    parser.add_argument("--save_strategy", type=str, default="epoch", help='[no/steps/epoch]The checkpoint save strategy to adopt during training')
    parser.add_argument("--learning_rate", default=2e-5, type=float)

    #Dataset
    parser.add_argument("--dataset", type=str, default='rule_mod', help='Training validation set large/small')

    #Model
    parser.add_argument("--model_card", type=str, default='csebert', help='The model directory checkpoint for weights initialization.')
    parser.add_argument("-all_steps", action='store_true', help='To Train on all steps check point')
    parser.add_argument("-small_dataset", action='store_true', help='To Train on small dataset')

   
    args = parser.parse_args()

    output_dir = args.output_dir
    logging_dir = args.logging_dir
    num_train_epochs = args.num_train_epochs
    per_device_train_batch_size = args.per_device_train_batch_size
    per_device_eval_batch_size = args.per_device_eval_batch_size
    learning_rate = args.learning_rate
    warmup_steps = args.warmup_steps
    weight_decay = args.weight_decay
    logging_steps = args.logging_steps
    save_steps = args.save_steps
    save_total_limit = args.save_total_limit
    save_strategy = args.save_strategy
    max_seq_length = 256
    small_dataset=args.small_dataset
    

    gradient_accumulation_steps = 8

    dataset = args.dataset
    train_file = datasets[dataset]['train_file']
    val_file = datasets[dataset]['val_file']
    # encode_data = args.encode_data
    model_card= args.model_card
    all_steps = args.all_steps

    model_dirs  = []
    model_dir = model_cards[model_card]
    model_dirs.append(model_dir)
    
    print(model_dirs)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    #TODO: Maybe want to save the dataset, so that processing is less
    train_texts, train_labels = read_data(train_file,small_dataset=small_dataset)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_seq_length)
    train_dataset = HRDataset(train_encodings, train_labels)
    val_texts, val_labels = read_data(val_file,small_dataset=small_dataset)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True,max_length=max_seq_length)
    val_dataset = HRDataset(val_encodings, val_labels)

    # encoded_dataset.save_to_disk("path/of/my/dataset/directory")
    # reloaded_encoded_dataset = load_from_disk("path/of/my/dataset/directory")

    class_weights = torch.FloatTensor(class_weight.compute_class_weight('balanced',
                                                         classes=np.unique(train_labels),
                                                         y=train_labels)).to(device)

    print(np.unique(train_labels))

    for model_dir in model_dirs:
        tmp_output_dir = output_dir + model_card+'_'+dataset
        tmp_logging_dir = logging_dir + model_card+'_'+dataset

        print('Saving to ', tmp_output_dir, tmp_logging_dir)

        training_args = TrainingArguments(
            output_dir=tmp_output_dir,  # output directory
            logging_dir=tmp_logging_dir,  # directory for storing logs
            num_train_epochs=num_train_epochs,  # total number of training epochs
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_train_batch_size=per_device_train_batch_size,  # batch size per device during training
            per_device_eval_batch_size=per_device_eval_batch_size,  # batch size for evaluation
            learning_rate = learning_rate,
            warmup_steps=warmup_steps,  # number of warmup steps for learning rate scheduler
            weight_decay=weight_decay,  # strength of weight decay
            logging_steps=logging_steps,
            save_total_limit = save_total_limit,
            save_strategy=save_strategy,
            save_steps=save_steps,
            report_to="wandb", #Log into Weight and Bias
            evaluation_strategy="steps" #Evaluate at very logging steps

        )

        model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=9, classifier_dropout=0.1)

        trainer = Trainer(
            model=model,  # the instantiated Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=val_dataset,  # evaluation dataset
            tokenizer = tokenizer,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            # class_weights=class_weights
        )

        train_result = trainer.train(resume_from_checkpoint=None)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # Evaluation
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics = fix_metrics(metrics)
        metrics["eval_samples"] = len(val_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


