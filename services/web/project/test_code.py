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
import pickle

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader,  SequentialSampler


from transformers import BertTokenizer, BertConfig, BertForSequenceClassification


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

# map_24h_internal_rule={"0":"0","1":"5","2":"1","3":"2","4":"3","5":"4","6":"6","7":"7","8":"8"}

class ModelLoad():

    def __init__(self):
        # configuration
        self.ROOT_FOLDER = '/import/cogsci/ravi/codes/comment-filter-24h-internal-rule/services/web/project/'

        print('ROOT_FOLDER',self.ROOT_FOLDER)


        # Load a trained model that you have fine-tuned
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model = 'EMBEDDIA/crosloengual-bert' #EMBEDDIA Model

        DIRECTORIES = {
            'ml_hate_speech_path': os.path.join(self.ROOT_FOLDER, 'models/ml_hate_speech_classifier', self.bert_model)
        }
        self.model_file = os.path.join(DIRECTORIES['ml_hate_speech_path'], 'pytorch_model.bin')
        self.model_file_nb = os.path.join(DIRECTORIES['ml_hate_speech_path'], 'nb_model_bigram.sav')

        print('model_file',self.model_file)
        print('model_dir',os.listdir(os.path.join(self.ROOT_FOLDER, 'models')))
        print('model_dir_s',os.listdir(DIRECTORIES['ml_hate_speech_path']))
        print(os.path.isfile(self.model_file))
        if not os.path.isfile(self.model_file):
            print('Please Download the model ...')
            exit(0)


        config_file = DIRECTORIES['ml_hate_speech_path']+'/config.json'
        token_file = DIRECTORIES['ml_hate_speech_path']+'/vocab.txt'

        config = BertConfig.from_pretrained(config_file, num_labels=9)
        self.tokenizer = BertTokenizer.from_pretrained(token_file, do_lower_case=False)

        print('Loading model ...', self.model_file)

        self.model = BertForSequenceClassification.from_pretrained(self.model_file, config=config)
        self.model_nb = pickle.load(open(self.model_file_nb, 'rb'))

        self.model.eval()
        self.model.to(self.device)


    def load_models(self):

        #Return Model

        return self.model,self.tokenizer

    def get_model(self):
        # Return Model

        return self.model

    def get_model_nb(self):
        # Return Model

        return self.model_nb

    def get_tokenizer(self):
        return self.tokenizer

    def get_device(self):
        return self.device
    

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

        
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_csv(cls, input_file, quotechar=None):
        """Reads a comma separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=",", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_data(cls, data):
        """Reads a tab separated value file."""
        lines = []
        for line in data:
            lines.append(line)
        # print("Lines: ", lines)
        return lines



class SemEvalProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "offenseval-training-v1_train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "offenseval-training-v1_train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "offenseval-training-v1_eval.tsv")), "dev")

    def get_test_examples(self, data):
        return self._create_examples(
            self._read_data(data), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        if set_type == 'dev':
            for (i, line) in enumerate(lines):
                #if i == 0:
                #    continue
                guid = "%s-%s" % (set_type, i)
                text_a = line[0]
                text_b = None
                label = "0" if line[1] == 'OFF' else "1"
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        elif set_type == 'train':
            for (i, line) in enumerate(lines):
                #if i == 0:
                #    continue
                guid = "%s-%s" % (set_type, i)
                text_a = line[0]
                text_b = None
                label = "0" if line[1] == 'OFF' else "1"
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            
        elif set_type == 'test':
            for (i, line) in enumerate(lines):
                #if i == 0:
                #    continue
                guid = str(i)
                text_a = line
                text_b = None
                label = "0"
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    if label_list:
        label_map = {label : i for i, label in enumerate(label_list)}
    else:
        label_map = {"0": i for i in range(len(examples))}

    features = []
    for (ex_index, example) in enumerate(examples):

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        # print(tokens)
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]


        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def predict_nb(text,model_nb):
    predicted = model_nb.predict(text)
    predicted_prob = model_nb.predict_probs(text)
    return  predicted,  predicted_prob

def predict_ml_hs(data, tokenizer, model, model_nb, device):

    #local_rank = -1
    max_seq_length = 128
    eval_batch_size = 1
    task_name = 'semeval'
    processors = {
        "semeval": SemEvalProcessor,
    }

    processor = processors[task_name]()
    test_examples = processor.get_test_examples(data)

    test_features = convert_examples_to_features(
        test_examples, None, max_seq_length, tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=eval_batch_size)

    model.eval()

    all_preds = []
    all_certainities = []
    all_details = []
    all_cassifier = []


    ###### NB prediction ###########
    nb_predicts = model_nb.predict(data)
    nb_predicts_prob = model_nb.predict_proba(data)

    counter = 0
    softmax = torch.nn.Softmax(dim=1)

    for input_ids, input_mask, segment_ids, label_ids in test_dataloader:

        # pred = nb_predicts[counter]
        # certainities = nb_predicts_prob[counter]

        #Only if NB is not predicting Rule 1.
        #TODO: Need to remove NB in future, no real benifit
        
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask)
            logits = outputs[0]

        logits = softmax(logits)
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).tolist()
        certainities = np.amax(logits, axis=1).tolist()
            
        details = {}
        #TODO: check something looks wrong here
        for idx in range(0,9):
            logit = logits[:,idx][0]

            if idx == 0:
                details["PASS"]=logit
            else:
                rule = "RULE-"+str(idx)
                details[rule]=logit
        # else:
        #     rule = "RULE-1"
        #     details[rule]=logit

        if len(certainities)>1:
                print(certainities)
        all_preds.extend(preds)
        all_certainities.extend(certainities)
        all_details.append(details)

        counter = counter +1

        # print(len(all_preds), len(all_certainities), len(all_details))

    preds_class = []
    rule = []
    for i in range(len(all_preds)):
        tmp_pred = all_preds[i]
        if str(tmp_pred) == '0':
            pred = 'PASS'
        else:
            pred = 'FAIL'
        preds_class.append(pred)
        rule.append(tmp_pred)

    # print(len(preds_class), len(all_certainities), len(all_details))

    return preds_class, all_certainities, all_details, rule


model_load = None
        
def predict(data):
    global model_load
    if model_load is None:
        model_load = ModelLoad()

    return predict_ml_hs(data, model_load.get_tokenizer(), model_load.get_model(), model_load.get_model_nb(), model_load.get_device())


        
if __name__ == "__main__":
    import pandas as pd
    data = pd.read_csv('/import/cogsci/ravi/datasets/24sata/csv_embeddia_export.csv')

    content = data.content.values.tolist()
    text = content
    # c = 1064468

    # for i, t in enumerate(data[data['infringed_on_rule'] == 1].content.values):
    #     text.append(t)
    #     if i > 2:
    #         break
    label, confidence,detail, rule = predict(text)
    print(len(text), len(label), len(confidence), len(detail))
    df = pd.DataFrame({'content':text, 'block':label,'embeddia_rule':rule, 'result':confidence, 'details':detail})

    df.to_csv('output.csv',index=False)
    
