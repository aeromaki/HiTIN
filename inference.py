#!/usr/bin/env python
# coding:utf-8
import helper.logger as logger
from models.model import HiAGM
import torch
from helper.configure import Configure
import os
from data_modules.data_loader import data_loaders
from data_modules.vocab import Vocab
from train_modules.criterions import ClassificationLoss
from train_modules. trainer import Trainer
from helper.utils import load_checkpoint, save_checkpoint
from helper.arg_parser import get_args

import time
import random
import numpy as np
import pprint
import warnings

from transformers import BertTokenizer
from helper.lr_schedulers import get_linear_schedule_with_warmup
from helper.adamw import AdamW

import tqdm

warnings.filterwarnings("ignore")

import helper.logger as logger
from train_modules.evaluation_metrics import evaluate


#!/usr/bin/env python
# coding: utf-8

from torch.utils.data.dataset import Dataset
import helper.logger as logger
import json
import os

from data_modules.collator import Collator
from torch.utils.data import DataLoader


from data_modules.preprocess import preprocess_line

class InferenceData(Dataset):
    def __init__(
        self,
        config,
        vocab,
        corpus_file,
        tokenizer=None
    ):
        """
        Dataset for text classification based on torch.utils.data.dataset.Dataset
        :param config: helper.configure, Configure Object
        :param vocab: data_modules.vocab, Vocab Object
        :param corpus_lines: List[Str], on-memory Data
        """
        super().__init__()

        self.config = config
        self.vocab = vocab
        self.max_input_length = self.config.text_encoder.max_length

        self.corpus_lines = [*map(preprocess_line, open(corpus_file, "r").readlines())]
        self.data = self.corpus_lines

        self.sample_position = range(len(self.corpus_lines))
        self.corpus_size = len(self.sample_position)

        self.tokenizer = tokenizer

    def __len__(self):
        """
        get the number of samples
        :return: self.corpus_size -> Int
        """
        return self.corpus_size

    def __getitem__(self, index):
        """
        sample from the overall corpus
        :param index: int, should be smaller in len(corpus)
        :return: sample -> Dict{'token': List[Str], 'label': List[Str], 'token_len': int}
        """
        if index >= self.__len__():
            raise IndexError
        sample_str = self.data[index]
        return self._preprocess_sample(sample_str)

    def create_features(self, sentences, max_seq_len=256):
        tokens = self.tokenizer.tokenize(sentences)

        if len(tokens) > max_seq_len - 2:
            tokens = tokens[:max_seq_len - 2]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        segment_ids = [0] * len(tokens)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_len - len(input_ids))
        input_len = len(input_ids)

        input_ids   += padding
        input_mask  += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len

        feature = {'input_ids': input_ids, 'input_mask': input_mask, 'segment_ids': segment_ids, 'input_len': input_len}
        return feature

    def _preprocess_sample(self, sample_str):
        """
        preprocess each sample with the limitation of maximum length and pad each sample to maximum length
        :param sample_str: Str format of json data, "Dict{'token': List[Str], 'label': List[Str]}"
        :return: sample -> Dict{'token': List[int], 'label': List[int], 'token_len': int}
        """
        raw_sample = json.loads(sample_str)
        sample = {'token': [], 'label': []}
        for k in raw_sample.keys():
            if k == 'token':
                sample[k] = [self.vocab.v2i[k].get(v.lower(), self.vocab.oov_index) for v in raw_sample[k]]

                if self.config.text_encoder.type == "bert":
                    sentences = " ".join(raw_sample[k])
                    features = self.create_features(sentences, self.max_input_length)
                    for (features_k, features_v) in features.items():
                        sample[features_k] = features_v
            else:
                sample[k] = []
                for v in raw_sample[k]:
                    if v not in self.vocab.v2i[k].keys():
                        logger.warning('Vocab not in ' + k + ' ' + v)
                    else:
                        sample[k].append(self.vocab.v2i[k][v])
        if not sample['token']:
            sample['token'].append(self.vocab.padding_index)
        sample['label'] = [0]
        sample['token_len'] = min(len(sample['token']), self.max_input_length)
        padding = [self.vocab.padding_index for _ in range(0, self.max_input_length - len(sample['token']))]
        sample['token'] += padding
        sample['token'] = sample['token'][:self.max_input_length]
        return sample




class Inference(object):
    def __init__(self, model, vocab, config):
        """
        :param model: Computational Graph
        :param vocab: vocab.v2i -> Dict{'token': Dict{vocabulary to id map}, 'label': Dict{vocabulary
        to id map}}, vocab.i2v -> Dict{'token': Dict{id to vocabulary map}, 'label': Dict{id to vocabulary map}}
        :param config: helper.Configure object
        """
        super().__init__()
        self.model = model
        self.vocab = vocab
        self.config = config

    def run(self, data):
        """
        :param data: Dict{'token':tensor, 'label':tensor, }
        """

        logits = self.model(data)
        if self.config.train.loss.recursive_regularization.flag:
            if self.config.structure_encoder.type == "TIN":
                recursive_constrained_params = [m.weight for m in self.model.hiagm.graph_model.model.model[0].linears_prediction]
            else:
                recursive_constrained_params = self.model.hiagm.linear.weight
        else:
            recursive_constrained_params = None

        sig = logits.sigmoid()

        return (sig > 0.5).argwhere().cpu()



def go(config, args):
    """
    :param config: helper.configure, Configure Object
    """
    # loading corpus and generate vocabulary
    corpus_vocab = Vocab(config,
                         min_freq=5,
                         max_size=50000)

    print("vocab ready")

    #################################################################################

    if config.text_encoder.type == "bert":
        tokenizer = BertTokenizer.from_pretrained(config.text_encoder.bert_model_dir)
    else:
        tokenizer = None

    #################################################################################

    inference_data = InferenceData(
        config,
        corpus_vocab,
        corpus_file="./inference_input",
        tokenizer=tokenizer
    )

    collate_fn = Collator(config, corpus_vocab)
    data_loader = DataLoader(inference_data,
            batch_size=len(inference_data),
            num_workers=config.train.device_setting.num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=True)
    data = next(iter(data_loader))

    print("data ready")

    #################################################################################

    # build up model
    hiagm = HiAGM(config, corpus_vocab, model_type=config.model.type, model_mode='EVAL')

    hiagm.to(config.train.device_setting.device)

    print("model loaded")

    #################################################################################

    load = torch.load("ckpt/1209_0918_wos_tin/best_macro_HiAGM-TP_2_128_sum_0.5_0")

    hiagm.load_state_dict(load["state_dict"])

    print(load["best_performance"])

    print("checkpoint loaded")

    #################################################################################

    inference = Inference(model=hiagm,
                      vocab=corpus_vocab,
                      config=config)

    print("inference ready")

    #################################################################################

    res = inference.run(data)
    pprint.pprint([*map(lambda x: [*map(corpus_vocab.i2v["label"].get, x)], res.tolist())])

    #################################################################################

    return


if __name__ == "__main__":
    args = get_args()
    pprint.pprint(vars(args))
    configs = Configure(config_json_file=args.config_file)
    configs.update(vars(args))

    if configs.train.device_setting.device == 'cuda':
        os.system('CUDA_VISIBLE_DEVICES=' + str(configs.train.device_setting.visible_device_list))
    else:
        os.system("CUDA_VISIBLE_DEVICES=''")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.multiprocessing.set_start_method('spawn')

    logger.Logger(configs)

    # if not os.path.isdir(configs.train.checkpoint.dir):
    #     os.mkdir(configs.train.checkpoint.dir)

    # train(config)
    go(configs, args)