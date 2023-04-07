# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
""" XNLI utils (dataset loading and evaluation) """


import logging
import os
import json
import random

from transformers import DataProcessor
from .utils import InputExample

logger = logging.getLogger(__name__)


class AmazonProcessor(DataProcessor):
    def __init__(self):
        pass

    def get_examples(self, data_dir, language='en', split='train'):
      """See base class."""
      examples = []
      for lg in language.split(','):
        #lines = self._read_tsv(os.path.join(data_dir, "{}-{}.tsv".format(split, lg)))
        file= open(os.path.join(data_dir, "amazon_{}_{}.json".format(split, lg)), "r", encoding="utf-8")
        lines =[]
        for line in file.readlines():
            dic=json.loads(line)
            lines.append(dic)
        
        for (i, line) in enumerate(lines):
          guid = "%s-%s-%s" % (split, lg, i)
          text = line['review_body']
          label = str(line['stars'])

          assert isinstance(text, str) and isinstance(label, str)
          examples.append(InputExample(guid=guid, text_a=text, label=label, language=lg))
      return examples

    def get_train_examples(self, data_dir, language='en'):
        examples=self.get_examples(data_dir, language, split='train')
        random.shuffle(examples)
        return examples

    def get_dev_examples(self, data_dir, language='en'):
        return self.get_examples(data_dir, language, split='dev')

    def get_test_examples(self, data_dir, language='en'):
        return self.get_examples(data_dir, language, split='test')

    def get_translate_train_examples(self, data_dir, language='en'):
        """See base class."""
        examples = []
        for lg in language.split(','):
            file_path = os.path.join(data_dir, "AMAZON-Translated/en-{}-translated.tsv".format(lg))
            logger.info("reading file from " + file_path)
            file= open(file_path)
            lines =[]
            for line in file.readlines():
                dic=json.loads(line)
                lines.append(dic)
            for (i, line) in enumerate(lines):
                guid = "%s-%s-%s" % ("translate-train", lg, i)
                text = line['review_body']
                label = str(line['stars'])
                assert isinstance(text, str) and isinstance(label, str)
                examples.append(InputExample(guid=guid, text_a=text, label=label, language=lg))
        return examples

    def get_translate_test_examples(self, data_dir, language='en'):
        lg = language
        file= open(os.path.join(data_dir, "AMAZON-Translated/test-{}-en-translated.tsv".format(lg)))
        lines =[]
        for line in file.readlines():
            dic=json.loads(line)
            lines.append(dic)
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s-%s" % ("translate-test", language, i)
            text = line['review_body']
            label = str(line['stars'])
            assert isinstance(text, str) and isinstance(label, str)
            examples.append(InputExample(guid=guid, text_a=text, label=label, language=lg))
        return examples
        
    def get_pseudo_test_examples(self, data_dir, language='en'):
        file= open(os.path.join(data_dir, "XNLI-Translated/pseudo-test-set/en-{}-pseudo-translated.csv".format(language)))
        lines =[]
        for line in file.readlines():
            dic=json.loads(line)
            lines.append(dic)
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s-%s" % ("translate-test", language, i)
            text = line['review_body']
            label = str(line['stars'])
            assert isinstance(text, str) and isinstance(label, str)
            examples.append(InputExample(guid=guid, text_a=text, label=label, language=lg))
        return examples

    def get_labels(self):
        """See base class."""
        #return ["contradiction", "entailment", "neutral"]
        return ["1","2","3","4","5"]

amazon_processors = {
    "amazon": AmazonProcessor,
}

amazon_output_modes = {
    "amazon": "classification",
}

amazon_tasks_num_labels = {
    "amazon": 5,
}

"""
xnli_processors = {
    "xnli": XnliProcessor,
}

xnli_output_modes = {
    "xnli": "classification",
}

xnli_tasks_num_labels = {
    "xnli": 3,
}
"""
