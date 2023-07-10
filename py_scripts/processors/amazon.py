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
""" Amazon utils (dataset loading and evaluation) """

import logging
import os
import random

from datasets import load_dataset
from transformers import DataProcessor
from .utils import InputExample

logger = logging.getLogger(__name__)


class AmazonProcessor(DataProcessor):
    """Processor for the Amazon dataset."""
    
    def __init__(self):
        pass

    def get_examples(self, data_dir, language='en', split='train', num_sample=-1):
        """See base class."""
        examples = []
        split = 'validation' if split=='dev' else split
        for lg in language.split(','):
            dataset = load_dataset(data_dir, language, split=split)

            for (i, data_ex) in enumerate(dataset):
                guid = "%s-%s-%s" % (split, lg, i)
                text = data_ex['review_body']
                # if split == 'test' and len(line) != 3:
                #     label = "neutral"
                # else:
                #     label = str(line[2].strip())
                label = data_ex['stars']
                assert isinstance(text, str) and isinstance(label, int)
                examples.append(InputExample(guid=guid, text_a=text, label=label, language=lg))
        if num_sample != -1:
            # examples = random.sample(examples, num_sample)
            random.shuffle(examples)
            l0, l1, l2, l3, l4 = [], [], [], [], []
            labels = list(set([e.label for e in examples]))
            for example in examples:
                if example.label==labels[0] and len(l0)<num_sample:
                    l0.append(example)
                elif example.label==labels[1] and len(l1)<num_sample:
                    l1.append(example)
                elif example.label==labels[2] and len(l2)<num_sample:
                    l2.append(example)
                elif example.label==labels[3] and len(l3)<num_sample:
                    l3.append(example)
                elif example.label==labels[4] and len(l4)<num_sample:
                    l4.append(example)
                elif len(l0)==num_sample and len(l1)==num_sample and len(l2)==num_sample and len(l3)==num_sample and len(l4)==num_sample:
                    break
            examples = l0+l1+l2+l3+l4

        return examples

    def get_train_examples(self, data_dir, language='en', num_sample=-1):
        return self.get_examples(data_dir, language, split='train', num_sample=num_sample)

    def get_dev_examples(self, data_dir, language='en', num_sample=-1):
        return self.get_examples(data_dir, language, split='dev', num_sample=num_sample)

    def get_test_examples(self, data_dir, language='en', num_sample=-1):
        return self.get_examples(data_dir, language, split='test', num_sample=num_sample)

    def get_translate_train_examples(self, data_dir, language='en', num_sample=-1):
        """See base class."""
        examples = []
        for lg in language.split(','):
            file_path = os.path.join(data_dir, "XNLI-Translated/en-{}-translated.tsv".format(lg))
            logger.info("reading file from " + file_path)
            lines = self._read_tsv(file_path)
            for (i, line) in enumerate(lines):
                guid = "%s-%s-%s" % ("translate-train", lg, i)
                text_a = line[0]
                text_b = line[1]
                label = "contradiction" if line[2].strip() == "contradictory" else line[2].strip()
                assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=lg))
        if num_sample != -1:
            examples = random.sample(example, num_sample)
        return examples

    def get_translate_test_examples(self, data_dir, language='en', num_sample=-1):
        lg = language
        lines = self._read_tsv(os.path.join(data_dir, "XNLI-Translated/test-{}-en-translated.tsv".format(lg)))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s-%s" % ("translate-test", language, i)
            text_a = line[0]
            text_b = line[1]
            label = "contradiction" if line[2].strip() == "contradictory" else line[2].strip()
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=language))
        return examples

    def get_pseudo_test_examples(self, data_dir, language='en', num_sample=-1):
        lines = self._read_tsv(
            os.path.join(data_dir, "XNLI-Translated/pseudo-test-set/en-{}-pseudo-translated.csv".format(language)))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s-%s" % ("pseudo-test", language, i)
            text_a = line[0]
            text_b = line[1]
            label = "contradiction" if line[2].strip() == "contradictory" else line[2].strip()
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=language))
        return examples

    def get_labels(self):
        """See base class."""
        return [1, 2, 3, 4, 5]


xnli_processors = {
    "amazon": AmazonProcessor,
}

xnli_output_modes = {
    "amazon": "classification",
}

xnli_tasks_num_labels = {
    "amazon": 5,
}
