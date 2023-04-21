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
""" PAWS-X utils (dataset loading and evaluation) """

import logging
import os
import random
from collections import Counter
from transformers import DataProcessor
from processors.utils import InputExample

logger = logging.getLogger(__name__)


class PawsxProcessor(DataProcessor):
    """Processor for the PAWS-X dataset."""

    def __init__(self):
        pass

    def get_examples(self, data_dir, language='en', split='train', num_sample=-1):
        """See base class."""
        examples = []
        for lg in language.split(','):
            lines = self._read_tsv(os.path.join(data_dir, "{}-{}.tsv".format(split, lg)))

            for (i, line) in enumerate(lines):
                guid = "%s-%s-%s" % (split, lg, i)
                text_a = line[0]
                text_b = line[1]
                # if split == 'test' and len(line) != 3:
                #     label = "0"
                # else:
                label = str(line[2].strip())
                assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=lg))
        # print the data distribution
        labels= []
        for example in examples:
          labels.append(example.label)
        labels_count=Counter(labels)
        print(labels_count)
        if num_sample != -1:
            # examples = random.sample(examples, num_sample)
            random.shuffle(examples)
            l0, l1= [], []
            labels = list(set([e.label for e in examples]))
            for example in examples:
                if example.label==labels[0] and len(l0)<num_sample:
                    l0.append(example)
                elif example.label==labels[1] and len(l1)<num_sample:
                    l1.append(example)
                elif len(l0)==num_sample and len(l1)==num_sample:
                    break
            examples = l0+l1
            # print the data distribution
            labels= []
            for example in examples:
                labels.append(example.label)
            labels_count=Counter(labels)
            print(labels_count)
        return examples

    def get_translate_examples(self, data_dir, language='en', split='train'):
        """See base class."""
        languages = language.split(',')
        examples = []
        for language in languages:
            if split == 'train':
                file_path = os.path.join(data_dir, "translated/en-{}-translated.tsv".format(language))
            else:
                file_path = os.path.join(data_dir, "translated/test-{}-en-translated.tsv".format(language))
            logger.info("reading from " + file_path)
            lines = self._read_tsv(file_path)
            for (i, line) in enumerate(lines):
                guid = "%s-%s-%s" % (split, language, i)
                text_a = line[0]
                text_b = line[1]
                label = str(line[2].strip())
                assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, language=language))
        return examples

    def get_train_examples(self, data_dir, language='en', num_sample=-1):
        """See base class."""
        return self.get_examples(data_dir, language, split='train', num_sample=num_sample)

    def get_translate_train_examples(self, data_dir, language='en'):
        """See base class."""
        return self.get_translate_examples(data_dir, language, split='train')

    def get_translate_test_examples(self, data_dir, language='en'):
        """See base class."""
        return self.get_translate_examples(data_dir, language, split='test')

    def get_test_examples(self, data_dir, language='en', num_sample=-1):
        """See base class."""
        return self.get_examples(data_dir, language, split='test', num_sample=num_sample)

    def get_dev_examples(self, data_dir, language='en', num_sample=-1):
        """See base class."""
        return self.get_examples(data_dir, language, split='dev', num_sample=num_sample)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]


pawsx_processors = {
    "pawsx": PawsxProcessor,
}

pawsx_output_modes = {
    "pawsx": "classification",
}

pawsx_tasks_num_labels = {
    "pawsx": 2,
}
