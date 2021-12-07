#
# Copyright (c) 2019-2021 James Thorne.
#
# This file is part of factual error correction.
# See https://jamesthorne.co.uk for further info.
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
#

"""
A masker that keeps a word if either of its neighbors is also in the evidence.


python3 src/error_correction/masking/trigram_heuristic_masker.py \
    --in_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_train_genre_50_2.jsonl \
    --out_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/trigram_train.jsonl

python3 src/error_correction/masking/trigram_heuristic_masker.py \
    --in_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_dev_genre_50_2.jsonl \
    --out_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/trigram_dev.jsonl

python3 src/error_correction/masking/trigram_heuristic_masker.py \
    --in_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_test_genre_50_2.jsonl \
    --out_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/trigram_test.jsonl
"""

import json
import random

from argparse import ArgumentParser
from transformers import AutoTokenizer


def all_lower(tokens):
    return [a.lower() for a in tokens]


def all_in(tokens, token_set):
    return [t in token_set for t in tokens]


def mask_uncommon(tokens, mask, mask_token="*"):
    return [t if m else mask_token for m, t in zip(mask, tokens)]


def common_tokens(sentence1, sentence2):

    mask_idx = []
    sentence2_words = set(sentence2)
    for idx in range(len(sentence1)):
        trigram_words = [sentence1[idx]]
        if idx == 0 or idx == len(sentence1) - 1:
            continue
        if idx != 0:
            trigram_words.append(sentence1[idx - 1])
        if idx != len(sentence1) - 1:
            trigram_words.append(sentence1[idx + 1])

        trigram_set = set(trigram_words)
        if len(trigram_set.intersection(sentence2_words)) < 3:
            # If none of the trigram words are in the evidence, mask the current word
            mask_idx.append(idx)

    return mask_idx


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--in_file")
    parser.add_argument("--out_file")
    args = parser.parse_args()

    random = random.Random()
    tok = AutoTokenizer.from_pretrained("bert-base-cased")
    with open(args.in_file) as f, open(args.out_file, "w+") as of:
        for line in f:
            instance = json.loads(line)

            tokens = tok.tokenize(instance["mutated"])
            words = tok.convert_tokens_to_string(tokens)
            ev_str = " ".join([" ".join(a) for a in instance["pipeline_text"]])
            ev_tokens = tok.tokenize(ev_str)
            ev_words = tok.convert_tokens_to_string(ev_tokens)
            explanation = common_tokens(words.split(), ev_words.split())

            instance["original_claim"] = words
            instance["master_explanation"] = explanation

            of.write(json.dumps(instance) + "\n")
