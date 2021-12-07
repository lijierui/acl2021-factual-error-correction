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

"""
A masker that keeps a word if either of its neighbors is also in the evidence.


PYTHONPATH=src python3 src/error_correction/masking/random_masker.py \
    --in_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_train_genre_50_2.jsonl \
    --out_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/random_30_train.jsonl \
    --sample_prob 0.21

PYTHONPATH=src python3 src/error_correction/masking/random_masker.py \
    --in_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_dev_genre_50_2.jsonl \
    --out_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/random_30_dev.jsonl \
    --sample_prob 0.21

PYTHONPATH=src python3 src/error_correction/masking/random_masker.py \
    --in_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_test_genre_50_2.jsonl \
    --out_file /home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/random_30_test.jsonl \
    --sample_prob 0.21

"""


import argparse
import json
import math
import random

from tqdm import tqdm
from transformers import AutoTokenizer
from error_correction.masking.json_encoder import NpEncoder
from error_correction.modelling.reader.fever_reader import FEVERReader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file")
    parser.add_argument("--out_file")
    parser.add_argument("--sample_prob", type=float, default=0.15)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained("bert-base-cased")

    reader = FEVERReader(True)
    generator = reader.read(args.in_file)

    with open(args.in_file) as f, open(args.out_file, "w+") as of:
        for line in f:
            instance = json.loads(line)

            tokens = tok.tokenize(instance["mutated"])
            master_explaination = []
            for idx in range(len(tokens)):
                if random.random() < args.sample_prob:
                    master_explaination.append(idx)

            instance["master_explanation"] = master_explaination
            instance["original_claim"] = " ".join(
                [word.lower() for word in instance["mutated"].split(" ")]
            )

            of.write(json.dumps(instance) + "\n")

    # with open(args.out_file, "w+") as log:
    #     for instance in tqdm(
    #         filter(lambda i: i["label"] != "NOT ENOUGH INFO", generator)
    #     ):
    #         split_claim = tokenizer.convert_tokens_to_string(
    #             tokenizer.tokenize(instance["mutated"])
    #         ).split()
    #         prem_idx = random.sample(
    #             range(len(split_claim)),
    #             k=max(
    #                 1,
    #                 min(
    #                     len(split_claim), math.ceil(len(split_claim) * args.sample_prob)
    #                 ),
    #             ),
    #         )

    #         instance.update(
    #             {"original_claim": " ".join(split_claim), "input_mask": prem_idx}
    #         )
    #         log.write(
    #             json.dumps(
    #                 {
    #                     "instance": instance,
    #                 },
    #                 cls=NpEncoder,
    #             )
    #             + "\n"
    #         )
