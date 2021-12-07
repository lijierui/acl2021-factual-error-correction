import pathlib
import json


mask_files = [
    pathlib.Path(
        "/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_train_genre_50_2.jsonl"
    ),
    pathlib.Path(
        "/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_dev_genre_50_2.jsonl"
    ),
    pathlib.Path(
        "/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_test_genre_50_2.jsonl"
    ),
    #
    pathlib.Path(
        "/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_gold_train_genre_50_2.jsonl"
    ),
    pathlib.Path(
        "/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_gold_dev_genre_50_2.jsonl"
    ),
    pathlib.Path(
        "/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_gold_test_genre_50_2.jsonl"
    ),
    #
    pathlib.Path(
        "/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_ir_train_genre_50_2.jsonl"
    ),
    pathlib.Path(
        "/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_ir_dev_genre_50_2.jsonl"
    ),
    pathlib.Path(
        "/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_ir_test_genre_50_2.jsonl"
    ),
    #
    pathlib.Path(
        "/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_ir_train_genre_50_2.jsonl"
    ),
    pathlib.Path(
        "/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_ir_dev_genre_50_2.jsonl"
    ),
    pathlib.Path(
        "/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_ir_test_genre_50_2.jsonl"
    ),
    #
    pathlib.Path(
        "/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/heuristic_ir_train_genre_50_2.jsonl"
    ),
    pathlib.Path(
        "/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/heuristic_ir_dev_genre_50_2.jsonl"
    ),
    pathlib.Path(
        "/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/heuristic_ir_test_genre_50_2.jsonl"
    ),
    #
    pathlib.Path(
        "/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/random_masked_50_dev_genre_50_2.jsonl"
    ),
    pathlib.Path(
        "/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/random_30_train.jsonl"
    ),
    pathlib.Path(
        "/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/random_30_dev.jsonl"
    ),
    pathlib.Path(
        "/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/random_30_test.jsonl"
    ),
]


def avg(data):
    return sum(data) / len(data)


for mask_file in mask_files:
    mask_counts = []
    mask_percents = []
    with open(mask_file, "r") as f:
        for line in f:
            instance = json.loads(line)
            if "master_explanation" in instance:
                mask = instance["master_explanation"]
            else:
                mask = instance["claim_tokens"]

            mask_counts.append(len(mask))
            if "original_claim" in instance:
                claim = instance["original_claim"]
            else:
                claim = instance["original"]
            mask_percents.append(len(mask) / (len(claim.split(" "))))

    print(
        f"{mask_file.name}. average_num_masks={avg(mask_counts):.4f}. average_%_masked={avg(mask_percents) * 100:.4f}"
    )
