import pathlib
import json


mask_files = [
    pathlib.Path("/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_gold_dev_genre_50_2.jsonl"),
    pathlib.Path("/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/blackbox_ir_dev_genre_50_2.jsonl"),
    pathlib.Path("/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_gold_dev_genre_50_2.jsonl"),
    pathlib.Path("/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/whitebox_ir_dev_genre_50_2.jsonl"),
    pathlib.Path("/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/trigram_train.jsonl"),
    pathlib.Path("/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/trigram_dev.jsonl"),
    pathlib.Path("/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/trigram_test.jsonl"),
    pathlib.Path("/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/heuristic_gold_dev_genre_50_2.jsonl"),
    pathlib.Path("/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/heuristic_ir_dev_genre_50_2.jsonl"),
    pathlib.Path("/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/resources/masking/random_masked_50_dev_genre_50_2.jsonl"),
]

def avg(data):
    return sum(data) / len(data)

for mask_file in mask_files:
    mask_counts = []
    mask_percents = []
    with open(mask_file, "r") as f:
        for line in f:
            instance = json.loads(line)
            mask_counts.append(len(instance["master_explanation"]))
            mask_percents.append(len(instance["master_explanation"]) / (len(instance["original_claim"].split(" "))))


    print(f"{mask_file.name}. average_num_masks={avg(mask_counts):.4f}. average_%_masked={avg(mask_percents) * 100:.4f}")

