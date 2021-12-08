"""
Take a model predictions file and create 3 files for human eval.

We will use 99 examples and have 20% overlap between the 3 files' examples.
"""

import argparse
import pathlib
import json
import random


_PREDICTION_FILES = [
    pathlib.Path("/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/50_prob_masker/model=t5-base,lr=5e-5,masker=random,mutation_source=false,mutation_target=false,labels=all/seed=42/final_predictions_set_test_file_blackbox_gold_test_genre_50_2_new.jsonl"),
    pathlib.Path("/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/50_prob_masker/model=t5-base,lr=5e-5,masker=random,mutation_source=false,mutation_target=false,labels=all/seed=42/final_predictions_set_test_file_whitebox_gold_test_genre_50_2.jsonl"),
    pathlib.Path("/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/50_prob_masker/model=t5-base,lr=5e-5,masker=random,mutation_source=false,mutation_target=false,labels=all/seed=42/final_predictions_set_test_file_blackbox_ir_test_genre_50_2.jsonl"),
    pathlib.Path("/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/50_prob_masker/model=t5-base,lr=5e-5,masker=random,mutation_source=false,mutation_target=false,labels=all/seed=42/final_predictions_set_test_file_whitebox_ir_test_genre_50_2.jsonl"),
    pathlib.Path("/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/50_prob_masker/model=t5-base,lr=5e-5,masker=random,mutation_source=false,mutation_target=false,labels=all/seed=42/final_predictions_set_test_file_heuristic_ir_test_genre_50_2.jsonl"),
    pathlib.Path("/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/50_prob_masker/model=t5-base,lr=5e-5,masker=random,mutation_source=false,mutation_target=false,labels=all/seed=42/final_predictions_set_test_file_attn_mask_test.jsonl"),
    pathlib.Path("/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/50_prob_masker/model=t5-base,lr=5e-5,masker=random,mutation_source=false,mutation_target=false,labels=all/seed=42/final_predictions_set_test_file_attn_mask_test_025.jsonl"),
    pathlib.Path("/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/50_prob_masker/model=t5-base,lr=5e-5,masker=random,mutation_source=false,mutation_target=false,labels=all/seed=42/final_predictions_set_test_file_grad_mask_test.jsonl"),
    pathlib.Path("/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/50_prob_masker/model=t5-base,lr=5e-5,masker=random,mutation_source=false,mutation_target=false,labels=all/seed=42/final_predictions_set_test_file_grad_mask_test_025.jsonl"),
    pathlib.Path("/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/50_prob_masker/model=t5-base,lr=5e-5,masker=random,mutation_source=false,mutation_target=false,labels=all/seed=42/final_predictions_set_test_file_comb_mask_test.jsonl"),
    pathlib.Path("/home/alex/Desktop/classes/F2021/CS388/final/acl2021-factual-error-correction/output/50_prob_masker/model=t5-base,lr=5e-5,masker=random,mutation_source=false,mutation_target=false,labels=all/seed=42/final_predictions_set_test_file_comb_mask_test_025.jsonl"),
]
_SAVE_DIR = pathlib.Path("human_eval")
_SAVE_DIR.mkdir(exist_ok=True)


def prepare_human_eval(predictions_file: pathlib.Path):
    """Load in all the predictions, randomly select a few for human eval."""

    corrections = []
    with open(predictions_file, "r") as f:
        for line in f:
            instance = json.loads(line)
            correction = {
                "original": instance["metadata"]["target"],
                "input_masked": instance["metadata"]["source"],
                "prediction": instance["prediction"],
                "evidence": instance["metadata"]["evidence"],
                "metadata": instance["metadata"]
            }
            corrections.append(correction)

    return corrections


def run_human_eval(idx: int, predictions_file: pathlib.Path):

    predictions = prepare_human_eval(predictions_file)
    set_1 = random.sample(predictions, 33)

    # Choose int(.2 * 33) examples from set_1 to put in set_2
    set_2 = random.sample(set_1, int(.2 * 33))

    while len(set_2) < 33:
        new_example = random.sample(predictions, 1)[0]
        if new_example not in set_2 and set_1 not in set_2:
            set_2.append(new_example)

    assert len(set_1) == len(set_2)
    # Now make the third file. Find .2 * 33 examples which are in either set_1
    # or set_2
    set_3 = []
    while len(set_3) < int(.2 * 33):
        example = random.sample(set_1 + set_2, 1)[0]
        
        # Make sure this example is only in one of the other sets
        if (example in set_1) ^ (example in set_2):
            set_3.append(example)


    # Now add the remaining unique examples
    while len(set_3) < 33:
        example = random.sample(predictions, 1)[0]
        if example not in set_1 and example not in set_2 and example not in set_3:
            set_3.append(example)

    assert len(set_3) == 33

    # Save the files
    alex_dir = _SAVE_DIR / "alex" / str(idx)
    alex_dir.mkdir(exist_ok=True, parents=True)
    jierui_dir = _SAVE_DIR / "jierui" / str(idx)
    jierui_dir.mkdir(exist_ok=True, parents=True)
    ayush_dir = _SAVE_DIR / "ayush" / str(idx)
    ayush_dir.mkdir(exist_ok=True, parents=True)

    (alex_dir / "alex.json").write_text(json.dumps({"predictions": set_1}, indent=2))
    (jierui_dir / "jierui.json").write_text(json.dumps({"predictions": set_2}, indent=2))
    (ayush_dir / "ayush.json").write_text(json.dumps({"predictions": set_3}, indent=2))



if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--predictions_file", type=pathlib.Path)
    #args = parser.parse_args()

    # Randomly shuffle the masks
    mask_types = [file_path for file_path in _PREDICTION_FILES]
    mask_types = {idx: mask_type for idx, mask_type in enumerate(mask_types)}
    random.shuffle(mask_types)
    print(mask_types)
    for idx, mask_type in mask_types.items():
        run_human_eval(idx, mask_type)

    mask_types = {idx: str(mask_type) for idx, mask_type in mask_types.items()}
    (_SAVE_DIR / "mask_idxs.json").write_text(json.dumps({"mask_idxs": mask_types}, indent=2))
