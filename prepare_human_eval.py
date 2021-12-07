"""
Take a model predictions file and create 3 files for human eval.

We will use 99 examples and have 20% overlap between the 3 files' examples.
"""

import argparse
import pathlib
import json
import random


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
            }
            corrections.append(correction)

    return corrections


def run_human_eval(predictions_file: pathlib.Path):

    predictions = prepare_human_eval(predictions_file)
    set_1 = random.sample(predictions, 33)

    # Choose int(.2 * 33) examples from set_1 to put in set_2
    set_2 = random.sample(set_1, int(.2 * 33))

    while len(set_2) < 33:
        new_example = random.sample(predictions, 1)
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
        example = random.sample(predictions, 1)
        if example not in set_1 and example not in set_2 and example not in set_3:
            set_3.append(example)

    assert len(set_3) == 33

    # Save the files
    predictions_file_dir = predictions_file.with_suffix("")
    predictions_file_dir.mkdir(exist_ok=True)

    (predictions_file_dir / "alex.json").write_text(json.dumps({"predictions": set_1}, indent=2))
    (predictions_file_dir / "jierui.json").write_text(json.dumps({"predictions": set_1}, indent=2))
    (predictions_file_dir / "ayush.json").write_text(json.dumps({"predictions": set_1}, indent=2))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_file", type=pathlib.Path)
    args = parser.parse_args()

    run_human_eval(args.predictions_file)
