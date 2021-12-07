import argparse
import pathlib
import json
import random


def is_intelligible():
    response = ""
    while response not in ["y", "n"]:
        response = input(
            "Is output sentence free of any grammatical mistakes and is comprehensible? (y/n): "
        )

    return response


def supported_by_evidence():
    response = ""
    while response not in ["y", "n"]:
        response = input("Does the evidence support the output sentence? (y/n): ")

    return response


def error_correction():
    response = ""
    while response not in ["1", "2", "3"]:
        response = input(
            """Did the output sentence correct the errors in the original sentence without including irrelevant information?"
            1) The sentence is improved with respect to the evidence.
            2) Unrelated information has been introduced.
            3) No update to the original sentence was needed but the output sentence contains changes.\n
            (1,2, or 3): """
        )

    return response


def prepare_human_eval(predictions_file: pathlib.Path):
    """Load in all the predictions, randomly select a few for human eval."""

    corrections = []
    for instance in json.loads(predictions_file.read_text())["predictions"]:
        correction = {
            "input_masked": instance["input_masked"],
            "prediction": instance["prediction"],
            "evidence": instance["evidence"],
        }
        corrections.append(correction)

    return corrections


def run_human_eval(predictions_file):

    predictions = prepare_human_eval(predictions_file)
    predictions = random.choices(predictions, k=2)

    results = []
    for prediction in predictions:
        print("\n\n", json.dumps(prediction, indent=2))
        results.append({})
        response = is_intelligible()
        results[-1]["intelligible"] = 0 if response == "n" else 1
        if response == "n":
            continue

        response = supported_by_evidence()
        results[-1]["supported_by_evidence"] = 0 if response == "n" else 1
        if response == "n":
            continue

        response = error_correction()
        results[-1]["error_correction"] = int(response)
        if response == "n":
            continue

    print(results)
    predictions_file.with_name(f"{predictions_file.stem}-results.json").write_text(json.dumps({"results": results}, indent=2))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_file", type=pathlib.Path)
    args = parser.parse_args()

    run_human_eval(args.predictions_file)
