"""
Read in the human evalutation and calculate metrics
"""

import argparse
import pathlib
import json



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process human evaluations.")
    parser.add_argument("--alex_results", type=pathlib.Path, required=True)
    parser.add_argument("--jierui_results", type=pathlib.Path, required=True)
    parser.add_argument("--ayush_results", type=pathlib.Path, required=True)
    args = parser.parse_args()
    