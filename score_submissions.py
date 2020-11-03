import math
from argparse import ArgumentParser
from logging import warning

import pathlib
import os
import re
import shutil
import json
from pathlib import Path

def process_script(package, file_name, id_, type_, name, known_results, path_to_results):
    from evaluate_ctrl import main
    key = f"{id_}_{type_}"
    if key not in known_results:
        script_results = main(file_name=file_name, package=package)
        known_results[key] = script_results
        known_results[key]["name"] = name
        with path_to_results.open("w") as f:
            json.dump(dict(sorted(known_results.items())), f, indent=4)
        print("#" * 100)
    return known_results

def load_current_results(results_folder):
    current_results = {}
    for p in results_folder.glob('*json'):
        try:
            for fres in results_folder.glob("results*.json"):
                with fres.open("r") as inp:
                    current_results.update(json.load(inp))
        except Exception as e:
            warning(str(e))
    return current_results

SOLUTION_REGEX = r"(?P<name>^[^_]*)_(?:LATE_)?(?P<id>[0-9]{4,})_.*_(?P<type>[\w-]*$)"

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--hw_folder", help="Path to solutions folder. "
                                            "Should have 'submissions' subfolder with assignment solution files"
                                            "solution file format should match SOLUTION_REGEX",
                        type=Path)
    parser.add_argument("--index", help="Index of the submission to evaluate. Use for parallel evaluation.",
                        default=None,
                        type=int)
    args = parser.parse_args()

    hw_folder = args.hw_folder

    if not hw_folder.exists():
        raise FileNotFoundError("Solutions directory does not exist")

    eval_folder = hw_folder / "evaluation"
    results_folder = hw_folder / "results"
    submissions_folder = hw_folder / "submissions"

    for folder in [eval_folder, results_folder]:
        if not folder.exists():
            folder.mkdir()

    current_results = load_current_results(results_folder)
    for index, file_path in enumerate(sorted(submissions_folder.glob("*/*.py"))):
        file_name = str(file_path.stem)
        if "__init__" in file_name:
            continue
        print(file_name, SOLUTION_REGEX)
        matched = re.search(SOLUTION_REGEX, file_name)
        if matched is None:
            print(f"Couldn't match filename {file_name} with regex {SOLUTION_REGEX}")
            continue
        matched = matched.groups()

        student_name, student_id, submission_type = matched
        print(index, file_name, student_name, student_id, submission_type, sep=', ', flush=True)
        if args.index is None or args.index == index:
            process_script(file_name=file_name, id_=student_id, type_=submission_type, name=student_name,
                           known_results=current_results,
                           package=".".join(str(file_path).split("/")[:-1]),
                           path_to_results=results_folder / f"results{index}.json")
