import math
from argparse import ArgumentParser
from logging import warning

import pandas
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


def prepare_final_xls(final_results, results_folder):
    my_df = pandas.DataFrame(final_results).T
    my_df = my_df.fillna(0.0)
    my_df["theory_part_1_comment"] = ""
    my_df["theory_part_2_comment"] = ""
    my_df.insert(0, "Student name", my_df.name)
    del my_df["name"]
    time_columns = [c for c in my_df.columns if "time" in c]
    acc_columns = [c for c in my_df.columns if "acc" in c]
    my_df[time_columns] = my_df[time_columns].applymap(lambda x: f'{math.ceil(x)}')
    my_df[acc_columns] = my_df[acc_columns].applymap(lambda x: f'{x:6.3f}')
    my_df.to_excel(results_folder / "results.xls")


SOLUTION_REGEX = r"(?P<name>^[^_]*)_(?P<id>[0-9]{4,})_.*_(?P<type>[\w-]*$)"

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--hw_folder", help="Path to solutions folder. "
                                            "Should have 'submissions' subfolder with assignment solution files"
                                            "solution file format should match SOLUTION_REGEX",
                        type=Path)
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
    try:
        current_results = json.load((results_folder / "results.json").open("r"))
    except Exception as e:
        warning(str(e))
        current_results = {}

    for file_path in sorted(submissions_folder.glob("*/*.py")):
        file_name = str(file_path.stem)
        if "__init__" in file_name:
            continue
        print(file_name, SOLUTION_REGEX)
        matched = re.search(SOLUTION_REGEX, file_name).groups()
        student_name, student_id, submission_type = matched
        print(file_name, student_name, student_id, submission_type, sep=', ')
        process_script(file_name=file_name, id_=student_id, type_=submission_type, name=student_name,
                       known_results=current_results,
                       package=".".join(str(file_path).split("/")[:-1]),
                       path_to_results=results_folder / "results.json")
    prepare_final_xls(current_results, results_folder)
