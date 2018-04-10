import pathlib
import os
import re
import shutil
import json


subs_folder = pathlib.Path("subs")
eval_folder = pathlib.Path(".")
path_to_results = pathlib.Path("results.json")


if not path_to_results.exists():
    with open(path_to_results, "w") as f:
        json.dump({}, f)
    known_results = {}
else:
    with open(path_to_results, "r") as f:
        known_results = json.load(f)


def process_script(id_, type_, name, known_results):
    from evaluate import main
    key = id_ + "_" + type_
    if key not in known_results:
        # current_dir = os.getcwd()
        # os.chdir(str(pathlib.Path(current_dir) / "filmdb_evaluation"))
        script_results = main()
        # os.chdir(current_dir)
        known_results[key] = script_results
        with open(path_to_results, "w") as f:
            json.dump(dict(sorted(known_results.items())), f, indent=4)
        print("#" * 100)
    elif known_results[key].get("name") is None:
        known_results[key]["name"] = name
        with open(path_to_results, "w") as f:
            json.dump(dict(sorted(known_results.items())), f, indent=4)


for file_path in sorted(subs_folder.glob("*.py")):
    file_name = str(file_path.name)
    student_id = re.search(r"_([0-9]+)_", file_name).group(1)
    student_name = file_name.split("_", 1)[0]
    subm_type = file_name.split(".", 1)[0].rsplit("_", 1)[-1]
    print(file_name, student_name, student_id, subm_type)
    path_to_copy = shutil.copy(str(file_path), eval_folder / "classifier.py")
    process_script(student_id, subm_type, student_name, known_results)
    os.remove(path_to_copy)