from fire import Fire
import pandas
from score_submissions import load_current_results
from pathlib import Path
import math 

def prepare_final_xls(results_folder):
    results_folder = Path(results_folder)
    final_results = load_current_results(results_folder)
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



Fire(prepare_final_xls)
