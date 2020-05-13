import Levenshtein as le
import numpy as np

def compute_metrics(predicted_strings, target_strings, metrics):
    metric_values = {}
    for m in metrics:
        if m == 'acc@1':
            metric_values[m] = sum(predicted_strings == target_strings) / len(target_strings)
        elif m =='mean_ld@1':
            metric_values[m] =\
                np.mean(list(map(lambda e: le.distance(*e), zip(predicted_strings, target_strings))))
        else: 
            raise ValueError(f'Unknown metric: {m}')
    return metric_values
