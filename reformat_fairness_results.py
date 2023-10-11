import os
import json
import argparse
import numpy as np
from collections import defaultdict

def read_json_file(file_path):
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Unable to decode JSON in '{file_path}'. Check if it's valid JSON.")
        return None

def compute_mean_and_std(data):
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)  # Use ddof=1 for sample standard deviation
    # print(std_dev)

    return f'{mean:.3f} $\\pm$ {std_dev:.3f}'

def run(file_path):
    eval_results = read_json_file(file_path)

    eval_summary = {}
    if eval_results:
            for protected_attribute in eval_results.keys():
                eval_summary[protected_attribute] = defaultdict(dict)

                value1 = list(eval_results[protected_attribute].keys())[0]
                value2 = list(eval_results[protected_attribute].keys())[1]

                for value in eval_results[protected_attribute].keys():

                    for metric in ['ACC', 'F1', 'SP', 'EoO']:
                        eval_summary[protected_attribute][value][metric] = compute_mean_and_std(eval_results[protected_attribute][value][metric])

                for metric in ['ACC', 'F1', 'SP', 'EoO']:
                     data1 = eval_results[protected_attribute][value1][metric]
                     data2 = eval_results[protected_attribute][value2][metric]
                     data = []
                     for i in range(len(data1)):
                        data.append(data1[i] - data2[i])
                     eval_summary[protected_attribute]['diff'][metric] = compute_mean_and_std(data)

                eval_summary[protected_attribute]['formatted'] = {}
                for value in [value1, value2, 'diff']:
                    eval_summary[protected_attribute]['formatted'][value] = ' & '.join(eval_summary[protected_attribute][value].values())


            with open(os.path.join(os.path.dirname(file_path), 'fairness_summary_formatted.json'), 'w+') as fout:
                json.dump(eval_summary, fout, indent=3)

def find_files_with_name(start_dir, target_name, results=None):
    if results is None:
        results = []

    for root, _, files in os.walk(start_dir):
        for file in files:
            if file == target_name:
                results.append(os.path.join(root, file))

    return results

if __name__ == "__main__":
    found_files = find_files_with_name("../outputs", "fairness_results.json")
    print(len(found_files))

    for file in found_files:
        run(file)
