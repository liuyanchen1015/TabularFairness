import os
import json
import argparse
import numpy as np
import pandas as pd
from statistics import mean
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

labels_dict = {
    "Adult" : ['less than or equal to 50K', 'greater than 50K'],
    "COMPAS": ['Reoffended', 'Did Not Reoffend'],
    'GermanCredit': ['bad', 'good'],
    'ACSIncome': ['False', 'True']
}

protected_attributes_dict = {
    "Adult" : ['sex'],
    "COMPAS": ['sex', 'race'],
    'GermanCredit': ['Sex'],
    'ACSIncome': ['SEX', 'RAC1P']
}

class EvaluationArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Evaluation Argument Parser')
        self.parser.add_argument('--benchmark_path', type=str, default='../data/benchmark/performance/')
        self.parser.add_argument('--task_name', type=str, default='Adult/prototypes-synthetic-performance-0')
        self.parser.add_argument('--results_dir', type=str, default='../outputs/Adult/prototypes-synthetic-performance-0/google-flan-t5-xl/10/flan')

    def parse_args(self):
        return self.parser.parse_args()

def compute_statistical_parity(confusion_matrix):
    # Extract the values from the confusion matrix
    tn, fp = confusion_matrix[0][0], confusion_matrix[0][1]
    fn, tp = confusion_matrix[1][0], confusion_matrix[1][1]

    # Calculate statistical parity
    return (tp + fp) / (tn + fp + fn + tp)

def compute_equality_of_opportunity(confusion_matrix):
    # Extract the values from the confusion matrix
    tn, fp = confusion_matrix[0][0], confusion_matrix[0][1]
    fn, tp = confusion_matrix[1][0], confusion_matrix[1][1]

    # Calculate equality of opportunity
    return tp / (tp + fn)

def main():
    arg_parser = EvaluationArgumentParser()
    args = arg_parser.parse_args()
    print('Benchmark Path:', args.benchmark_path)
    print('Task Name:', args.task_name)
    print('Results File Directory:', args.results_dir)

    benchmark_path = args.benchmark_path
    task_name = args.task_name
    results_dir = args.results_dir
    protected_attributes = protected_attributes_dict[task_name.split("/")[0]]
    print('Protected Attributes:', protected_attributes)

    df = pd.read_csv(os.path.join(benchmark_path, task_name, 'test.csv'))
    print(df['y_temp'].value_counts())
    attributes = df.columns
    print(attributes)
    references = df['y_temp'].tolist()

    # # for finetuning t5, merge the results from several runs
    # if 'finetuning' in results_dir and 't5' in results_dir:
    #! for finetuning, merge the results from several runs
    if 'finetuning' in results_dir:
        results_list = []
        for seed in os.listdir(results_dir):
            subdir_path = os.path.join(results_dir, seed)
            if os.path.isdir(subdir_path):
                if 't5' in results_dir:
                    predictions_file_path = os.path.join(subdir_path, 'results', 'predictions.json')
                else:
                    predictions_file_path = os.path.join(subdir_path, 'predictions.json')
                if os.path.exists(predictions_file_path):
                    with open(predictions_file_path, 'r') as file:
                        data = json.load(file)
                        results_list.extend(data)
        print(len(results_list))

    else:
        results_list = json.load(open(os.path.join(results_dir, 'predictions.json') , 'r'))

    # assert len(results_list) == 5 #! average over 5 runs

    # compute overall f1 and acc
    results_summary = defaultdict(list)
    for results in results_list:
        predictions = results['predictions'] if isinstance(results, dict) else results

        print(len(predictions), len(references))
        # references = [str(item) for item in references]

        f1 = f1_score(references,
                      predictions,
                      average="macro",
                      labels=np.unique(references))
        acc = accuracy_score(references, predictions)

        results_summary['ACC'].append(acc)
        results_summary['F1'].append(f1)

    with open(os.path.join(results_dir, 'overall_summary.json'), 'w+') as fout:
        json.dump(results_summary, fout, indent=3)

    # compute f1 and acc for each subgroup
    eval_results = {}
    eval_summary = {}
    for protected_attribute in protected_attributes:
        eval_results[protected_attribute] = {}
        eval_summary[protected_attribute] = defaultdict(dict)

        for value in sorted(df[protected_attribute].unique()):
            print(f"Number of {value}/{protected_attribute}:{len(df.loc[df[protected_attribute] == value])}")

            eval_results[protected_attribute][value] = defaultdict(list)

            for i, results in enumerate(results_list):
                predictions = results['predictions'] if isinstance(results, dict) else results

                indices = df[protected_attribute] == value
                references_filtered = [x for x, mask in zip(references, indices) if mask]
                predictions_filtered = [x for x, mask in zip(predictions, indices) if mask]

                f1 = f1_score(references_filtered,
                            predictions_filtered,
                            average="macro",
                            labels=np.unique(references_filtered))
                acc = accuracy_score(references_filtered, predictions_filtered)

                eval_results[protected_attribute][value]['ACC'].append(acc)
                eval_results[protected_attribute][value]['F1'].append(f1)

                cm = confusion_matrix(references_filtered, predictions_filtered, labels=labels_dict[task_name.split("/")[0]])
                eval_results[protected_attribute][value]['CM'].append(cm.tolist())

                eval_results[protected_attribute][value]['SP'].append(compute_statistical_parity(cm))
                eval_results[protected_attribute][value]['EoO'].append(compute_equality_of_opportunity(cm))


    for protected_attribute in protected_attributes:
        for value in df[protected_attribute].unique():
            mean_f1 = mean(eval_results[protected_attribute][value]['F1'])
            mean_acc = mean(eval_results[protected_attribute][value]['ACC'])
            mean_sp = mean(eval_results[protected_attribute][value]['SP'])
            mean_eoo = mean(eval_results[protected_attribute][value]['EoO'])

            eval_summary[protected_attribute][value]['ACC'] = mean_acc
            eval_summary[protected_attribute][value]['F1'] = mean_f1
            eval_summary[protected_attribute][value]['SP'] = mean_sp
            eval_summary[protected_attribute][value]['EoO'] = mean_eoo

    with open(os.path.join(results_dir, 'fairness_results.json'), 'w+') as fout:
        json.dump(eval_results, fout, indent=3)
    with open(os.path.join(results_dir, 'fairness_summary.json'), 'w+') as fout:
        json.dump(eval_summary, fout, indent=3)



if __name__ == '__main__':
    main()