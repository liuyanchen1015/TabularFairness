import os
import json
import time
import openai
import argparse
from tqdm import tqdm
from sklearn.utils import resample
from datasets import(
    Dataset,
    DatasetDict,
    load_dataset,
    concatenate_datasets
)

os.environ["OPENAI_API_KEY"] = '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'
openai.api_key = os.getenv("OPENAI_API_KEY")

class EvaluationArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Evaluation Argument Parser')
        self.parser.add_argument('--benchmark_path', type=str, default='../data/benchmark/performance/')
        self.parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo')
        self.parser.add_argument('--task_name', type=str, default='Adult/prototypes-synthetic-performance-0')
        self.parser.add_argument('--output_dir', type=str, default='../outputs')
        self.parser.add_argument('--resampling_mode', type=str, default=None)
        self.parser.add_argument('--resampling_by', type=str, default=None)
        self.parser.add_argument('--job_id', type=str, default=None)

    def str2bool(self, v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def parse_args(self):
        return self.parser.parse_args()

def count(dataset, attribute):
    counts = {}
    for example in dataset:
        x = example[attribute]
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    return counts

#! apply prompt to examples
def process_flan_lift(examples):
    """Collates dataset to prompts for FLAN + LIFT.

    Intended for use with dataset.map

    :param: example: the current example.
    """

    final = [
        (f"{examples['lift_header'][i]} "
        f"{examples['class_text'][i]}\n\n"
        f"{examples['serialization'][i]}Answer:")
        for i in range(len(examples['class_text']))]
    examples["text"] = [clean_up(f) for f in final]
    return examples

# Cleans up quirks in text
def clean_up(text: str):
    """Remove formatting quirks"""
    return text.replace("with", "with ").replace("  ", " ").replace("\n ", "\n").replace("::", ":").replace(
        "The patient answered yes to the following questions "
        "about their symptoms.", "Here are the patient's "
                                "responses to questions about "
                                "their symptoms.").replace("\n\n\n", "\n\n")
#!

if __name__ == '__main__':
    arg_parser = EvaluationArgumentParser()
    args = arg_parser.parse_args()
    print(args)

    # terminate if 'predictions.json' already exists
    if os.path.exists(os.path.join(args.output_dir, "predictions.json")):
        print("results already exist!")
        exit(0)

    # load the datasets
    raw_datasets = load_dataset("json", data_files={"train": os.path.join(args.benchmark_path, args.task_name, 'train.json'),
                                                    "test": os.path.join(args.benchmark_path, args.task_name, 'test.json'),})

    label_list = raw_datasets['train'].unique('label')
    print(label_list)

    # split train and eval dataset
    tmp = raw_datasets['train'].train_test_split(test_size=0.1)
    raw_datasets['train'] = tmp['train']
    raw_datasets['eval'] = tmp['test']
    raw_datasets['test'] = raw_datasets['test']

    #? resampling the train dataset
    if args.resampling_mode == 'None':
        args.resampling_mode = None
    if args.resampling_mode:
        train_dataset = raw_datasets['train']

        def extract_attribute(example, attribute):
            if attribute in ['sex', 'Sex', 'SEX']:
                if ('Female' in example['serialization']) or ('female' in example['serialization']):
                    return 'Female'
                else:
                    return 'Male'
            elif attribute in ['race', 'RAC1P']:
                if 'African-American' in example['serialization']:
                    return 'African-American'
                else:
                    return 'Not African-American'
            else:
                raise ValueError('Unsupported protected attribute!')

        # Add a new column resampling_by based on the 'serialization' field
        train_dataset = train_dataset.map(lambda ex: {args.resampling_by: extract_attribute(ex, args.resampling_by)})
        counts = count(train_dataset, args.resampling_by)
        print(f'original counts: {counts}')
        minority_class = min(counts, key=counts.get)
        majority_class = max(counts, key=counts.get)

        # Oversampling
        if args.resampling_mode == 'oversampling':
            minority_samples = train_dataset.filter(lambda example: example[args.resampling_by] == minority_class)
            minority_samples_oversampled = Dataset.from_dict(
                resample(minority_samples, replace=True, n_samples=counts[majority_class] - counts[minority_class], random_state=42))
            train_dataset = concatenate_datasets([train_dataset, minority_samples_oversampled])

        # Undersampling
        elif args.resampling_mode == 'undersampling':
            majority_samples = train_dataset.filter(lambda example: example[args.resampling_by] == majority_class)
            majority_samples_undersampled = Dataset.from_dict(
                resample(majority_samples, replace=False, n_samples=counts[minority_class], random_state=42))
            train_dataset = concatenate_datasets([train_dataset.filter(lambda example: example[args.resampling_by] == minority_class),
                                                majority_samples_undersampled])

        elif args.resampling_mode:
            raise ValueError('only oversampling or undersampling supported.')

        # Shuffle the dataset
        train_dataset = train_dataset.shuffle(seed=42)

        # Verify the balanced dataset
        counts_after = count(train_dataset, args.resampling_by)
        print(f'counts after resampling: {counts_after}')

        raw_datasets['train'] = train_dataset.remove_columns(args.resampling_by)
    #?

    # preprocess the datasets
    raw_datasets = raw_datasets.map(process_flan_lift, batched=True) # apply template

    keep_column_names = ['text', 'label']
    column_names = list(raw_datasets.values())[0].column_names if isinstance(raw_datasets, DatasetDict) else raw_datasets.column_names
    remove_column_names = [name for name in column_names if name not in keep_column_names]
    raw_datasets = raw_datasets.remove_columns(remove_column_names)

    prefix = args.task_name.split('/')[0][:6] + (f'_{args.resampling_mode[0]}_{args.resampling_by}' if args.resampling_mode else '')
    train_file = os.path.join(args.output_dir, f"{prefix}_train.jsonl")
    val_file = os.path.join(args.output_dir, f"{prefix}_val.jsonl")
    test_file = os.path.join(args.output_dir, f"{prefix}_test.jsonl")
    raw_datasets['train'].to_json(train_file, orient='records', lines=True, index=False)
    raw_datasets['test'].to_json(val_file, orient='records', lines=True, index=False)
    raw_datasets['test'].to_json(test_file, orient='records', lines=True, index=False)

    def prepocess_example(ex):
        return {"messages": [{"role": "system", "content": ""}, {"role": "user", "content": ex['text']}, {"role": "assistant", "content": ex['label']}]}

    def process_jsonl(filename):
        with open(filename, 'r') as file:
            lines = file.readlines()

        processed_lines = []
        for line in lines:
            data = json.loads(line)
            processed_data = prepocess_example(data)
            processed_lines.append(json.dumps(processed_data))

        with open(filename, 'w') as file:
            for line in processed_lines:
                file.write(line + "\n")

    process_jsonl(train_file)
    process_jsonl(val_file)
    process_jsonl(test_file)

    #! Training
    training_file = openai.File.create(
        file=open(train_file, "rb"),
        purpose='fine-tune'
    )['id']

    validation_file = openai.File.create(
        file=open(val_file, "rb"),
        purpose='fine-tune'
    )['id']


    if not args.job_id:
        job_id = None
        while not job_id:
            try:
                job_id = openai.FineTuningJob.create(training_file=training_file, validation_file=validation_file, model="gpt-3.5-turbo", suffix=prefix)['id']
            except openai.error.RateLimitError as e:
                print(e)
                time.sleep(300)

        print(f"Job ID: {job_id}")

        #! Wait for finetuning
        while(openai.FineTuningJob.retrieve(job_id)['finished_at'] == None):
            time.sleep(60)
    else:
        job_id = args.job_id

    #! Test
    print("Testing...")
    finetuned_model = openai.FineTuningJob.retrieve(job_id)['fine_tuned_model']
    texts, references = raw_datasets['test']['text'], raw_datasets['test']['label']

    results = []
    for text in tqdm(texts):
        completion = openai.ChatCompletion.create(
            model=finetuned_model,
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": text}
            ]
        )
        results.append(completion.choices[0].message['content'])

    json.dump([results], open(os.path.join(args.output_dir, 'predictions.json'), 'w+'), indent=4)