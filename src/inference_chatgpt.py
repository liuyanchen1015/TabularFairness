import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from Tablet import evaluate
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

os.environ["OPENAI_API_KEY"] = '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'

class EvaluationArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Evaluation Argument Parser')
        self.parser.add_argument('--benchmark_path', type=str, default='../data/benchmark/performance/')
        self.parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo')
        self.parser.add_argument('--task_name', type=str, default='Adult/prototypes-synthetic-performance-0')
        self.parser.add_argument('--output_dir', type=str, default='../outputs')
        self.parser.add_argument('--k_shot', type=int, default=20)
        self.parser.add_argument('--run_num', type=int, default=5)
        self.parser.add_argument('--encoding_format', type=str, default='flan-LIFT')
        self.parser.add_argument('--if_flip_in_context_example_label', type=self.str2bool, default=False)
        self.parser.add_argument('--if_flip_instruction', type=self.str2bool, default=False)
        self.parser.add_argument('--temperature', type=int, default=0)

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

if __name__ == '__main__':
    arg_parser = EvaluationArgumentParser()
    args = arg_parser.parse_args()
    print(args)

    benchmark_path = args.benchmark_path
    output_dir = args.output_dir
    task_name = args.task_name
    model_name = args.model_name
    k_shot = args.k_shot
    run_num = args.run_num
    encoding_format = args.encoding_format
    if_flip_in_context_example_label = args.if_flip_in_context_example_label
    if_flip_instruction = args.if_flip_instruction

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(benchmark_path, task_name, 'test.csv'))
    attributes = df.columns
    print(attributes)

    evaluator = evaluate.Evaluator(benchmark_path=benchmark_path,
                                   tasks_to_run=[task_name],
                                   encoding_format=encoding_format,
                                   k_shot=k_shot,
                                   if_flip_in_context_example_label=if_flip_in_context_example_label,
                                   if_flip_instruction=if_flip_instruction)

    datasets = evaluator.get_test_hf_datasets()[0]
    texts, references = datasets['text'], datasets['label']
    # texts, references = datasets['text'][:10], datasets['label'][:10]

    prompt = PromptTemplate(
    input_variables=["text"],
    template="{text}"
    )
    temperature = args.temperature
    print(f"Temperature: {temperature}")

    llm = OpenAI(model_name=model_name, temperature=temperature)
    chain = LLMChain(llm=llm, prompt=prompt)

    results_list = []
    for i in range(run_num):
        results = []
        for text in tqdm(texts):
            ans = chain.run(text)
            results.append(ans)
        results_list.append(results)

    json.dump(results_list, open(os.path.join(output_dir, 'predictions.json'), 'w+'), indent=3)
