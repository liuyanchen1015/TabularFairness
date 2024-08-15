
# Confronting LLMs with Traditional ML: Rethinking the Fairness of Large Language Models in Tabular Classifications

This repository contains the code implementation for the paper titled "[Confronting LLMs with Traditional ML: Rethinking the Fairness of Large Language Models in Tabular Classifications]([https://arxiv.org/abs/2305.13406](https://arxiv.org/abs/2310.14607))". 

## Table of Contents

0. [Abstract](#abstract)
1. [Task Creation](#taskcreation)
2. [Usage](#usage)
3. [Citation and Contact](#citation-and-contact)

## Abstract
Recent literature has suggested the potential of using large language models (LLMs) to make classifications for tabular tasks. However, LLMs have been shown to exhibit harmful social biases that reflect the stereotypes and inequalities present in society. To this end, as well as the widespread use of tabular data in many high-stake applications, it is important to explore the following questions: what sources of information do LLMs draw upon when making classifications for tabular tasks; whether and to what extent are LLM classifications for tabular data influenced by social biases and stereotypes; and what are the consequential implications for fairness?
Through a series of experiments, we delve into these questions and show that LLMs tend to inherit social biases from their training data which significantly impact their fairness in tabular classification tasks. Furthermore, our investigations show that in the context of bias mitigation, though in-context learning and finetuning have a moderate effect, the fairness metric gap between different subgroups is still larger than that in traditional machine learning models, such as Random Forest and shallow Neural Networks. This observation emphasizes that the social biases are inherent within the LLMs themselves and inherited from their pretraining corpus, not only from the downstream task datasets. Besides, we demonstrate that label-flipping of in-context examples can significantly reduce biases, further highlighting the presence of inherent bias within LLMs.

## Task Creation
The datasets used in our paper are stored in the `/data` directory. However, if you want to create your own task, you can follow the template in `create_new_task.ipynb` to transform data stored in ".csv" format into the "Dataset" format used in our work.

## Usage
### In-Context Learning
Please run `/src/inference_chatgpt.py` to get the predictions by prompting ChatGPT with in-context examples (with optional label flipping).

### Finetuning
Please run  `/src/finetune.py` to finetune ChatGPT (with optional resampling) and get the predictions.

### Evaluation
Please run  `/evaluate/fairness_evaluate.py` to evaluate the fairness metrics for the predictions obtained above.
 
## Citation and Contact

If you find this repository helpful, please cite our paper.

```
@inproceedings{liu-etal-2024-confronting,
    title = "Confronting {LLM}s with Traditional {ML}: Rethinking the Fairness of Large Language Models in Tabular Classifications",
    author = "Liu, Yanchen  and
      Gautam, Srishti  and
      Ma, Jiaqi  and
      Lakkaraju, Himabindu",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    year = "2024"
}
```

Feel free to contact Yanchen at yanchenliu@g.harvard.edu, if you have any questions about the paper.


