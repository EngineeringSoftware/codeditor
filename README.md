# Multilingual Code Co-Evolution Using Large Language Models

This repo hosts the code and data for the following FSE 2023 paper:

Title: [Multilingual Code Co-Evolution Using Large Language Models](https://arxiv.org/abs/2307.14991)

Authors: [Jiyang Zhang](https://jiyangzhang.github.io/), [Pengyu Nie](https://pengyunie.github.io/), [Junyi Jessy Li](https://jessyli.com/), [Milos Gligoric](http://users.ece.utexas.edu/~gligoric/)

```bibtex
@inproceedings{ZhangETAL23Codeditor,
  author = {Zhang, Jiyang and Nie, Pengyu and Li, Junyi Jessy and Gligoric, Milos},
  title = {Multilingual Code Co-Evolution Using Large Language Models},
  booktitle = {Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering},
  year = {2023},
}
```

## News
May 2024
The fine-tuned EditsTranslation model is released on 🤗 ! 🔥[cs2java](https://huggingface.co/EngineeringSoftware/EditsTranslation-cs2java) and [java2cs](https://huggingface.co/EngineeringSoftware/EditsTranlation-java2cs/settings) 

## How to Use

[sec-howto]: #how-to-use

```python
from transformers import T5ForConditionalGeneration, AutoTokenizer

checkpoint = "EngineeringSoftware/EditsTranlation-java2cs"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint)

code_input = """class HelloWorld { public static void main(String[] args) { System.out.println("Hello, World!")"""

input_ids = tokenizer(code_input, return_tensors="pt").input_ids
generated_ids = model.generate(input_ids, max_length=200)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
# output: <INSERT>; } } ;<INSERT_END> class HelloWorld { public static void main(String[] args) { System.out.println("Hello, World!") ; } } ;
```




## Introduction

This repo contains the code and artifacts for reproducing the experiments in [Multilingual Code Co-Evolution Using Large Language Models](https://arxiv.org/abs/2307.14991).
In this work, we introduce Codeditor for co-evolving software implemented in multiple programming languages.

The code includes:

- scripts for processing dataset
- scripts for training and evaluating codeditor models

The artifacts include:

- Java to C# raw paired changes
- Java to C# translation dataset processed for codeditor models

## Data Downloads

[sec-downloads]: #data-downloads

All our data is hosted on UTBox via [a shared folder](https://utexas.box.com/s/iwcvwgx23g9xvowu9joa661rz74k9eea).


## Code for Processing Fine-tuning Data

[sec-process]: #code-for-processing-fine-tuning-data

We provide the sample script to process the datasets for edit-translation. Requires the raw data files at `raw_data/`.

```
cd python/
python -m deltr.collector.DataProcessor edit_translation_data_process --exp cs2java --src_lang cs --tgt_lang java

```

## Code for Training and Evaluating Models

[sec-traineval]: #code-for-training-and-evaluating-models

### Train ML models

```
cd python/
python -m deltr.coditT5.CodeT5 fit --exp_dir {MODELS_DIR}/${model_name}/${dataset} --data.dataset {dataset} --data.model ${model_name} --config  configs/coditT5.yaml

# Example: python -m deltr.coditT5.CodeT5 fit --exp_dir models/edit-translation/java2cs --data.dataset java2cs --data.model edit-translation --config  configs/coditT5.yaml
```

Results are generated to `models/${model}/${dataset}/`, where:

- `model/`: stores the trained model.

- `logs/`: stores logs during training.

### Run ML models to do inference

Requires the dataset at `data/${model}/${dataset}/`, the trained model at `models/${model}/${dataset}/model/`.

```
cd python/
python -m deltr.coditT5.CodeT5 predict --exp_dir {MODELS_DIR}/${model_name}/${dataset} --data.dataset {dataset} --data.model ${model_name} --config  configs/coditT5.yaml

```

Results are generated to `models/${model}/${dataset}/`, where:

- `output.hyp`: the predictions.
