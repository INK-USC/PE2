## Prompt Engineering a Prompt Engineer

Code for paper "Prompt Engineering a Prompt Engineer" (https://arxiv.org/abs/2311.05661), to appear at ACL 2024 (Findings).

### Environment

```bash
conda create --name pe2 python=3.11
conda activate pe2
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Data

* We uploaded all preprocessed data files in `data`. 
* We included scripts to download and preprocess the data in `data/preprocess`.

### Run

See `example_scripts/bbh_boolean_expressions.sh` for running prompt optimization for the task of boolean expressions (from BIG-bench Hard), with gpt-3.5-turbo-instruct as the task model, and gpt-4-turbo as the optimization model.


