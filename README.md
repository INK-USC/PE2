## Prompt Engineering a Prompt Engineer

Code for paper "Prompt Engineering a Prompt Engineer" (https://arxiv.org/abs/2311.05661)

### Environment

```bash
conda create --name pe2 python=3.11
conda activate pe2
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Data

* We uploaded all preprocessed data files in `/data`. 
* We included scripts to download and preprocess the data in `/data/preprocess`.

