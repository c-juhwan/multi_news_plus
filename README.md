# Multi-News+: Cost-efficient Dataset Cleansing via LLM-based Data Annotation

## How to start

First, unzip two .tar.gz files in the `cleansing` directory. Then, run the following commands to install the required packages and execute the experiment.

```shell
$ conda create -n proj-multinewsplus python=3.8
$ conda activate proj-multinewsplus
$ pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
$ pip install -r requirements.txt
$ bash run_experiment.sh
```
