# **Modeling Transitions of Focal Entities for Conversational Knowledge Base Question Answering**

This is the code for the paper: [Modeling Transitions of Focal Entities for Conversational Knowledge Base Question Answering](https://aclanthology.org/2021.acl-long.255/)\
Yunshi Lan, Jing Jiang \
[ACL2021](https://2021.aclweb.org/).

If you find this code useful in your research, please cite

> @inproceedings{lan:acl2021, \
> title={Modeling Transitions of Focal Entities for Conversational Knowledge Base Question Answering},\
> author={Lan, Yunshi and Jiang, Jing}, \
> booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL)}, \
> year={2021} \
> } 

## Setups
All codes were developed and tested in the following environment.

* Ubuntu 18.04
* Python 3.6.9
* Pytorch 1.4.0

Download the code:
```
git clone https://github.com/lanyunshi/ConversationalKBQA.git
pip install requirements.txt
```

## Prepare the Data
We evaluate our methods on [CONVEX](https://convex.mpi-inf.mpg.de/) and ConvCSQA, which is a subset of [CSQA](https://amritasaha1812.github.io/CSQA/).

To obtain the pre-processed data, you can simply run:
```
./DownloadData.sh
python code/PreProcessConvex.py
```
It takes some time to do the pre-processing. Alternatively, the processed data *CONVEX* can be downloaded from [link](https://drive.google.com/drive/folders/1MeQmdvHMLkoz4542N92kUSn1WygL85MJ?usp=sharing)

## Check WikiData API is working
Simply run:
```
python code/SPARQL_test.py
```
Check whether the results are:
```
{'head': {'vars': ['r', 'e1']}, 'results': {'bindings': [{'e1': {'type': 'uri', 'value': 'http://www.wikidata.org/entity/Q7985008'}, 'r': {'type': 'uri', 'value': 'http://www.wikidata.org/prop/direct/P175'}}]}}
```

## Download the Pre-trained Model and the KB cache (optional)
You can download our pre-trained models from the [link](https://drive.google.com/drive/folders/1MeQmdvHMLkoz4542N92kUSn1WygL85MJ?usp=sharing) and put the files into the path trained_model/ and files into config/

In order to save your time to validate our methods, we also recommend you to download the KB cache that collected during our exploration. You can download our cache from the [link](https://drive.google.com/drive/folders/1sV-YZanhu80REi2a9bu9Vr-jXziPawXn?usp=sharing).

## Test the Pre-trained Model
To test our pre-trained model, simply run:
```
CUDA_VISIBLE_DEVICES=1 python code/ConversationKBQA_Runner.py  \
        --train_folder  CONVEX/data/train_set \
        --dev_folder CONVEX/data/dev_set \
        --test_folder CONVEX/data/test_set \
        --vocab_file config/vocab.txt \
        --output_dir trained_model/convex \
        --config config/config_RecurrentRanker.json \
        --gpu_id 0\
        --load_model trained_model/convex/RecurrentRanker \
        --save_model RecurrentRanker+test\
        --cache_dir /PATH/TO/CACHE \
        --num_train_epochs 100 \
        --do_train 0\
        --do_eval 1\
        --do_policy_gradient 2\
        --learning_rate 3e-5 \
```

The predicted answers are saved in *trained_model* folder. To obtain the evaluation results, simply run:
```
python code/ErrorAnalysis.py \
    --data_path trained_model/convex \
    --data_file RecurrentRanker+test \
    --mode breakdown  \
```

You can obtain the breakdown results of CONVEX.
 

## Train a New Model
Before training a new model, make sure the 
If you want to train your model, for example CONVEX, you can input
```
CUDA_VISIBLE_DEVICES=1 python code/ConversationKBQA_Runner.py  \
        --train_folder  CONVEX/data/train_set \
        --dev_folder CONVEX/data/dev_set \
        --test_folder CONVEX/data/test_set \
        --vocab_file config/vocab.txt \
        --output_dir trained_model/convex \
        --config config/config_RecurrentRanker.json \
        --gpu_id 0\
        --load_model trained_model/convex/new \
        --save_model new \
        --cache_dir /PATH/TO/CACHE \
        --num_train_epochs 100 \
        --do_train 1\
        --do_eval 1\
        --do_policy_gradient 2\
        --learning_rate 3e-5 \
```
You can also try baselines (*SimpleRecurrentRanker*, *SimpleRanker*) :)
