# **Modeling Transitions of Focal Entities for Conversational Knowledge Base Question Answering**

This is the code for the paper: [Modeling Transitions of Focal Entities for Conversational Knowledge Base Question Answering]()\
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
* Python 2.7.17
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
python code/PreProcessCSQA.py
```
It takes some time to do the pre-processing. Alternatively, the processed data can be downloaded from [link]()

## Get the 

## Download the Pre-trained Model
You can download our pre-trained models from the [link]() and put the folders under the path trained_model/.

## Test the Pre-trained Model
To test our pre-trained model, simply run shell files:
```
./[CONVEX|ConvCSQA]_Runner.sh
```
