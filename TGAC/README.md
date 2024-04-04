## Temporal Graph Representation Learning with Adaptive Augmentation Contrastive

PyTorch Implementation of paper:

> **Temporal Graph Representation Learning with Adaptive Augmentation Contrastive (ECML-PKDD2023)**
> 
> Hongjiang Chen, Pengfei Jiao, Huijun Tang, Huaming Wu.

Paper link: [ArXiv](https://arxiv.org/pdf/2311.03897.pdf)
## Running the experiments
### Requirements

Dependencies (with python >= 3.7):

```{bash}
pandas==1.1.0
torch==1.6.0
scikit_learn==0.23.1
```

### Dataset and Preprocessing

#### Download the public data
Download the sample datasets (eg. wikipedia and reddit) from
[here](http://snap.stanford.edu/jodie/) and store their csv files in a folder named
```data/```.

#### Preprocess the data
We use the dense `npy` format to save the features in binary format. If edge features or nodes 
features are absent, they will be replaced by a vector of zeros. 
```{bash}
python utils/preprocess_data.py --data wikipedia --bipartite
python utils/preprocess_data.py --data reddit --bipartite
```

## Model Training
```shell
# pagerank
python train_link_prediction.py --drop_scheme pr

```


## Acknowledgment
This repo is built upon the following work:
```
TGN: Temporal Graph Networks  
https://github.com/twitter-research/tgn
```
Many thanks to the authors and developers!

## Cite Us
<pre>
@inproceedings{chen2023temporal,
  title={Temporal Graph Representation Learning with Adaptive Augmentation Contrastive},
  author={Chen, Hongjiang and Jiao, Pengfei and Tang, Huijun and Wu, Huaming},
  booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  pages={683--699},
  year={2023},
  organization={Springer}
}
</pre>