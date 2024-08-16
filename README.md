# KCHML: Knowledge-aware Contrastive Heterogeneous Molecular Graph Learning for Property Prediction
This is the code of paper **Knowledge-aware Contrastive Heterogeneous Molecular
Graph Learning for Property Prediction** 

## Dependencies
- dgl==2.3.0+cu118
- k_means_constrained==0.7.3
- networkx==3.0
- numpy==1.24.3
- pandas==2.0.3
- rdkit==2024.3.5
- scikit_learn==1.3.0
- torch==2.1.2+cu118
- tqdm==4.66.5
- pyg==2.5.2
### Preparing
### Pre-training data
We collect 250K unlabeled molecules sampled from the ZINC 250 datasets and align them with drkg to pre-train KCHML. 

The raw pre-training data can be found in https://github.com/gnn4dr/DRKG and [zinc15_drugbank_canonical.csv](dataset%2Fpretrain%2Fzinc15_drugbank_canonical.csv).

In order to achieve the greatest possible molecular similarity within a small batch and assign a certain number of drug molecules to it, [cluster.py](data%2Fcluster.py) is used to generate batches

A reference batch is in dataset/pretrain/[zinc15_drugbank_canonical.csv](dataset%2Fpretrain%2Fzinc15_drugbank_canonical.csv)

### Knowledge feature initialization
The element knowledge graph is stored in the form of triples in [ekg.csv](dataset%2Fpretrain%2Fekg.csv).

Its encoding method is provided by https://github.com/MIRALab-USTC/KGE-HAKE/.

https://github.com/gnn4dr/DRKG provides the drug knowledge graph and its encoding.

## Running

### Pre-train Models

The pre-training process is described in detail in the file [pretrain.py](pretrain.py) . 

The key innovations are [HMGEncoder.py](model%2FHMGEncoder.py) and [ContrastiveLoss.py](layers%2FContrastiveLoss.py).
[HGTLayer.py](layers%2FHGTLayer.py) implements the message passing process between nodes and edges.

The file [cluster.py](data%2Fcluster.py) provides the detailed implementation process of HMG and its two enhancement methods.

### Fine-tuning
[finetune.py](finetune.py) provides methods for fine-tuning on a variety of downstream tasks.