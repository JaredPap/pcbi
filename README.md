Certainly! Here's the updated version of the `README.md` with all the necessary details for using the `cluster.py` file integrated into it.

---

# KCHML: Knowledge-aware Contrastive Heterogeneous Molecular Graph Learning for Property Prediction

## Dependencies
The following Python packages are required to run the code:

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

To install these dependencies, use:

```bash
pip install -r requirements.txt
```

## Preparing the Data

### Pre-training Data
For pre-training, we use 250K unlabeled molecules sampled from the ZINC 250 dataset. The data is aligned with the DRKG (Drug Repurposing Knowledge Graph). 

- **Raw Pre-training Data**: The data can be accessed from the following sources:
  - ZINC15: [zinc15_drugbank_canonical.csv](dataset/pretrain/zinc15_drugbank_canonical.csv)
  - DRKG: [GitHub repository](https://github.com/gnn4dr/DRKG)


### Knowledge Feature Initialization
The element knowledge graph (EKG) is stored in the form of triples in `dataset/pretrain/ekg.csv`. The encoding method for the EKG is provided by [KGE-HAKE](https://github.com/MIRALab-USTC/KGE-HAKE/).

Additionally, the drug knowledge graph is also encoded in the DRKG repository.

### `cluster.py`: Clustering Molecules for Model Training

To achieve optimal molecular similarity within small batches, we use `cluster.py` to generate batches. A reference batch can be found in the directory `dataset/pretrain/`.

#### Key Functions in `cluster.py`

1. **`get_fingerprint(smiles)`**: 
   - This function generates a molecular fingerprint from a SMILES string using RDKit's fingerprint generator.
   - **Usage**: 
     ```python
     fingerprint = get_fingerprint("CCO")  # Example SMILES
     ```

2. **`compute_fingerprints(df)`**: 
   - This function computes the fingerprints for all molecules in a given dataframe.
   - **Usage**: 
     ```python
     df = pd.read_csv("molecules.csv")
     df = compute_fingerprints(df)
     ```

3. **`get_cluster(sample_all=250000, n_clusters=1250)`**: 
   - This function returns a KMeansConstrained model for clustering molecules into a specified number of clusters.
   - **Usage**: 
     ```python
     cluster_model, min_smi_sample = get_cluster(250000, 1250)
     ```

4. **`get_similar_molecules(cluster_centers, with_id, n=20)`**:
   - This function retrieves the top `n` similar molecules for each cluster center based on fingerprint similarity.
   - **Usage**: 
     ```python
     similar_molecules = get_similar_molecules(cluster_centers, df, 20)
     ```

5. **`cluster(f, batch_num=256, batch_num_rest=6)`**: 
   - This is the main function to cluster molecules into batches for model training. It reads a CSV file containing the molecules (including their SMILES and drug IDs), computes their fingerprints, performs clustering, and assigns molecules to batches.
   - **Usage**: 
     ```bash
     python cluster.py
     ```

#### How to Use `cluster.py`

To run the clustering process, use the `cluster.py` script to organize your molecules into batches. The script will output the clustered batches and any unused drug IDs that couldn't be assigned.

1. **Prepare Data**:
   - Ensure your molecule data is in a CSV file where each row contains the SMILES string of a molecule, along with the corresponding drug ID (if available).
   - Example columns: `drugid`, `smiles`

2. **Run Clustering**:
   Run the `cluster.py` script using the following command:
   ```bash
   python cluster.py
   ```

   This will read from the file `../dataset/pretrain/zinc15_drugbank_canonical.csv` and output the clustered batches in `../dataset/pretrain/batches.csv`.

3. **Parameters**:
   - **`f`**: The file path to the input data (CSV file).
   - **`batch_num`**: The desired number of molecules per batch (default 256).
   - **`batch_num_rest`**: The number of additional molecules to complete the batch size (default 6).

4. **Output**:
   - The script will generate two files:
     - `batches.csv`: Contains the clustered batches of molecules.
     - `unused.csv`: Contains any drug IDs that could not be assigned to a batch.

### Example Command:
To run the clustering script with the default settings, simply execute:
```bash
python cluster.py
```

If you need to specify a different input file or batch size, modify the parameters in the script.


### Pre-training Data
For pre-training, we use 250K unlabeled molecules sampled from the ZINC 250 dataset. The data is aligned with the DRKG (Drug Repurposing Knowledge Graph).

- **Raw Pre-training Data**: The data can be accessed from the following sources:
  - ZINC15: [zinc15_drugbank_canonical.csv](dataset/pretrain/zinc15_drugbank_canonical.csv)
  - DRKG: [GitHub repository](https://github.com/gnn4dr/DRKG)

To achieve optimal molecular similarity within small batches, we use `cluster.py` to generate batches. A reference batch can be found in the directory `dataset/pretrain/`.

### Knowledge Feature Initialization
The element knowledge graph (EKG) is stored in the form of triples in `dataset/pretrain/ekg.csv`. The encoding method for the EKG is provided by [KGE-HAKE](https://github.com/MIRALab-USTC/KGE-HAKE/).

Additionally, the drug knowledge graph is also encoded in the DRKG repository.

## Running the Code



### Pre-train Models

1. **Pre-training Model Execution**: 
   The pre-training process is described in detail in the file `pretrain.py`. This file initializes the models and performs the training loop.

   Key components:
   - **HMGEncoder.py**: Implements the heterogeneous molecular graph encoder.
   - **ContrastiveLoss.py**: Defines the contrastive loss used during pre-training.
   - **HGTLayer.py**: Handles message passing between nodes and edges.

   The pre-training process can be executed with the following command:

   ```bash
   python pretrain.py --pretrain_data ./dataset/pretrain/batches.csv --element_KG ./dataset/pretrain/ekg.csv --batch_size 256 --lr 1e-4 --epoch_num 256 --element_embedding ./path/to/element_embedding.npy
   ```

   ### Pre-train Arguments Explanation:
   The `get_args` function in `pretrain.py` parses the command-line arguments. Below are the main arguments explained with usage examples:

   - **--seed** (default=42): Sets the random seed for reproducibility.
     ```bash
     --seed 1234
     ```
   - **--gpu** (default=3): Specifies the GPU device number.
     ```bash
     --gpu 0
     ```
   - **--hmg_dir** (optional): Directory to store pre-trained data. If the data is already processed and stored in a file, it can be loaded via this parameter.
     ```bash
     --hmg_dir ./dataset/pretrain/
     ```
   - **--pretrain_data**: Path to the pre-training data, which includes SMILES strings and corresponding drug IDs.
     ```bash
     --pretrain_data ../dataset/pretrain/batches.csv
     ```
   - **--element_KG**: Path to the element knowledge graph.
     ```bash
     --element_KG ./dataset/pretrain/ekg.csv
     ```
   - **--element_embedding**: Path to the element embedding file. This should point to a `.npy` file containing the pre-trained element embeddings.
     ```bash
     --element_embedding ./dataset/pretrain/element_embedding.npy
     ```
   - **--batch_size** (default=256): Batch size for training.
     ```bash
     --batch_size 128
     ```
   - **--hidden_feats** (default=256): Number of hidden features in the model.
     ```bash
     --hidden_feats 512
     ```
   - **--num_step_message_passing** (default=6): Number of message-passing steps, which controls the depth of the model.
     ```bash
     --num_step_message_passing 8
     ```
   - **--lr** (default=1e-4): Learning rate.
     ```bash
     --lr 1e-3
     ```
   - **--epoch_num** (default=256): Total number of training epochs.
     ```bash
     --epoch_num 100
     ```
   - **--temperature** (default=0.1): The temperature parameter for the contrastive loss function, which controls the range of similarity impact.
     ```bash
     --temperature 0.2
     ```

2. **Fine-tuning Models**: 
   After pre-training, the model can be fine-tuned on downstream tasks. Fine-tuning is handled in `finetune.py`. To fine-tune the model, run:

   ```bash
   python finetune.py --data_name [dataset_name] --encoder_path [pretrained_model_path] --gpu 0 --batch_size 256 --lr 1e-4 --epoch_num 50 --element_embedding ./path/to/element_embedding.npy
   ```

   ### Fine-tuning Arguments Explanation:
   The `get_args` function in `finetune.py` also parses command-line arguments. Below are the main arguments explained with usage examples:

   - **--seed** (default=42): Sets the random seed for reproducibility.
     ```bash
     --seed 42
     ```
   - **--gpu** (default=3): Specifies the GPU device number.
     ```bash
     --gpu 1
     ```
   - **--data_name**: The name of the dataset to use for fine-tuning. Available options include: 'SIDER', 'FreeSolv', 'Lipophilicity', etc.
     ```bash
     --data_name SIDER
     ```
   - **--batch_size** (default=256): Batch size for training.
     ```bash
     --batch_size 128
     ```
   - **--encoder_path**: Path to the pre-trained model to be used for fine-tuning.
     ```bash
     --encoder_path ./model/pretrained_model.pth
     ```
   - **--element_embedding**: Path to the element embedding file. This should point to a `.npy` file containing the pre-trained element embeddings.
     ```bash
     --element_embedding ./dataset/pretrain/element_embedding.npy
     ```
   - **--lr** (default=1e-4): Learning rate.
     ```bash
     --lr 1e-3
     ```
   - **--epoch_num** (default=500): Total number of fine-tuning epochs.
     ```bash
     --epoch_num 100
     ```
   - **--hidden_feats** (default=256): Number of hidden features in the model.
     ```bash
     --hidden_feats 512
     ```
   - **--num_step_message_passing** (default=6): Number of message-passing steps, which controls the depth of the model.
     ```bash
     --num_step_message_passing 8
     ```

