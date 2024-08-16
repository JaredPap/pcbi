# -*- coding: utf-8 -*-
# @Time    : 2023/11/13 上午10:18
# @Author  : Chen Mukun
# @File    : cluster.py
# @Software: PyCharm
# @desc    :
from collections import defaultdict
import random
from k_means_constrained import KMeansConstrained
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

import concurrent.futures
import math
import numpy as np

from sklearn.neighbors import NearestNeighbors

# Initialize the fingerprint generator with specific parameters
fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, includeChirality=True, useBondTypes=True, fpSize=256)

# 计算分子指纹
def get_fingerprint(smiles):
    """ Generate a molecular fingerprint from a SMILES string. """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return fp_gen.GetFingerprintAsNumPy(mol)
    else:
        print(smiles)
        return None


def compute_fingerprints(df):
    """ Compute fingerprints for all molecules in the dataframe. """
    smiles = df['smiles'].tolist()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        fingerprints = list(executor.map(get_fingerprint, smiles))
    df.loc[:, 'fp'] = fingerprints
    return df


def get_cluster(sample_all=250000, n_clusters=1250):
    return KMeansConstrained(
        n_clusters=n_clusters,
        size_min=int(sample_all / n_clusters * 0.9),
        size_max=int(sample_all / n_clusters * 1.1),
        random_state=42,
        n_jobs=20), int(sample_all / n_clusters * 0.9)


def get_similar_molecules(cluster_centers, with_id, n=20):
    """
        Retrieve the top `n` similar molecules for each cluster center based on fingerprint similarity.
        Parameters:
            cluster_centers (np.array): Numpy array of cluster centers' fingerprints.
            with_id (pd.DataFrame): DataFrame containing molecules with drug IDs and their fingerprints.
            n (int): Number of similar molecules to retrieve for each cluster center.
        Returns:
            pd.DataFrame: DataFrame of the top `n` similar molecules for each cluster center.
    """
    # Initialize the NearestNeighbors model
    nn = NearestNeighbors(n_neighbors=n, algorithm='ball_tree', metric='euclidean')
    nn.fit(with_id['fp'].values.tolist())  # Fit the model on the fingerprints of molecules with IDs

    # Find the n closest points for each cluster center
    distances, indices = nn.kneighbors(cluster_centers)

    # Retrieve the similar molecules' indices
    similar_molecules_indices = indices.flatten()
    unique_indices = np.unique(similar_molecules_indices)

    # Select the similar molecules from the DataFrame
    similar_molecules = with_id.iloc[unique_indices]

    return similar_molecules

def cluster(f, batch_num=256, batch_num_rest=6):
    """ Cluster molecules into batches for model training, adding similar molecules as needed. """
    batch_num = batch_num - batch_num_rest  # Adjust batch size for the amount to fill with similar molecules

    # df = pd.read_csv(f, sep="\t").iloc[np.r_[0:1000, -100:0]]
    df = pd.read_csv(f, sep="\t")
    print("computing fingerprints")
    df = compute_fingerprints(df)
    print("compute fingerprints over")

    without_id = df[df['drugid'].isna()].sample(frac=1).reset_index(drop=True)

    print("clustering")
    cluster_model, min_smi_sample = get_cluster(sample_all=len(without_id),
                                                n_clusters=math.floor(len(without_id) // (0.7 * batch_num)))
    labels = cluster_model.fit_predict(without_id['fp'].tolist())
    centers = cluster_model.cluster_centers_
    clusters = defaultdict(list)
    for i in range(len(labels)):
        clusters[labels[i]].append(i)
    print("cluster over")

    print("assigning drugs")
    with_id = df[df['drugid'].notna()].sample(frac=1).reset_index(drop=True)
    used_drugs = set()
    results = []
    nn = NearestNeighbors(n_neighbors=batch_num - min_smi_sample, algorithm='auto', n_jobs=20)
    nn.fit(with_id['fp'].tolist())
    _, indices = nn.kneighbors(centers)

    without_id = without_id.drop('fp', axis=1)
    with_id = with_id.drop('fp', axis=1)

    for key in clusters:
        similar_molecules = indices[key][0:  batch_num - len(clusters[key])]
        used_drugs.update(with_id['drugid'].iloc[similar_molecules].tolist())
        results.append(pd.concat([without_id.iloc[clusters[key]], with_id.iloc[similar_molecules]]))
    print("assign drugs over")

    unused_drugs = set(with_id['drugid']) - used_drugs
    print("unused Drugs :")
    print(len(unused_drugs))

    results = complete(results, unused_drugs, with_id, batch_num_rest=batch_num_rest)
    combined_df = pd.concat(results, ignore_index=True)
    combined_df.to_csv('../dataset/pretrain/batches.csv', index=False)

    with open('../dataset/pretrain/unused.csv', "w") as f:
        f.write("\r\n".join(unused_drugs))
        f.flush()
        f.close()

    return results, unused_drugs


# len(results) * batch_num_rest < total_N
def complete(results, unused_drugs, with_id, batch_num_rest=6):
    """ Complete batches to ensure all batches have equal size by adding drugs from unused pool. """
    unused_drugs = list(unused_drugs)
    random.shuffle(unused_drugs)
    complete_drug_ids = []
    total_N = len(results) * batch_num_rest # Total drugs needed to complete batches
    repeat_time = total_N // len(unused_drugs)  # Number of times to repeat the list of unused drugs
    for i in range(repeat_time):
        complete_drug_ids.extend(unused_drugs)
    complete_drug_ids.extend(random.sample(unused_drugs[:-batch_num_rest], total_N % len(unused_drugs)))
    drug_to_smiles = with_id.set_index('drugid')['smiles'].to_dict()

    additional_drugs = pd.DataFrame({
        'drugid': complete_drug_ids,
        'smiles': [drug_to_smiles[drug] for drug in complete_drug_ids if drug in drug_to_smiles]
    })
    additional_drugs['Group'] = 'Supplemental'

    for i, result in enumerate(results):
        start_index = i * batch_num_rest
        # 将对应的药物添加到 DataFrame
        appendix = additional_drugs.iloc[start_index:start_index + batch_num_rest]
        result = pd.concat([result, appendix], ignore_index=True)
        result['Group'] = i
        results[i] = result
    return results


if __name__ == '__main__':
    cluster(f="../dataset/pretrain/zinc15_drugbank_canonical.csv", batch_num=256, batch_num_rest=6)
