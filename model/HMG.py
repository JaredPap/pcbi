# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from rdkit import Chem
import torch
from torch.utils.data import Dataset
import dgl
from dgl.dataloading import GraphDataLoader
from rdkit.Chem.BRICS import FindBRICSBonds, BreakBRICSBonds
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import MACCSkeys
from rdkit import RDConfig
from rdkit import RDLogger
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np

RDLogger.DisableLog('rdApp.*')
import os

fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

BOND_FEATURES = {
    'bond_type': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    'stereo': [
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ],

}


# 1*16
def bond_features(bond: Chem.rdchem.Bond):
    if bond is None:
        fbond = [1] + [0] * (len(BOND_FEATURES['bond_type']) + 1 + len(BOND_FEATURES['stereo']) + 1 + 3)
    else:
        fbond = [0] + \
                onek_encoding_unk(bond.GetBondType(), BOND_FEATURES['bond_type']) + \
                onek_encoding_unk(bond.GetStereo(), BOND_FEATURES['stereo']) + \
                [1 if bond.GetIsConjugated() else 0] + \
                [1 if bond.GetIsAromatic() else 0] + \
                [1 if bond.IsInRing() else 0]
    # dim = 1 + 5 + 7 + 3
    return fbond


def pharm_property_types_feats(mol, factory=factory):
    types = [i.split('.')[1] for i in factory.GetFeatureDefs().keys()]
    feats = [i.GetType() for i in factory.GetFeaturesForMol(mol)]
    result = [0] * len(types)
    for i in range(len(types)):
        if types[i] in list(set(feats)):
            result[i] = 1
    return result


def GetBricsBonds(mol):
    brics_bonds = list()
    brics_bonds_rules = list()
    bonds_tmp = FindBRICSBonds(mol)

    bonds = [b for b in bonds_tmp]
    for item in bonds:  # item[0] is bond, item[1] is brics type
        brics_bonds.append([int(item[0][0]), int(item[0][1])])
        brics_bonds_rules.append([[int(item[0][0]), int(item[0][1])], GetBricsBondFeature([item[1][0], item[1][1]])])
        brics_bonds.append([int(item[0][1]), int(item[0][0])])
        brics_bonds_rules.append([[int(item[0][1]), int(item[0][0])], GetBricsBondFeature([item[1][1], item[1][0]])])

    result = []
    for bond in mol.GetBonds():
        beginatom = bond.GetBeginAtomIdx()
        endatom = bond.GetEndAtomIdx()
        if [beginatom, endatom] in brics_bonds:
            result.append([bond.GetIdx(), beginatom, endatom])
    return result, brics_bonds_rules


def GetBricsBondFeature(action):
    result = []
    start_action_bond = int(action[0]) if (action[0] != '7a' and action[0] != '7b') else 7
    end_action_bond = int(action[1]) if (action[1] != '7a' and action[1] != '7b') else 7
    emb_0 = [0 for i in range(17)]
    emb_1 = [0 for i in range(17)]
    emb_0[start_action_bond] = 1
    emb_1[end_action_bond] = 1
    result = emb_0 + emb_1
    return result


def maccskeys_emb(mol):
    return list(MACCSkeys.GenMACCSKeys(mol))


def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)  # aviod index 0
    return mol


def GetFragmentFeats(mol):
    break_bonds = [mol.GetBondBetweenAtoms(i[0][0], i[0][1]).GetIdx() for i in FindBRICSBonds(mol)]
    if break_bonds == []:
        tmp = mol
    else:
        tmp = Chem.FragmentOnBonds(mol, break_bonds, addDummies=False)
    frags_idx_lst = Chem.GetMolFrags(tmp)
    result_ap = {}
    result_p = {}
    pharm_id = 0
    for frag_idx in frags_idx_lst:
        for atom_id in frag_idx:
            result_ap[atom_id] = pharm_id
        try:
            mol_pharm = Chem.MolFromSmiles(Chem.MolFragmentToSmiles(mol, frag_idx))
            emb_0 = maccskeys_emb(mol_pharm)  # 167
            emb_1 = pharm_property_types_feats(mol_pharm)  # 27
        except Exception:
            emb_0 = [0 for i in range(167)]
            emb_1 = [0 for i in range(27)]
        result_p[pharm_id] = emb_0 + emb_1
        pharm_id += 1
    return result_ap, result_p


def onek_encoding_unk(value, choices):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


ATOM_FEATURES = {
    'atomic_num': [5, 6, 7, 8, 9, 15, 16, 17, 19, 27, 33, 35, 53, 64, 78],  # 15
    'total_degree': [0, 1, 2, 3, 4, 5, 6],  # 7
    'degree': [0, 1, 2, 3, 4, 5, 6],  # 7
    'formal_charge': [-3, -2, - 1, 0, 1, 2, 3],  # 7
    'chiral_tag': [Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                   Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                   Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW],  # 3
    'num_Hs': [0, 1, 2, 3, 4],  # 5
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],  # 5
    "total_valence": [0, 1, 2, 3, 4, 5, 6],  # 7
}


# 1* 68
def atom_features(atom: Chem.rdchem.Atom):
    features = onek_encoding_unk(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num']) + \
               onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['total_degree']) + \
               onek_encoding_unk(atom.GetDegree(), ATOM_FEATURES['degree']) + \
               onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
               onek_encoding_unk(atom.GetChiralTag(), ATOM_FEATURES['chiral_tag']) + \
               onek_encoding_unk(atom.GetTotalNumHs(), ATOM_FEATURES['num_Hs']) + \
               onek_encoding_unk(atom.GetHybridization(), ATOM_FEATURES['hybridization']) + \
               onek_encoding_unk(atom.GetTotalValence(), ATOM_FEATURES['total_valence']) + \
               [atom.GetIdx() + 1] + \
               [1 if atom.GetIsAromatic() else 0] + \
               [1 if atom.IsInRing() else 0] + \
               [atom.GetMass() * 0.01]
    # dim = 16 + 8 +8  +8+ 4 + 6 + 6+ 8+ 4
    return features


def Mol2HeteroGraph(mol):
    # build graphs
    edge_types = [('a', 'b', 'a'), ('f', 'r', 'f'), ('a', 'j', 'f'), ('f', 'j', 'a')]
    edges = {k: [] for k in edge_types}
    result_ap, result_p = GetFragmentFeats(mol)
    reac_idx, bbr = GetBricsBonds(mol)

    for bond in mol.GetBonds():
        edges[('a', 'b', 'a')].append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edges[('a', 'b', 'a')].append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])

    for r in reac_idx:
        begin = r[1]
        end = r[2]
        edges[('f', 'r', 'f')].append([result_ap[begin], result_ap[end]])
        edges[('f', 'r', 'f')].append([result_ap[end], result_ap[begin]])

    for k, v in result_ap.items():
        edges[('a', 'j', 'f')].append([k, v])
        edges[('f', 'j', 'a')].append([v, k])
    g = dgl.heterograph(edges)

    f_atom = []
    for idx in g.nodes('f'):
        atom = mol.GetAtomWithIdx(idx.item())
        f_atom.append(atom_features(atom))

    f_atom = torch.FloatTensor(f_atom)
    g.nodes['f'].data['x'] = f_atom

    f_pharm = []
    for k, v in result_p.items():
        f_pharm.append(v)
    f_pharm = torch.FloatTensor(f_pharm)
    g.nodes['f'].data['x'] = f_pharm

    # features of edges
    f_bond = []
    src, dst = g.edges(etype=('a', 'b', 'a'))
    for i in range(g.num_edges(etype=('a', 'b', 'a'))):
        f_bond.append(bond_features(mol.GetBondBetweenAtoms(src[i].item(), dst[i].item())))
    g.edges[('a', 'b', 'a')].data['x'] = torch.FloatTensor(f_bond)

    f_reac = []
    src, dst = g.edges(etype=('f', 'r', 'f'))
    for idx in range(g.num_edges(etype=('f', 'r', 'f'))):
        p0_g = src[idx].item()
        p1_g = dst[idx].item()
        for i in bbr:
            p0 = result_ap[i[0][0]]
            p1 = result_ap[i[0][1]]
            if p0_g == p0 and p1_g == p1:
                f_reac.append(i[1])
    g.edges[('f', 'r', 'f')].data['x'] = torch.FloatTensor(f_reac)

    return g


class MolGraphSet(Dataset):
    def __init__(self, df, log=print):
        self.data = df
        self.mols = []
        self.graphs = []
        self.ids = []
        for i, row in df.iterrows():
            id = row['ids']
            smi = row['smiles']
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    log('invalid', smi)
                else:
                    g = Mol2HeteroGraph(mol)
                    if g.num_nodes('a') == 0:
                        log('no edge in graph', smi)
                    else:
                        self.mols.append(mol)
                        self.graphs.append(g)
                        self.ids.append(id)
            except Exception as e:
                log(e, 'invalid', smi)

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, idx):
        return self.graphs[idx]


def create_dataloader(file, shuffle=True, train=True):
    dataset = MolGraphSet(pd.read_csv(file))
    if train:
        batch_size = 1
    else:
        batch_size = min(4200, len(dataset))

    dataloader = GraphDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


def random_split(load_path, save_dir, sizes=[0.9, 0.1],
                 seed=42):
    df = pd.read_csv(load_path, sep='\t', header=None,
                     names=['index', 'ids', 'smiles'], usecols=['index', 'ids', 'smiles'])
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(seed)
    df = df.loc[torch.randperm(len(df))].reset_index(drop=True)
    train_size = int(sizes[0] * len(df))
    train = df[:train_size]
    test = df[train_size:]
    train.to_csv(os.path.join(save_dir) + f'train.csv', index=False)
    test.to_csv(os.path.join(save_dir) + f'test.csv', index=False)
    return train, test


def get_HMG_data():
    seed = 42
    train, test = random_split('../data/DDIDataset/DrugSetSmliesVector.txt', '../data/', seed=seed)
    trainloader = create_dataloader("../data/train.csv", shuffle=True)
    testloader = create_dataloader("../data/test.csv", shuffle=False, train=False)
    return trainloader, test


if __name__ == '__main__':
    get_HMG_data()


