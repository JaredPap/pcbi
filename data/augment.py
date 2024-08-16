# -*- coding: utf-8 -*-
# @Time    : 2023/9/3 5:36 下午
# @Author  : Chen Mukun
# @File    : HMG_features.py
# @Software: PyCharm
# @desc    :
from typing import List

import dgl
import networkx as nx
import pandas as pd
import torch
from dgl.dataloading import GraphDataLoader
from rdkit import Chem
from rdkit import RDConfig
from rdkit import RDLogger
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import MACCSkeys
from rdkit.Chem.BRICS import FindBRICSBonds
from torch.utils.data import Dataset
import torch.nn.functional as F
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate

RDLogger.DisableLog('rdApp.*')
import os

fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
text = """Alkyl                   [CX4]
Alkenyl                 [$([CX3]=[CX3])]
Alkynyl                 [$([CX2]#C)]
Phenyl                  c
bromoalkane             [Br]
chloro                  [Cl]
fluoro                  [F]
halo                    [#6][F,Cl,Br,I]
iodo                    [I]
Acetal                  O[CH1][OX2H0]
Haloformyl              [CX3](=[OX1])[F,Cl,Br,I]
Hydroxyl                [#6][OX2H]
Aldehyde                [CX3H1](=O)[#6]
CarbonateEster          [CX3](=[OX1])(O)O
Carboxylate             [CX3](=O)[O-]
Carboxyl                [CX3](=O)[OX2H1]
Carboalkoxy             [CX3](=O)[OX2H0]
Ether                   [OD2]
Hemiacetal              O[CH1][OX2H1]
Hemiketal               OC[OX2H1]
Methylenedioxy          C([OX2])([OX2])
Hydroperoxy             O[OX2H]
Ketal                   OC[OX2H0]
Carbonyl                [CX3]=[OX1]
CarboxylicAnhydride     [CX3](=O)[OX2H0][CX3](=O)
OrthocarbonateEster     C([OX2])([OX2])([OX2])([OX2])
Orthoester              C([OX2])([OX2])([OX2])
Peroxy                  O[OX2H0]
Carboxamide             [NX3][CX3](=[OX1])[#6]
Amidine                 [NX3][CX3]=[NX2]
4ammoniumIon            [NX4+]
PrimaryAmine            [NX3;H2,H1;!$(NC=O)]
SecondaryAmine          [NX3;H2;!$(NC=O)]
TertiaryAmine           [NX3;!$(NC=O)]
Azide                   [$(*-[NX2-]-[NX2+]#[NX1]),$(*-[NX2]=[NX2+]=[NX1-])]
Azo                     [NX2]=N
Carbamate               [NX3,NX4+][CX3](=[OX1])[OX2,OX1-]
Cyanate                 OC#N
Isocyanate              [O]=[CX2]=[NX2]
Imide                   [CX3](=[OX1])[NX3H][CX3](=[OX1])
PrimaryAldimine         [CX3H1]=[NX2H1]
PrimaryKetimine         [CX3]=[NX2H1]
SecondaryAldimine       [CX3H1]=[NX2H0]
SecondaryKetimine       [CX3]=[NX2H0]
Nitrate                 [$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)]
Isonitrile              [CX1-]#[NX2+]
Nitrile                 [NX1]#[CX2]
Nitrosooxy              O[NX2]=[OX1]
Nitro                   [$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]
Nitroso                 [NX2]=[OX1]
Oxime                   C=N[OX2H1]
Pyridyl                 ccccnc
Disulfide               [#16X2H0]S
CarbodithioicAcid       [#16X2H1]C=[#16]
Carbodithio             [#16X2H0]C=[#16]
Sulfide                 [#16X2H0]
Sulfino                 [$([#16X3](=[OX1])[OX2H,OX1H0-]),$([#16X3+]([OX1-])[OX2H,OX1H0-])]
Sulfoate                [$([#16X4](=[OX1])(=[OX1])([#6])[OX2H0]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H0])]
Sulfonyl                [$([#16X4](=[OX1])(=[OX1])([#6])[#6]),$([#16X4+2]([OX1-])([OX1-])([#6])[#6])]
Sulfo                   [$([#16X4](=[OX1])(=[OX1])([#6])[OX2H,OX1H0-]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H,OX1H0-])]
Sulfinyl                [$([#16X3]=[OX1]),$([#16X3+][OX1-])]
Thial                   [#16]=[CX3H1]
CarbothioicOAcid        [OX2H1]C=[#16]
CarbothioicSAcid        [#16X2H1]C=O
Isothiocyanate          [#16]=[CX2]=[NX2]
Thiocyanate             [#16]C#N
Thiolester              [#16X2H0]C=O
Thionoester             [OX2H0]C=[#16]
Thioketone              [#16]=[CX3H0]
Sulfhydryl              [#16X2H]
Phosphate               [OX2H0][PX4](=[OX1])([OX2H1])([OX2H1])
Phosphino               [PX3]
Phosphodiester          [OX2H1][PX4](=[OX1])([OX2H0])([OX2H0])
Phosphono               [PX4](=[OX1])([OX2H1])([OX2H1])
Borino                  [BX3]([OX2H1])
Borinate                [BX3]([OX2H0])
Borono                  [BX3]([OX2H1])([OX2H1])
Boronate                [BX3]([OX2H0])([OX2H0])
Alkylaluminium          [#13].[#13]
Alkyllithium            [#3]
AlkylmagnesiumHalide    [#12X2][F,Cl,Br,I]
SilylEther              [#14X4][OX2]"""

funcgroups = text.strip().split('\n')
func_names = [i.split()[0] for i in funcgroups]
smart = [Chem.MolFromSmarts(i.split()[1]) for i in funcgroups]
smart2name = dict(zip(smart, func_names))

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

# random edge feats
# [('a', 'ad', 'd'), ('d', 'rad', 'a'), ('fr', 'fd', 'd'), ('d', 'rfd', 'fr')
ad_feat = torch.randn(1, 3)
rad_feat = torch.randn(1, 3)
fd_feat = torch.randn(1, 3)
rfd_feat = torch.randn(1, 3)

# [('a', 'b', 'a'), ('fr', 'r', 'fr'), ('a', 'j', 'fr')]
# [('a', 'rj', 'fr')]
# b
# r
j_feat = torch.randn(1, 3)
rj_feat = torch.randn(1, 3)

# [('a', 'ej', 'e'), ('fr', 'fj', 'fu'), ('e', 'pr', 'e'), ('fu', 'sj', 'fu'), ('e', 'uj', 'fu')]
# [('e', 'rej', 'a'), ('fu', 'rfj', 'fr'),('fu', 'ruj', 'e')]
ej_feat = torch.randn(1, 3)
fj_feat = torch.randn(1, 3)
# pr
uj_feat = torch.randn(1, 3)
rej_feat = torch.randn(1, 3)
rfj_feat = torch.randn(1, 3)
ruj_feat = torch.randn(1, 3)


class ElementKG:
    def __init__(self, g,
                 emb=None,
                 name_space=None):
        """
            Initialize the ElementKG class with a graph, optional embeddings, and an optional namespace.

                Args:
                    g (networkx.Graph): The graph representing the knowledge graph of elements.
                    emb (torch.Tensor, optional): Predefined embeddings for the nodes in the graph.
                    name_space (list, optional): List of node names; defaults to graph nodes if not provided.
        """
        # Use graph nodes as the default namespace if none is provided
        if name_space is None:
            self.name_space = list(g.nodes())  # func_names included
        else:
            self.name_space = name_space

        # Generate random embeddings if none are provided
        if emb is None:
            self.emb = torch.randn(len(self.name_space), 3)
        else:
            self.emb = emb

        self.kg = g
        self.ele_pro = {}  # Dictionary to store computed properties between elements

    def name_to_index(self, n):
        """
        Convert an element's name to its index in the name space.

            Args:
                n (str): The name of the element.

            Returns:
                int: The index of the element in the name space.
        """
        return self.name_space.index(n)

    def index_to_emb(self, id):
        """
            Retrieve the embedding for an element based on its index.

            Args:
                id (int): The index of the element.

            Returns:
                torch.Tensor: The embedding of the specified element.
        """
        return self.emb[id]

    def common_property(self, e1, e2):
        """
            Compute a common property between two elements, defined as nodes reachable within two hops in the KG.

            Args:
                e1 (str): The name of the first element.
                e2 (str): The name of the second element.

            Returns:
                list: List of indices of the intermediate nodes in all simple paths of length 2 between e1 and e2.
        """
        if (self.name_to_index(e1), self.name_to_index(e2)) not in self.ele_pro:
            self.ele_pro[(self.name_to_index(e1), self.name_to_index(e2))] = []
            for path in nx.all_simple_paths(self.kg, source=e1, target=e2, cutoff=2):
                self.ele_pro[(self.name_to_index(e1), self.name_to_index(e2))].append(self.name_to_index(path[1]))
        return self.ele_pro[(self.name_to_index(e1), self.name_to_index(e2))]


def match_fg(mol):
    """
        Find all functional groups in a molecule that match predefined SMARTS patterns.

        Args:
            mol (rdkit.Chem.Mol): The RDKit molecule object to be analyzed.

        Returns:
            list: A list of names corresponding to the functional groups found in the molecule.
    """
    fg = []
    for id, sm in enumerate(smart):
        if mol.HasSubstructMatch(sm):
            fg.append(func_names[id])
    return fg


def onek_encoding_unk(value,
                      choices):
    """
    Create a one-hot encoding for a specified value from a list of choices.
    Includes an additional category for unknown values not found in the list.

    Args:
        value (str): The value to encode.
        choices (list): A list of possible valid choices.

    Returns:
        list: A list representing the one-hot encoding. An extra bit is added for 'unknown' values.
    """
    # Initialize a list of zeros with length of choices plus one for the unknown category.
    encoding = [0] * (len(choices) + 1)
    # Try to find the index of the value in choices. If not found, use -1 for the unknown category.
    index = choices.index(value) if value in choices else -1
    # Set the appropriate index to 1 in the encoding list.
    encoding[index] = 1
    return encoding


# 1*16
def bond_features(bond: Chem.rdchem.Bond):
    """
    Extracts a feature vector from a chemical bond.

    Args:
        bond (Chem.rdchem.Bond): An RDKit Bond object.

    Returns:
        list: A list of integers representing the features of the bond. This includes one-hot encoded
              bond types, stereochemistry, and binary features indicating if the bond is conjugated,
              aromatic, and whether it is part of a ring.
    """
    # Define possible bond types and stereochemistry options
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


# 1*68
def atom_features(atom: Chem.rdchem.Atom):
    """
        Generates a feature vector for an atom.

        Args:
            atom (Chem.rdchem.Atom): An RDKit Atom object.

        Returns:
            list: A feature vector representing various atomic properties.
    """
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


def GetFragmentFeats(mol,
                     ele=False):
    """
        Extracts features from fragments of a molecule.

        Args:
            mol (Chem.Mol): RDKit molecule object.
            ele (bool, optional): If True, extract element kg features using SMARTS. Defaults to False.

        Returns:
            tuple: Contains mappings of atoms to fragments, fragment features, and optional functional group features.
    """
    break_bonds = [mol.GetBondBetweenAtoms(i[0][0], i[0][1]).GetIdx() for i in FindBRICSBonds(mol)]
    if break_bonds == []:
        tmp = mol
    else:
        tmp = Chem.FragmentOnBonds(mol, break_bonds, addDummies=False)
    frags_idx_lst = Chem.GetMolFrags(tmp)
    afr = {}  # Atom to fragment mapping
    frag_feat = {}  # Fragment features
    frfu = {}  # Fragment to Functional group mapping, if 'ele' is True
    frag_id = 0
    for frag_idx in frags_idx_lst:
        for atom_id in frag_idx:
            afr[atom_id] = frag_id
        try:
            mol_pharm = Chem.MolFromSmiles(Chem.MolFragmentToSmiles(mol, frag_idx))
            if ele:
                frfu[frag_id] = match_fg(mol_pharm)
            emb_0 = maccskeys_emb(mol_pharm)  # MACCS keys (167 bits)
            emb_1 = pharm_property_types_feats(mol_pharm)  # Pharmacophore properties (27 features)
        except Exception:
            emb_0 = [0 for i in range(167)]
            emb_1 = [0 for i in range(27)]
        frag_feat[frag_id] = emb_0 + emb_1
        frag_id += 1
    return afr, frag_feat, frfu


def GetBricsBondFeature(action):
    """
        Converts a Brics bond action into a feature vector representing the start and end bond types.

        Args:
            action (tuple or list): A pair of strings or numbers indicating the bond actions,
                                    with special cases for '7a' and '7b' treated as type '7'.

        Returns:
            list: A feature vector where the start and end bond types are encoded as one-hot vectors.
    """
    start_action_bond = int(action[0]) if (action[0] != '7a' and action[0] != '7b') else 7
    end_action_bond = int(action[1]) if (action[1] != '7a' and action[1] != '7b') else 7
    emb_0 = [0 for i in range(17)]
    emb_1 = [0 for i in range(17)]
    emb_0[start_action_bond] = 1
    emb_1[end_action_bond] = 1
    result = emb_0 + emb_1
    return result


def pharm_property_types_feats(mol,
                               factory=factory):
    """
        Generate a binary feature vector indicating the presence of various pharmacophore features in a molecule.

        Args:
            mol (rdkit.Chem.Mol): RDKit molecule object.
            factory (rdkit.Chem.Pharm2D.SigFactory): A factory object used to define and generate pharmacophore features.

        Returns:
            list: A binary list where each position corresponds to a feature type defined in the factory,
                  with 1 indicating presence and 0 absence.
    """
    if factory is None:
        factory = Gobbi_Pharm2D.factory
    types = [i.split('.')[1] for i in factory.GetFeatureDefs().keys()]
    feats = [i.GetType() for i in factory.GetFeaturesForMol(mol)]
    result = [0] * len(types)
    for i in range(len(types)):
        if types[i] in list(set(feats)):
            result[i] = 1
    return result


def GetBricsBonds(mol):
    """
        Identify and classify bonds in a molecule based on BRICS rules.

        Args:
            mol (Chem.Mol): RDKit molecule object.

        Returns:
            tuple: A tuple containing two lists:
                   - A list of bonds identified as BRICS bonds with their indices and atom indices.
                   - A list of BRICS bonds with corresponding feature vectors.
    """

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


def maccskeys_emb(mol):
    return list(MACCSkeys.GenMACCSKeys(mol))


def Mol2HeteroGraph(mol,
                    ele=False,
                    elementKG=None):
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)

    ele_list = None
    ele_map = None
    fu_list = None
    fu_map = None
    # Initialize variables for storing graph edges
    edge_types = [('a', 'b', 'a'), ('fr', 'r', 'fr'), ('a', 'j', 'fr')]
    edge_types.extend([('fr', 'rj', 'a')])
    if ele:
        edge_types.extend([('a', 'ej', 'e'), ('fr', 'fj', 'fu'), ('e', 'pr', 'e'), ('fu', 'sj', 'fu'), ('e', 'uj', 'fu')])
        edge_types.extend([('e', 'rej', 'a'), ('fu', 'rfj', 'fr'), ('fu', 'ruj', 'e')])
        ele_list = []
        ele_map = []
        fu_list = []
        fu_map = []
    edges = {k: [] for k in edge_types}

    # afr: atom to fragment edge; fragment features
    afr, frag_feat, frfu = GetFragmentFeats(mol, ele=ele)

    # reaction type and feature
    reac_idx, bbr = GetBricsBonds(mol)

    # ('atom', 'bond', 'atom')
    for bond in mol.GetBonds():
        src = bond.GetBeginAtomIdx()
        dst = bond.GetEndAtomIdx()
        edges[('a', 'b', 'a')].extend([[src, dst], [dst, src]])

    # ('fragment', 'reaction', 'fragment')
    for r in reac_idx:
        src_frag = afr[r[1]]
        dst_frag = afr[r[2]]
        edges[('fr', 'r', 'fr')].extend([[src_frag, dst_frag], [dst_frag, src_frag]])

    # ('atom', 'join', 'frag')
    for k, v in afr.items():
        edges[('a', 'j', 'fr')].append([k, v])
        edges[('fr', 'rj', 'a')].append([v, k])

    # Additional edge types if element kg are included
    if ele:
        # ('atom', 'ej', 'element')
        for atom in mol.GetAtoms():
            if atom.GetSymbol() not in ele_list:
                ele_list.append(atom.GetSymbol())
                ele_map.append(elementKG.name_to_index(atom.GetSymbol()))
            edges[('a', 'ej', 'e')].append([atom.GetIdx(), ele_list.index(atom.GetSymbol())])
            edges[('e', 'rej', 'a')].append([ele_list.index(atom.GetSymbol()), atom.GetIdx()])

        # ('fragment', 'fj', 'function group')
        for k, v in frfu.items():
            for fu in v:
                if fu not in fu_list:
                    fu_list.append(fu)
                    fu_map.append(elementKG.name_to_index(fu))
                edges[('fr', 'fj', 'fu')].append([k, fu_list.index(fu)])
                edges[('fu', 'rfj', 'fr')].append([fu_list.index(fu), k])
    g = dgl.heterograph(edges)

    # initial features of nodes (atom)
    f_atom = []
    for idx in g.nodes('a'):
        atom = mol.GetAtomWithIdx(idx.item())
        f_atom.append(atom_features(atom))
    f_atom = torch.FloatTensor(f_atom)
    g.nodes['a'].data['x'] = f_atom

    # initial features of nodes (fragment)
    f_fr = []
    # already sorted [0,1,2....]
    for k, v in frag_feat.items():
        f_fr.append(v)
    f_fr = torch.FloatTensor(f_fr)
    g.nodes['fr'].data['x'] = f_fr

    # element view
    if ele:
        # initial features of nodes (element)
        f_e = []
        for idx in g.nodes('e'):
            f_e.append(ele_map[idx.item()])
        g.nodes['e'].data['x'] = elementKG.index_to_emb(f_e)
        # initial features of nodes (function group)
        f_fu = []
        for idx in g.nodes('fu'):
            f_fu.append(fu_map[idx.item()])
        g.nodes['fu'].data['x'] = elementKG.index_to_emb(f_fu)

    # features of edges ('atom', 'bond', 'atom')
    f_bond = []
    src, dst = g.edges(etype=('a', 'b', 'a'))
    for i in range(g.num_edges(etype=('a', 'b', 'a'))):
        f_bond.append(bond_features(mol.GetBondBetweenAtoms(src[i].item(), dst[i].item())))
    g.edges['b'].data['x'] = torch.FloatTensor(f_bond)

    # features of edges ('fragment', 'reaction', 'fragment')
    f_reac = []
    src, dst = g.edges(etype=('fr', 'r', 'fr'))
    for idx in range(g.num_edges(etype=('fr', 'r', 'fr'))):
        p0_g = src[idx].item()
        p1_g = dst[idx].item()
        for i in bbr:
            p0 = afr[i[0][0]]
            p1 = afr[i[0][1]]
            if p0_g == p0 and p1_g == p1:
                f_reac.append(i[1])
    g.edges['r'].data['x'] = torch.FloatTensor(f_reac)

    # features of edges ('atom', 'join', 'fragment')
    g.edges['j'].data['x'] = j_feat.repeat(g.num_edges("j"), 1)
    g.edges['rj'].data['x'] = rj_feat.repeat(g.num_edges("j"), 1)

    if ele:
        # features of edges ('atom', 'is', 'element')
        g.edges['ej'].data['x'] = ej_feat.repeat(g.num_edges("ej"), 1)
        g.edges['rej'].data['x'] = rej_feat.repeat(g.num_edges("rej"), 1)

        # features of edges ('fragment', 'have', 'function group')
        g.edges['fj'].data['x'] = fj_feat.repeat(g.num_edges("fj"), 1)
        g.edges['rfj'].data['x'] = rfj_feat.repeat(g.num_edges("rfj"), 1)

        # features of edges ('element', 'property', 'element')
        p_edges = [[], []]
        property_feat = []
        if len(ele_list) > 1:
            for m in range(len(ele_list)):
                for n in range(m + 1, len(ele_list)):
                    common_properties = elementKG.common_property(ele_list[m], ele_list[n])
                    if len(common_properties) != 0:
                        p_edges[0].extend([m] * len(common_properties))
                        p_edges[1].extend([n] * len(common_properties))
                        property_feat.append(elementKG.index_to_emb(common_properties))
            g.add_edges(torch.tensor(p_edges[0]), torch.tensor(p_edges[1]), etype='pr')
            g.add_edges(torch.tensor(p_edges[1]), torch.tensor(p_edges[0]), etype='pr')
            g.edges['pr'].data['x'] = torch.cat(property_feat, dim=0).repeat(2, 1)

        # features of edges ('function group', 'property', 'function group')
        s_edges = [[], []]
        property_feat = []
        if len(fu_list) > 1:
            for m in range(len(fu_list)):
                for n in range(m + 1, len(fu_list)):
                    common_properties = elementKG.common_property(fu_list[m], fu_list[n])
                    if len(common_properties) != 0:
                        s_edges[0].extend([m] * len(common_properties))
                        s_edges[1].extend([n] * len(common_properties))
                        property_feat.append(elementKG.index_to_emb(common_properties))
            g.add_edges(torch.tensor(s_edges[0]), torch.tensor(s_edges[1]), etype='sj')
            g.add_edges(torch.tensor(s_edges[1]), torch.tensor(s_edges[0]), etype='sj')
            g.edges['sj'].data['x'] = torch.cat(property_feat, dim=0).repeat(2, 1)

        # features of edges ('element', 'in', 'function group')
        i_edges = [[], []]
        for m in range(len(ele_list)):
            for n in range(len(fu_list)):
                common_properties = elementKG.common_property(ele_list[m], fu_list[n])
                if len(common_properties) != 0:
                    i_edges[0].append(m)
                    i_edges[1].append(n)
        if len(i_edges[0]) > 0:
            g.add_edges(torch.tensor(i_edges[0]), torch.tensor(i_edges[1]), etype='uj')
            g.add_edges(torch.tensor(i_edges[1]), torch.tensor(i_edges[0]), etype='ruj')
            g.edges['uj'].data['x'] = uj_feat.repeat(g.num_edges("uj"), 1)
            g.edges['ruj'].data['x'] = ruj_feat.repeat(g.num_edges("ruj"), 1)
    return g


def add_new_edge_type(original_g,
                      edge_types):
    """
        Extend a DGL graph with new edge types without initial edges.

        Args:
            original_g (dgl.DGLGraph): The original DGL heterogeneous graph.
            edge_types (list): A list of new edge types to add, each edge type is a tuple (src_type, edge_type, dst_type).

        Returns:
            dgl.DGLGraph: A new DGL heterogeneous graph including the original and new edge types.
    """
    # Initialize the data dictionary from the original graph
    data_dict = {}
    for etype in original_g.canonical_etypes:
        src, dst = original_g.edges(etype=etype)
        data_dict[etype] = (src, dst)
    # Add new edge types with empty edge lists
    for edge_type in edge_types:
        data_dict[edge_type] = (torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64))
    # Create the new heterogeneous graph
    new_g = dgl.heterograph(data_dict)
    # Copy node features
    for ntype in original_g.ntypes:
        new_g.nodes[ntype].data.update(original_g.nodes[ntype].data)
    # Copy edge features for existing edge types
    for etype in original_g.canonical_etypes:
        new_g.edges[etype].data.update(original_g.edges[etype].data)
    return new_g


def hmg_to_drug(g: dgl.DGLGraph,
                drug_emb=torch.randn(1, 16)):
    """
        Extend a DGL graph with edges connecting existing 'atom' and 'fragment' nodes
        to a new 'drug' node, and initialize features for the new nodes and edges.

        Args:
            g (dgl.DGLGraph): The original graph representing hetero molecules.
            drug_emb (torch.Tensor, optional): Embedding tensor for the drug node. Defaults to a random tensor.

        Returns:
            dgl.DGLGraph: The modified graph with new edges and nodes.
    """
    g = add_new_edge_type(g, [('a', 'ad', 'd'), ('d', 'rad', 'a'),
                              ('fr', 'fd', 'd'), ('d', 'rfd', 'fr')])

    drug_node_idx = torch.tensor([g.number_of_nodes('d')], dtype=torch.int64)
    # Connect all atoms and fragments to the drug node
    for ntype in ['a', 'fr']:
        nodes = g.nodes(ntype).numpy()
        if len(nodes) > 0:
            g.add_edges(nodes, drug_node_idx.expand(len(nodes)), etype=ntype[0] + 'd')
            g.add_edges(drug_node_idx.expand(len(nodes)), nodes, etype='r' + ntype[0] + 'd')

    g.nodes['d'].data['x'] = torch.FloatTensor(drug_emb)

    g.edges['ad'].data['x'] = ad_feat.repeat(g.num_edges("ad"), 1)
    g.edges['rad'].data['x'] = rad_feat.repeat(g.num_edges("rad"), 1)

    g.edges['fd'].data['x'] = fd_feat.repeat(g.num_edges("fd"), 1)
    g.edges['rfd'].data['x'] = rfd_feat.repeat(g.num_edges("rfd"), 1)
    return g


def create_dataloader(file,
                      shuffle=True,
                      train=True):
    dataset = MolViewSet(pd.read_csv(file))
    if train:
        batch_size = 1
    else:
        batch_size = min(4200, len(dataset))

    dataloader = GraphDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


class MolViewSet(Dataset):
    def __init__(self,
                 df,
                 log=print):
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


def get_HMG_data():
    seed = 42
    train, test = random_split('../data/DDIDataset/DrugSetSmliesVector.txt', '../../data/', seed=seed)
    trainloader = create_dataloader("../data/train.csv", shuffle=True)
    testloader = create_dataloader("../data/test.csv", shuffle=False, train=False)
    return trainloader, testloader


def random_split(load_path,
                 save_dir,
                 sizes=[0.9, 0.1],
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


def find_original_index(arr,
                        a):
    cumulative_sum = 0
    for i, count in enumerate(arr):

        previous_sum = cumulative_sum
        cumulative_sum += count

        if a < cumulative_sum:
            index_in_subarray = a - previous_sum
            return i, index_in_subarray
    return -1, -1


def input_info(he):
    node_dict = {ntype: i for i, ntype in enumerate(he.ntypes)}
    edge_dict = {etype: i for i, etype in enumerate(he.etypes)}
    in_dims = {ntype: he.nodes[ntype].data['x'].shape[1] for ntype in node_dict}
    in_dims.update({etype: he.edges[etype].data['x'].shape[1] for etype in edge_dict})
    return node_dict, edge_dict, in_dims


if __name__ == '__main__':
    g = nx.Graph()
    import os
    print(os.getcwd())

    with open("../../data/ekg.csv", 'r') as f:
        for line in f:
            try:
                arr = line.strip().split('\t')
                g.add_edge(arr[0], arr[2], r=arr[1])
            except Exception as e:
                print(line)

    elementKG = ElementKG(g)
    # he1 = Mol2HeteroGraph("C1CC1", False)
    # print(he1)
    #
    # he1 = hmg_to_drug(he1)
    # print(he1)

    he1 = Mol2HeteroGraph("CC1=C(C)C=C2N(C[C@H](O)[C@H](O)[C@H](O)CO)C3=NC(=O)NC(=O)C3=NC2=C1", True,
                          elementKG=elementKG)
    print(he1)
