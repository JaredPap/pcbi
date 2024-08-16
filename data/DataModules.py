# -*- coding: utf-8 -*-
# @Time    : 2023/11/15 上午9:51
# @Author  : Chen Mukun
# @File    : DataModules.py
# @Software: PyCharm
# @desc    :
import dgl
import dgl.backend as F
import numpy as np
import os
import torch
from dgl.data.utils import get_download_dir, download, _get_dgl_url, extract_archive
from dgl.data.utils import save_graphs, load_graphs
from augment import Mol2HeteroGraph
from dgllife.utils.io import pmap
import pandas as pd
from torch.utils.data.dataloader import DataLoader
from dgllife.utils import ScaffoldSplitter, RandomSplitter

__all__ = ['MoleculeCSVDataset']

def Mol2EleHeteroGraph(pack):
    return Mol2HeteroGraph(pack[0],
                           ele=pack[1],
                           elementKG=pack[2])

class MoleculeCSVDataset(object):

    def __init__(self,
                 df,
                 smiles_column,
                 cache_file_path,
                 task_names=None,
                 load=False,
                 log_every=1000,
                 init_mask=True,
                 n_jobs=1,
                 elementKG=None):
        self.df = df
        self.smiles = self.df[smiles_column].tolist()
        if task_names is None:
            self.task_names = self.df.columns.drop([smiles_column]).tolist()
        else:
            self.task_names = task_names
        self.n_tasks = len(self.task_names)
        self.cache_file_path = cache_file_path
        if elementKG != None:
            smiles_to_graph = Mol2EleHeteroGraph
        else:
            smiles_to_graph = Mol2HeteroGraph
        self._pre_process(smiles_to_graph,
                          load, log_every, init_mask, n_jobs)
        # Only useful for binary classification tasks
        self._task_pos_weights = None
        self.elementKG = None

    def _pre_process(self,
                     smiles_to_graph,
                     load,
                     log_every,
                     init_mask,
                     n_jobs=1):

        if os.path.exists(self.cache_file_path) and load:
            # DGLGraphs have been constructed before, reload them
            print('Loading previously saved dgl graphs...')
            self.graphs, label_dict = load_graphs(self.cache_file_path)
            self.labels = label_dict['labels']
            if init_mask:
                self.mask = label_dict['mask']
            self.valid_ids = label_dict['valid_ids'].tolist()
        else:
            print('Processing dgl graphs from scratch...')
            if n_jobs > 1:
                if self.elementKG is not None:
                    smiles = [(smi, True, self.elementKG) for smi in self.smiles]
                else:
                    smiles = self.smiles
                self.graphs = pmap(smiles_to_graph,
                                   smiles,
                                   n_jobs=n_jobs)
            else:
                if self.elementKG is not None:
                    smiles = [(smi, True, self.elementKG) for smi in self.smiles]
                else:
                    smiles = self.smiles
                self.graphs = []
                for i, s in enumerate(smiles):
                    if (i + 1) % log_every == 0:
                        print('Processing molecule {:d}/{:d}'.format(i + 1, len(self)))
                    self.graphs.append(smiles_to_graph(s))

            # Keep only valid molecules
            self.valid_ids = []
            graphs = []
            for i, g in enumerate(self.graphs):
                if g is not None:
                    self.valid_ids.append(i)
                    graphs.append(g)
            self.graphs = graphs
            _label_values = self.df[self.task_names].values
            # np.nan_to_num will also turn inf into a very large number
            self.labels = F.zerocopy_from_numpy(
                np.nan_to_num(_label_values).astype(np.float32))[self.valid_ids]
            valid_ids = torch.tensor(self.valid_ids)
            if init_mask:
                self.mask = F.zerocopy_from_numpy(
                    (~np.isnan(_label_values)).astype(np.float32))[self.valid_ids]
                save_graphs(self.cache_file_path, self.graphs,
                            labels={'labels': self.labels, 'mask': self.mask,
                                    'valid_ids': valid_ids})
            else:
                self.mask = None
                save_graphs(self.cache_file_path, self.graphs,
                            labels={'labels': self.labels, 'valid_ids': valid_ids})

        self.smiles = [self.smiles[i] for i in self.valid_ids]

    def __getitem__(self, item):

        if self.mask is not None:
            return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]
        else:
            return self.smiles[item], self.graphs[item], self.labels[item]

    def __len__(self):
        """Size for the dataset

        Returns
        -------
        int
            Size for the dataset
        """
        return len(self.smiles)

    def task_pos_weights(self, indices):
        """Get weights for positive samples on each task

        This should only be used when all tasks are binary classification.

        Returns
        -------
        Tensor of dtype float32 and shape (T)
            Weight of positive samples on all tasks
        """
        task_pos_weights = torch.ones(self.labels.shape[1])
        num_pos = F.sum(self.labels[indices], dim=0)
        num_indices = F.sum(self.mask[indices], dim=0)
        task_pos_weights[num_pos > 0] = ((num_indices - num_pos) / num_pos)[num_pos > 0]

        return task_pos_weights


class BBBP(MoleculeCSVDataset):

    def __init__(self,
                 load=False,
                 log_every=1000,
                 cache_file_path='./bbbp_dglgraph.bin',
                 n_jobs=1,
                 elementKG=None):

        self._url = 'dataset/bbbp.zip'
        data_path = get_download_dir() + '/bbbp.zip'
        dir_path = get_download_dir() + '/bbbp'
        download(_get_dgl_url(self._url), path=data_path, overwrite=False)
        extract_archive(data_path, dir_path)
        df = pd.read_csv(dir_path + '/BBBP.csv')

        super(BBBP, self).__init__(df=df,
                                   smiles_column='smiles',
                                   cache_file_path=cache_file_path,
                                   task_names=['p_np'],
                                   load=load,
                                   log_every=log_every,
                                   init_mask=True,
                                   n_jobs=n_jobs,
                                   elementKG=elementKG)

        self.load_full = False
        self.names = df['name'].tolist()
        self.names = [self.names[i] for i in self.valid_ids]

    def __getitem__(self, item):
        if self.load_full:
            return self.smiles[item], self.graphs[item], self.labels[item], \
                self.mask[item], self.names[item]
        else:
            return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]


class BACE(MoleculeCSVDataset):
    def __init__(self,
                 load=False,
                 log_every=1000,
                 cache_file_path='./bace_dglgraph.bin',
                 n_jobs=1,
                 elementKG=None):

        self._url = 'dataset/bace.zip'
        data_path = get_download_dir() + '/bace.zip'
        dir_path = get_download_dir() + '/bace'
        download(_get_dgl_url(self._url), path=data_path, overwrite=False)
        extract_archive(data_path, dir_path)
        df = pd.read_csv(dir_path + '/bace.csv')

        super(BACE, self).__init__(df=df,
                                   smiles_column='mol',
                                   cache_file_path=cache_file_path,
                                   task_names=['Class'],
                                   load=load,
                                   log_every=log_every,
                                   init_mask=True,
                                   n_jobs=n_jobs,
                                   elementKG=elementKG)

        self.load_full = False
        self.ids = df['CID'].tolist()
        self.ids = [self.ids[i] for i in self.valid_ids]

    def __getitem__(self, item):
        if self.load_full:
            return self.smiles[item], self.graphs[item], self.labels[item], \
                self.mask[item], self.ids[item]
        else:
            return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]


class MUV(MoleculeCSVDataset):
    def __init__(self,
                 load=False,
                 log_every=1000,
                 cache_file_path='./muv_dglgraph.bin',
                 n_jobs=20,
                 elementKG=None):

        self._url = 'dataset/muv.zip'
        data_path = get_download_dir() + '/muv.zip'
        dir_path = get_download_dir() + '/muv'
        download(_get_dgl_url(self._url), path=data_path, overwrite=False)
        extract_archive(data_path, dir_path)
        df = pd.read_csv(dir_path + '/muv.csv')

        self.ids = df['mol_id'].tolist()
        self.load_full = False

        df = df.drop(columns=['mol_id'])

        super(MUV, self).__init__(df=df,
                                  smiles_column='smiles',
                                  cache_file_path=cache_file_path,
                                  load=load,
                                  log_every=log_every,
                                  init_mask=True,
                                  n_jobs=n_jobs,
                                  elementKG=elementKG)

        self.ids = [self.ids[i] for i in self.valid_ids]

    def __getitem__(self, item):
        if self.load_full:
            return self.smiles[item], self.graphs[item], self.labels[item], \
                self.mask[item], self.ids[item]
        else:
            return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]


class ClinTox(MoleculeCSVDataset):
    def __init__(self,
                 load=False,
                 log_every=1000,
                 cache_file_path='./clintox_dglgraph.bin',
                 n_jobs=1,
                 elementKG=None):
        self._url = 'dataset/clintox.zip'
        data_path = get_download_dir() + '/clintox.zip'
        dir_path = get_download_dir() + '/clintox'
        download(_get_dgl_url(self._url), path=data_path, overwrite=False)
        extract_archive(data_path, dir_path)
        df = pd.read_csv(dir_path + '/clintox.csv')

        super(ClinTox, self).__init__(df=df,
                                      smiles_column='smiles',
                                      cache_file_path=cache_file_path,
                                      load=load,
                                      log_every=log_every,
                                      init_mask=True,
                                      n_jobs=n_jobs,
                                      elementKG=elementKG)

    def __getitem__(self, item):
        return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]


class SIDER(MoleculeCSVDataset):
    def __init__(self,
                 load=False,
                 log_every=1000,
                 cache_file_path='./sider_dglgraph.bin',
                 n_jobs=1,
                 elementKG=None):
        self._url = 'dataset/sider.zip'
        data_path = get_download_dir() + '/sider.zip'
        dir_path = get_download_dir() + '/sider'
        download(_get_dgl_url(self._url), path=data_path, overwrite=False)
        extract_archive(data_path, dir_path)
        df = pd.read_csv(dir_path + '/sider.csv')

        super(SIDER, self).__init__(df=df,
                                    smiles_column='smiles',
                                    cache_file_path=cache_file_path,
                                    load=load,
                                    log_every=log_every,
                                    init_mask=True,
                                    n_jobs=n_jobs,
                                    elementKG=elementKG)

    def __getitem__(self, item):
        return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]


class ToxCast(MoleculeCSVDataset):
    def __init__(self,
                 load=False,
                 log_every=1000,
                 cache_file_path='./toxcast_dglgraph.bin',
                 n_jobs=1,
                 elementKG=None):
        self._url = 'dataset/toxcast.zip'
        data_path = get_download_dir() + '/toxcast.zip'
        dir_path = get_download_dir() + '/toxcast'
        download(_get_dgl_url(self._url), path=data_path, overwrite=False)
        extract_archive(data_path, dir_path)
        df = pd.read_csv(dir_path + '/toxcast_data.csv')

        super(ToxCast, self).__init__(df=df,
                                      smiles_column='smiles',
                                      cache_file_path=cache_file_path,
                                      load=load,
                                      log_every=log_every,
                                      init_mask=True,
                                      n_jobs=n_jobs,
                                      elementKG=elementKG)

    def __getitem__(self, item):
        return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]


class HIV(MoleculeCSVDataset):
    def __init__(self,
                 load=False,
                 log_every=1000,
                 cache_file_path='./hiv_dglgraph.bin',
                 n_jobs=1,
                 elementKG=None):

        self._url = 'dataset/hiv.zip'
        data_path = get_download_dir() + '/hiv.zip'
        dir_path = get_download_dir() + '/hiv'
        download(_get_dgl_url(self._url), path=data_path, overwrite=False)
        extract_archive(data_path, dir_path)
        df = pd.read_csv(dir_path + '/HIV.csv')

        self.activity = df['activity'].tolist()
        self.load_full = False

        df = df.drop(columns=['activity'])

        super(HIV, self).__init__(df=df,
                                  smiles_column='smiles',
                                  cache_file_path=cache_file_path,
                                  load=load,
                                  log_every=log_every,
                                  init_mask=True,
                                  n_jobs=n_jobs,
                                  elementKG=elementKG)

        self.activity = [self.activity[i] for i in self.valid_ids]

    def __getitem__(self, item):
        if self.load_full:
            return self.smiles[item], self.graphs[item], self.labels[item], \
                self.mask[item], self.activity[item]
        else:
            return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]


class Tox21(MoleculeCSVDataset):
    def __init__(self,
                 load=False,
                 log_every=1000,
                 cache_file_path='./tox21_dglgraph.bin',
                 n_jobs=1,
                 elementKG=None):
        self._url = 'dataset/tox21.csv.gz'
        data_path = get_download_dir() + '/tox21.csv.gz'
        download(_get_dgl_url(self._url), path=data_path, overwrite=False)
        df = pd.read_csv(data_path)
        self.id = df['mol_id']

        df = df.drop(columns=['mol_id'])

        self.load_full = False

        super(Tox21, self).__init__(df,
                                    smiles_column='smiles',
                                    cache_file_path=cache_file_path,
                                    load=load,
                                    log_every=log_every,
                                    n_jobs=n_jobs,
                                    elementKG=elementKG)

        self.id = [self.id[i] for i in self.valid_ids]

    def __getitem__(self, item):
        if self.load_full:
            return self.smiles[item], self.graphs[item], self.labels[item], \
                self.mask[item], self.id[item]
        else:
            return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]


class ESOL(MoleculeCSVDataset):
    def __init__(self,
                 load=False,
                 log_every=1000,
                 cache_file_path='./esol_dglgraph.bin',
                 n_jobs=1,
                 elementKG=None):

        self._url = 'dataset/ESOL.zip'
        data_path = get_download_dir() + '/ESOL.zip'
        dir_path = get_download_dir() + '/ESOL'
        download(_get_dgl_url(self._url), path=data_path, overwrite=False)
        extract_archive(data_path, dir_path)
        df = pd.read_csv(dir_path + '/delaney-processed.csv')

        super(ESOL, self).__init__(df=df,
                                   smiles_column='smiles',
                                   cache_file_path=cache_file_path,
                                   task_names=['measured log solubility in mols per litre'],
                                   load=load,
                                   log_every=log_every,
                                   init_mask=False,
                                   n_jobs=n_jobs,
                                   elementKG=elementKG)

        self.load_full = False
        # Compound names in PubChem
        self.compound_names = df['Compound ID'].tolist()
        self.compound_names = [self.compound_names[i] for i in self.valid_ids]
        # Estimated solubility
        self.estimated_solubility = df['ESOL predicted log solubility in mols per litre'].tolist()
        self.estimated_solubility = [self.estimated_solubility[i] for i in self.valid_ids]
        # Minimum atom degree
        self.min_degree = df['Minimum Degree'].tolist()
        self.min_degree = [self.min_degree[i] for i in self.valid_ids]
        # Molecular weight
        self.mol_weight = df['Molecular Weight'].tolist()
        self.mol_weight = [self.mol_weight[i] for i in self.valid_ids]
        # Number of H-Bond Donors
        self.num_h_bond_donors = df['Number of H-Bond Donors'].tolist()
        self.num_h_bond_donors = [self.num_h_bond_donors[i] for i in self.valid_ids]
        # Number of rings
        self.num_rings = df['Number of Rings'].tolist()
        self.num_rings = [self.num_rings[i] for i in self.valid_ids]
        # Number of rotatable bonds
        self.num_rotatable_bonds = df['Number of Rotatable Bonds'].tolist()
        self.num_rotatable_bonds = [self.num_rotatable_bonds[i] for i in self.valid_ids]
        # Polar Surface Area
        self.polar_surface_area = df['Polar Surface Area'].tolist()
        self.polar_surface_area = [self.polar_surface_area[i] for i in self.valid_ids]

    def __getitem__(self, item):
        if self.load_full:
            return self.smiles[item], self.graphs[item], self.labels[item], \
                self.compound_names[item], self.estimated_solubility[item], \
                self.min_degree[item], self.mol_weight[item], \
                self.num_h_bond_donors[item], self.num_rings[item], \
                self.num_rotatable_bonds[item], self.polar_surface_area[item]
        else:
            return self.smiles[item], self.graphs[item], self.labels[item]


class FreeSolv(MoleculeCSVDataset):
    def __init__(self,
                 load=False,
                 log_every=1000,
                 cache_file_path='./freesolv_dglgraph.bin',
                 n_jobs=1,
                 elementKG=None):

        self._url = 'dataset/FreeSolv.zip'
        data_path = get_download_dir() + '/FreeSolv.zip'
        dir_path = get_download_dir() + '/FreeSolv'
        download(_get_dgl_url(self._url), path=data_path, overwrite=False)
        extract_archive(data_path, dir_path)
        df = pd.read_csv(dir_path + '/SAMPL.csv')

        super(FreeSolv, self).__init__(df=df,
                                       smiles_column='smiles',
                                       cache_file_path=cache_file_path,
                                       task_names=['expt'],
                                       load=load,
                                       log_every=log_every,
                                       init_mask=False,
                                       n_jobs=n_jobs,
                                       elementKG=elementKG)

        self.load_full = False

        # Iupac names
        self.iupac_names = df['iupac'].tolist()
        self.iupac_names = [self.iupac_names[i] for i in self.valid_ids]
        # Calculated hydration free energy
        self.calc_energy = df['calc'].tolist()
        self.calc_energy = [self.calc_energy[i] for i in self.valid_ids]

    def __getitem__(self, item):
        if self.load_full:
            return self.smiles[item], self.graphs[item], self.labels[item], \
                self.iupac_names[item], self.calc_energy[item]
        else:
            return self.smiles[item], self.graphs[item], self.labels[item]


class Lipophilicity(MoleculeCSVDataset):
    def __init__(self,
                 load=False,
                 log_every=1000,
                 cache_file_path='./lipophilicity_dglgraph.bin',
                 n_jobs=1,
                 elementKG=None):

        self._url = 'dataset/lipophilicity.zip'
        data_path = get_download_dir() + '/lipophilicity.zip'
        dir_path = get_download_dir() + '/lipophilicity'
        download(_get_dgl_url(self._url), path=data_path, overwrite=False)
        extract_archive(data_path, dir_path)
        df = pd.read_csv(dir_path + '/Lipophilicity.csv')

        super(Lipophilicity, self).__init__(df=df,
                                            smiles_column='smiles',
                                            cache_file_path=cache_file_path,
                                            task_names=['exp'],
                                            load=load,
                                            log_every=log_every,
                                            init_mask=False,
                                            n_jobs=n_jobs,
                                            elementKG=elementKG)

        self.load_full = False

        # ChEMBL ids
        self.chembl_ids = df['CMPD_CHEMBLID'].tolist()
        self.chembl_ids = [self.chembl_ids[i] for i in self.valid_ids]

    def __getitem__(self, item):

        if self.load_full:
            return self.smiles[item], self.graphs[item], self.labels[item], self.chembl_ids[item]
        else:
            return self.smiles[item], self.graphs[item], self.labels[item]


def collate_molgraphs(data):
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    labels = torch.stack(labels, dim=0)

    if len(data[0]) == 3:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)

    return smiles, bg, labels, masks


class DataModule(object):
    def __init__(self, encoder_name, data_name, num_workers: int, split_type: str, split_ratio: list, batch_size: int=256,
                 seed: int = 42, elementKG=None) -> None:
        self.data_name = data_name
        if self.data_name in ['MUV', 'BACE', 'BBBP', 'ClinTox', 'SIDER', 'ToxCast', 'HIV', 'PCBA', 'Tox21']:
            self.task_type = 'classification'
        elif self.data_name in ['FreeSolv', 'Lipophilicity', 'ESOL']:
            self.task_type = 'regression'

        if self.data_name in ['BACE', 'BBBP', 'Tox21', 'ToxCast', 'SIDER', 'ClinTox', 'HIV']:
            self.matrix = 'roc_auc_score'
        elif self.data_name in ['PCBA', 'MUV']:
            self.matrix = 'pr_auc'
        elif self.data_name in ['ESOL', 'FreeSolv', 'Lipophilicity']:
            self.matrix = 'rmse'
        self.seed = seed

        self.elementKG=elementKG
        self.num_workers = num_workers
        self.encoder_name = encoder_name
        self.dataset = self.load_dataset()
        self.split_type = split_type
        self.split_ratio = split_ratio
        self.train_set, self.val_set, self.test_set = self.split_dataset()

        self.batch_size = batch_size
        self.task_num = self.dataset.n_tasks

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, drop_last=True,
                          collate_fn=collate_molgraphs, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, collate_fn=collate_molgraphs,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, collate_fn=collate_molgraphs,
                          num_workers=self.num_workers)

    def load_dataset(self):
        if self.encoder_name == 'ele':
            if self.data_name == 'MUV':
                dataset = MUV(n_jobs=self.num_workers,elementKG=self.elementKG)
            elif self.data_name == 'BACE':
                dataset = BACE(n_jobs=self.num_workers,elementKG=self.elementKG)
            elif self.data_name == 'BBBP':
                dataset = BBBP(n_jobs=self.num_workers,elementKG=self.elementKG)
            elif self.data_name == 'ClinTox':
                dataset = ClinTox(n_jobs=self.num_workers,elementKG=self.elementKG)
            elif self.data_name == 'SIDER':
                dataset = SIDER(n_jobs=self.num_workers,elementKG=self.elementKG)
            elif self.data_name == 'ToxCast':
                dataset = ToxCast(n_jobs=self.num_workers,elementKG=self.elementKG)
            elif self.data_name == 'HIV':
                dataset = HIV(n_jobs=self.num_workers,elementKG=self.elementKG)
            elif self.data_name == 'Tox21':
                dataset = Tox21(n_jobs=self.num_workers,elementKG=self.elementKG)
            elif self.data_name == 'FreeSolv':
                dataset = FreeSolv(n_jobs=self.num_workers,elementKG=self.elementKG)
            elif self.data_name == 'ESOL':
                dataset = ESOL(n_jobs=self.num_workers,elementKG=self.elementKG)
            elif self.data_name == 'Lipophilicity':
                dataset = Lipophilicity(n_jobs=self.num_workers,elementKG=self.elementKG)
            else:
                raise ValueError('Unexpected dataset: {}'.format(self.data_name))
        else:
            if self.data_name == 'MUV':
                dataset = MUV(n_jobs=self.num_workers)
            elif self.data_name == 'BACE':
                dataset = BACE(n_jobs=self.num_workers)
            elif self.data_name == 'BBBP':
                dataset = BBBP(n_jobs=self.num_workers)
            elif self.data_name == 'ClinTox':
                dataset = ClinTox(n_jobs=self.num_workers)
            elif self.data_name == 'SIDER':
                dataset = SIDER(n_jobs=self.num_workers)
            elif self.data_name == 'ToxCast':
                dataset = ToxCast(n_jobs=self.num_workers)
            elif self.data_name == 'HIV':
                dataset = HIV(n_jobs=self.num_workers)
            elif self.data_name == 'Tox21':
                dataset = Tox21(n_jobs=self.num_workers)
            elif self.data_name == 'FreeSolv':
                dataset = FreeSolv(n_jobs=self.num_workers)
            elif self.data_name == 'ESOL':
                dataset = ESOL(n_jobs=self.num_workers)
            elif self.data_name == 'Lipophilicity':
                dataset = Lipophilicity(n_jobs=self.num_workers)
            else:
                raise ValueError('Unexpected dataset: {}'.format(self.data_name))
        return dataset

    def split_dataset(self):
        train_ratio, val_ratio, test_ratio = self.split_ratio
        if self.split_type == 'scaffold':
            train_set, val_set, test_set = ScaffoldSplitter.train_val_test_split(
                self.dataset, frac_train=train_ratio, frac_val=val_ratio, frac_test=test_ratio,
                scaffold_func='smiles')
        elif self.split_type == 'random':
            train_set, val_set, test_set = RandomSplitter.train_val_test_split(
                self.dataset, frac_train=train_ratio, frac_val=val_ratio, frac_test=test_ratio, random_state=self.seed)
        else:
            return ValueError("Expect the splitting method to be '', got {}".format(self.split_type))

        return train_set, val_set, test_set
