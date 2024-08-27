import warnings
import random

import networkx as nx
import tqdm
from typing import List
from data.DataModules import DataModule
from data.augment import Mol2HeteroGraph, input_info, ElementKG
from layers.Predictor import Predictor
from model.HMGEncoder import HMGEncoder

warnings.filterwarnings('ignore')

import numpy as np
import torch.nn as nn
from typing import List
from argparse import ArgumentParser, Namespace
import torch
from dgllife.utils import EarlyStopping, Meter
from torch.optim import Adam
import logging

logger = logging.getLogger()


class Reproduce(object):
    def __init__(self, args, data):
        self.args = args
        self.device = torch.device(f"cuda:{args['gpu']}" if torch.cuda.is_available() else "cpu")
        self.data = data
        args['device'] = self.device
        self.args = args

        self.elementKG = None
        self.get_kg()

        if self.args['data_name'] in ['FreeSolv', 'Lipophilicity', 'ESOL']:
            self.criterion = nn.MSELoss(reduction='none')
        elif self.args['data_name'] in ['BACE', 'BBBP', 'ClinTox', 'SIDER', 'ToxCast', 'HIV', 'Tox21', 'MUV', 'PCBA']:
            self.criterion = nn.BCEWithLogitsLoss(reduction='none')

        self.encoder = self.get_model()
        self.encoder.load_state_dict(torch.load(args['encoder_path'], map_location=self.device))
        self.encoder.reset_out()

        self.predictor = Predictor(self.args["mol_dim"], data.task_num, self.args["predictor_dropout"]).to(self.device)

        if args['eval'] == 'freeze':
            self.optimizer = Adam(self.predictor.parameters(), lr=self.args['lr'])
        else:
            self.optimizer = Adam([{"params": self.predictor.parameters()}, {"params": self.encoder.parameters()}],
                                  lr=self.args['lr'])

    def get_kg(self):
        g = nx.Graph()
        with open(self.args["element_KG"], 'r') as f:
            for line in f:
                try:
                    arr = line.strip().split('\t')
                    g.add_edge(arr[0], arr[2], r=arr[1])
                except Exception as e:
                    print("error line: ", line, e)
        if self.args['element_embedding'] is not None:
            self.elementKG = ElementKG(g, emb=np.load(self.args['element_embedding']))
        else:
            self.elementKG = ElementKG(g)

    def get_model(self):
        hidden_feats = self.args["hidden_feats"]
        out_feats = self.args["mol_dim"]
        n_heads = self.args["num_head"]
        n_layers = self.args["num_step_message_passing"]

        example_smi = "CC1=C(C)C=C2N(C[C@H](O)[C@H](O)[C@H](O)CO)C3=NC(=O)NC(=O)C3=NC2=C1"
        if args['encoder_name'] == 'ele':
            he = Mol2HeteroGraph(example_smi, True, elementKG=self.elementKG)
        else:
            he = Mol2HeteroGraph(example_smi, False)
        node_dict, edge_dict, in_dims = input_info(he)
        model = HMGEncoder(in_dims, hidden_feats, out_feats, n_heads, n_layers, node_dict, edge_dict)

        return model.to(self.args["device"])

    def run_train_epoch(self, dataloader):
        self.encoder.eval()
        self.predictor.train()
        total_loss = 0
        for batch_id, batch_data in tqdm.tqdm(enumerate(dataloader), ):
            smiles, bg, labels, masks = batch_data
            if len(smiles) == 1:
                continue
            bg, labels, masks = bg.to(self.device), labels.to(self.device), masks.to(self.device)

            with torch.no_grad():
                graph_embedding = self.encoder(bg)
            logits = self.predictor(graph_embedding)
            loss = (self.criterion(logits, labels) * (masks != 0).float()).mean()
            total_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        logger.info(f'train loss: {total_loss}')

    def run_eval_epoch(self, dataloader):
        self.encoder.eval()
        self.predictor.eval()
        eval_meter = Meter()
        with torch.no_grad():
            for batch_id, batch_data in enumerate(dataloader):
                smiles, bg, labels, masks = batch_data
                bg, labels = bg.to(self.device), labels.to(self.device)
                graph_embedding = self.encoder(bg)
                logits = self.predictor(graph_embedding)
                eval_meter.update(logits, labels, masks)
        return np.mean(eval_meter.compute_metric(self.data.matrix))

    def run(self):
        # self.run_analyse(self.data.train_dataloader(), f'{self.args["data_name"]}-{self.args["split_type"]}-nodemask-train.png')
        stopper = EarlyStopping(patience=args['patience'], metric=data.matrix)
        for epoch_idx in range(self.args['epoch_num']):
            self.run_train_epoch(self.data.train_dataloader())
            val_score = self.run_eval_epoch(data.val_dataloader())
            test_score = self.run_eval_epoch(self.data.test_dataloader())
            logger.info(f'val score: {val_score} | test score: {test_score}')
            early_stop = stopper.step(val_score, self.predictor)
            if early_stop:
                break
        stopper.load_checkpoint(self.predictor)
        val_score = self.run_eval_epoch(data.val_dataloader())
        test_score = self.run_eval_epoch(data.test_dataloader())

        logger.info(f'val_score: {val_score} | test_score: {test_score}')


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--data_name', type=str, default='SIDER')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epoch_num', type=int, default=500)
    parser.add_argument('--hidden_feats', type=int, default=256)
    parser.add_argument('--num_step_message_passing', type=int, default=6)
    parser.add_argument('--num_head', type=int, default=8)
    parser.add_argument('--mol_dim', type=int, default=128)
    parser.add_argument('--element_embedding', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--split_type', type=str, default='random')
    parser.add_argument('--split_ratio', type=List, default=[0.8, 0.1, 0.1])
    parser.add_argument('--encoder_name', type=str, default='mol')
    parser.add_argument('--encoder_path', type=str, default=None)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--eval', type=str, default='freeze')
    parser.add_argument('--lr', type=float, default=0.01)
    # predictor
    parser.add_argument('--predictor_dropout', type=float, default=0.0)

    return parser.parse_args().__dict__


if __name__ == '__main__':
    args = get_args()
    torch.manual_seed(args["seed"])
    random.seed(args["seed"])  # 设置 Python 内置随机库的种子
    np.random.seed(args["seed"])  # 设置 NumPy 随机库的种子
    torch.manual_seed(args["seed"])  # 设置 PyTorch 随机库的种子
    torch.cuda.manual_seed(args["seed"])  # 为当前 CUDA 设备设置种子
    torch.cuda.manual_seed_all(args["seed"])  # 为所有 CUDA 设备设置种子

    data = DataModule(args['encoder_name'], args['data_name'], args['num_workers'], args['split_type'],
                      args['split_ratio'], args['batch_size'], args['seed'])

    reproducer = Reproduce(args, data)
    reproducer.run()
