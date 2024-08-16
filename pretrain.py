import sys

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from rdkit import Chem
from data.augment import ElementKG, Mol2HeteroGraph, hmg_to_drug, input_info
import dgl
import logging
import networkx as nx
import tqdm
import pickle
from argparse import ArgumentParser

from layers.ContrastiveLoss import ContrastiveLoss
from layers.Projector import Projector
from model.HMGEncoder import HMGEncoder
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger()


class Pretrainer(object):
    def __init__(self, args) -> None:
        super().__init__()
        torch.manual_seed(args["seed"])  # Set the random seed for reproducibility
        self.device = torch.device(
            f"cuda:{args['gpu']}" if torch.cuda.is_available() else "cpu")  # Setup device for torch
        args['device'] = self.device
        self.args = args
        self.epoch_num = self.args['epoch_num']  # Number of epochs to train
        self.global_step = 0  # Global step counter

        # Load embeddings and knowledge graphs
        self.drug_emb = None  # Placeholder for drug embeddings
        self.elementKG = None  # Placeholder for element knowledge graph
        self.get_kgs()  # Initialize knowledge graphs

        # Initialize models for molecules, drugs, and elements
        self.model_mol, self.model_drug, self.model_ele = self.get_models()
        self.projector = Projector(self.args["mol_dim"]).to(self.device)

        # Setup optimizer and learning rate scheduler
        self.optimizer = Adam([
            {"params": self.model_mol.parameters()},
            {"params": self.model_drug.parameters()},
            {"params": self.model_ele.parameters()},
            {"params": self.projector.parameters()}
        ], lr=args['lr'])

        self.scheduler = ExponentialLR(self.optimizer, 0.99, -1)
        self.step_per_schedule = args['step_per_schedule']

        # Initialize the contrastive loss criterion
        self.criterion = ContrastiveLoss(self.device, args['temperature']).to(self.device)

        # Prepare batches of heterogeneous molecular graphs
        self.hmg_batches = []
        if args['hmg_dir'] is None:
            self.get_hmgs()  # Generate and store batches if not already done
            with open("hmg_batches.pickle", "wb") as file:
                pickle.dump(self.hmg_batches, file)
        else:
            with open(args['hmg_dir'] + "hmg_batches.pickle", "rb") as file:
                self.hmg_batches = pickle.load(file)

    # def get_hmgs(self):
    #     smis = []
    #     count = 0
    #     with open(self.args["pretrain_data"], 'r') as f:
    #         for line in f:
    #             try:
    #                 arr = line.strip().split(',')
    #                 smis.append((arr[0], arr[1]))
    #             except Exception as e:
    #                 print("error read line : ", line, e)
    #     batches=[]
    #     batch=[]
    #     for mol in smis:
    #         batch.append((mol,count))
    #         count += 1
    #         if count % self.args["batch_size"] == 0:
    #             batches.append(batch)
    #             batch = []
    #
    #     threads = []
    #     with ThreadPoolExecutor(max_workers=40) as executor:
    #         for batch in tqdm.tqdm(batches):
    #             batch_result=[]
    #             for mol,idx in batch:
    #                 threads.append(executor.submit(self.get_mol,mol,idx))
    #             for task in as_completed(threads):
    #                 g1, g2, g3 ,idx = task.result()
    #                 batch_result.append([(g1, g2, g3), idx])
    #             batch_result = sorted(batch_result, key=lambda result: result[-1])
    #             hmg_one_batch=[[],[],[]]
    #             for  result in batch_result:
    #                 hmg_one_batch[0].append(result[0][0].to(self.device))
    #                 if result[0][1] is not None:
    #                     hmg_one_batch[1].append(result[0][1].to(self.device))
    #                 hmg_one_batch[2].append(result[0][2].to(self.device))
    #             self.hmg_batches.append(hmg_one_batch)

    def get_hmgs(self):
        """ Generate batches of heterogeneous molecular graphs. """
        smis = []
        count = 0
        batch = [[], [], []]

        # Read SMILES and drug IDs from file
        with open(self.args["pretrain_data"], 'r') as f:
            for line in f:
                try:
                    arr = line.strip().split(',')
                    smis.append((arr[0], arr[1]))  # Extract SMILES and drug IDs
                except Exception as e:
                    print("Error reading line:", line, e)

        # Process each molecule and generate graphs
        for mol in tqdm.tqdm(smis):
            g1, g2, g3, idx = self.get_mol(mol)
            if None in (g1, g3):
                print("Error processing molecule:", mol)
                continue
            batch[0].append(g1)
            if g2 is not None:
                batch[1].append(g2)
            batch[2].append(g3)
            count += 1
            if count % self.args["batch_size"] == 0:
                self.hmg_batches.append(batch)
                batch = [[], [], []]

    def get_mol(self, mol, idx=0):
        """ Generate molecular graphs for a given molecule. """
        smi, drug_id = mol
        try:
            mol_obj = Chem.MolFromSmiles(smi)  # Convert SMILES to RDKit molecule object
            g1 = Mol2HeteroGraph(mol_obj, ele=False)
            g2 = hmg_to_drug(g1, self.drug_emb[drug_id]) if drug_id and self.drug_emb else None
            g3 = Mol2HeteroGraph(mol_obj, ele=True, elementKG=self.elementKG)
            return g1, g2, g3, idx
        except Exception as e:
            print(f"Error processing molecule {smi}: {e}")
            return None, None, None, idx

    def get_kgs(self):
        """ Load or create knowledge graphs and embeddings. """
        g = nx.Graph()
        with open(self.args["element_KG"], 'r') as f:
            for line in f:
                try:
                    arr = line.strip().split('\t')
                    g.add_edge(arr[0], arr[2], r=arr[1])
                except Exception as e:
                    print("Error line:", line, e)
        self.elementKG = ElementKG(g, emb=np.load(self.args['element_embedding'])) if self.args[
                                                                                          'element_embedding'] is not None else ElementKG(
            g)
        self.drug_emb = np.load(self.args['drug_embedding']) if self.args['drug_embedding'] is not None else None

    def get_models(self):
        """ Initialize and return models for molecules, drugs, and elements. """

        hidden_feats = self.args["hidden_feats"]
        out_feats = self.args["mol_dim"]
        n_heads = self.args["num_head"]
        n_layers = self.args["num_step_message_passing"]

        # Example molecule to initialize models
        example_smi = "CC1=C(C)C=C2N(C[C@H](O)[C@H](O)[C@H](O)CO)C3=NC(=O)NC(=O)C3=NC2=C1"

        he = Mol2HeteroGraph(example_smi, False)
        node_dict, edge_dict, in_dims = input_info(he)
        model_mol = HMGEncoder(in_dims, hidden_feats, out_feats, n_heads, n_layers, node_dict, edge_dict)

        he = hmg_to_drug(he)
        node_dict, edge_dict, in_dims = input_info(he)
        model_drug = HMGEncoder(in_dims, hidden_feats, out_feats, n_heads, n_layers, node_dict, edge_dict)

        he = Mol2HeteroGraph(example_smi, True, elementKG=self.elementKG)

        node_dict, edge_dict, in_dims = input_info(he)
        model_ele = HMGEncoder(in_dims, hidden_feats, out_feats, n_heads, n_layers, node_dict, edge_dict)

        return model_mol.to(self.args["device"]), model_drug.to(self.args["device"]), model_ele.to(self.args["device"])

    def run_train_epoch(self, epoch_idx):
        # Iterate over all batches of graphs for the current epoch
        for batch_id, batch_graph_idx in tqdm.tqdm(enumerate(self.hmg_batches), desc=f"epoch{epoch_idx:04d}"):
            # Process molecule graphs by batching them together, moving to the appropriate device, and passing through the molecule model
            mol = self.model_mol(dgl.batch(batch_graph_idx[0]).to(self.device), True)
            # Process drug graphs similarly
            drug = self.model_drug(dgl.batch(batch_graph_idx[1]).to(self.device), True)
            # Process element graphs similarly
            ele = self.model_ele(dgl.batch(batch_graph_idx[2]).to(self.device), True)

            # Project the outputs of the models for dimensionality reduction or feature learning
            mol = self.projector(mol)
            drug = self.projector(drug)
            ele = self.projector(ele)

            # Calculate contrastive loss between different view (molecule, drug, element)
            # loss = self.criterion(mol, drug)
            # loss += self.criterion(drug, mol)
            # loss += self.criterion(mol, ele)
            # loss += self.criterion(ele, mol)
            # loss += self.criterion(drug, ele)
            # loss += self.criterion(ele, drug)
            loss = self.criterion(drug, mol)
            loss += self.criterion(drug, ele)
            loss += self.criterion(mol, ele)


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.global_step += 1

            logger.info(f'[{epoch_idx:03d}/{batch_id:03d}] train loss {loss.item():.4f}')

            # save model
            if self.global_step % 3000 == 0:
                torch.save(self.model_mol.state_dict(),
                           self.args["model_dir"] + f'save_mol_{self.global_step:08d}.pt')
                torch.save(self.model_drug.state_dict(),
                           self.args["model_dir"] + f'save_drug_{self.global_step:08d}.pt')
                torch.save(self.model_ele.state_dict(),
                           self.args["model_dir"] + f'save_ele_{self.global_step:08d}.pt')
            if self.global_step % self.step_per_schedule == 0:
                self.scheduler.step()

    def run(self):
        for epoch_idx in range(self.epoch_num):
            self.run_train_epoch(epoch_idx)


def get_args():
    parser = ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--hmg_dir', type=str, default=None)
    parser.add_argument('--pretrain_data', type=str, default='./dataset/pretrain/batches.csv')
    parser.add_argument('--element_KG', type=str, default="./dataset/pretrain/ekg.csv")
    parser.add_argument('--element_embedding', type=str, default=None)
    parser.add_argument('--drug_embedding', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--hidden_feats', type=int, default=256)
    parser.add_argument('--num_step_message_passing', type=int, default=6)
    parser.add_argument('--num_head', type=int, default=8)
    parser.add_argument('--mol_dim', type=int, default=128)
    parser.add_argument('--model_dir', type=str, default="./model/")
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--step_per_schedule', type=int, default=125)
    parser.add_argument('--epoch_num', type=int, default=256)

    return parser.parse_args().__dict__


if __name__ == '__main__':
    pretrainer = Pretrainer(get_args())
    pretrainer.run()
