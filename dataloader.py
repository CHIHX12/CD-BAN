"""
CD-BAN Dataloader — torch_geometric version.
Reads SMILES_Guest, SMILES_Host, label columns; featurizer matches CDGNN.
"""
import torch
import torch.utils.data as data
from rdkit import Chem
from torch_geometric.data import Data, Batch

ATOM_LIST = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
    'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag',
    'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd',
    'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb'
]
HYBRIDIZATION = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]


def _one_hot(val, choices):
    enc = [0] * len(choices)
    if val in choices:
        enc[choices.index(val)] = 1
    return enc


def atom_features(atom):
    return (
        _one_hot(atom.GetSymbol(), ATOM_LIST)
        + _one_hot(atom.GetDegree(), list(range(11)))
        + _one_hot(atom.GetImplicitValence(), list(range(7)))
        + [atom.GetFormalCharge()]
        + [atom.GetNumRadicalElectrons()]
        + _one_hot(atom.GetHybridization(), HYBRIDIZATION)
        + [int(atom.GetIsAromatic())]
        + _one_hot(atom.GetTotalNumHs(), list(range(5)))
    )  # 74-dim


def smiles_to_pyg(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles('C')   # fallback
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges += [[i, j], [j, i]]
    edge_index = (torch.tensor(edges, dtype=torch.long).t().contiguous()
                  if edges else torch.zeros((2, 0), dtype=torch.long))
    return Data(x=x, edge_index=edge_index)


class CDBinaryDataset(data.Dataset):
    def __init__(self, list_IDs, df):
        self.list_IDs = list_IDs
        self.df       = df

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        idx = self.list_IDs[index]
        row = self.df.iloc[idx]
        g_guest = smiles_to_pyg(row['SMILES_Guest'])
        g_host  = smiles_to_pyg(row['SMILES_Host'])
        y = float(row['label'])
        return g_guest, g_host, y


def cd_collate_func(batch):
    guests, hosts, labels = zip(*batch)
    bg = Batch.from_data_list(list(guests))
    bh = Batch.from_data_list(list(hosts))
    yl = torch.tensor(labels, dtype=torch.float32)
    return bg, bh, yl
