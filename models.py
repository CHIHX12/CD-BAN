"""
CD-BAN models.py
DrugBAN-inspired architecture: two MolecularGCN (torch_geometric) + BAN + MLPDecoder → binary
Host uses GCN in place of ProteinCNN; otherwise identical to DrugBAN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn.utils.weight_norm import weight_norm

import sys, os
_BAN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
if _BAN_DIR not in sys.path:
    sys.path.append(_BAN_DIR)
from ban import BANLayer


def binary_cross_entropy(pred_output, labels, pos_weight=None):
    """BCE loss with optional pos_weight for class imbalance."""
    if pos_weight is not None:
        pw  = torch.tensor([pos_weight], device=pred_output.device)
        loss = nn.BCEWithLogitsLoss(pos_weight=pw)(pred_output.squeeze(1), labels)
        n    = torch.sigmoid(pred_output.squeeze(1))
    else:
        n    = torch.sigmoid(pred_output.squeeze(1))
        loss = nn.BCELoss()(n, labels)
    return n, loss


class CDBAN(nn.Module):
    """
    CD-BAN = DrugBAN with ProteinCNN replaced by MolecularGCN (torch_geometric).
    guest_extractor : MolecularGCN
    host_extractor  : MolecularGCN
    bcn             : BANLayer (bilinear attention, identical to DrugBAN)
    mlp_classifier  : MLPDecoder → 1 logit → sigmoid → binary
    """
    def __init__(self, **config):
        super(CDBAN, self).__init__()
        self.max_guest = config["GUEST"]["MAX_NODES"]
        self.max_host  = config["HOST"]["MAX_NODES"]

        self.guest_extractor = MolecularGCN(
            in_feats     = config["GUEST"]["NODE_IN_FEATS"],
            dim_embedding= config["GUEST"]["NODE_IN_EMBEDDING"],
            hidden_feats = config["GUEST"]["HIDDEN_LAYERS"],
        )
        self.host_extractor = MolecularGCN(
            in_feats     = config["HOST"]["NODE_IN_FEATS"],
            dim_embedding= config["HOST"]["NODE_IN_EMBEDDING"],
            hidden_feats = config["HOST"]["HIDDEN_LAYERS"],
        )
        self.bcn = weight_norm(
            BANLayer(
                v_dim = config["GUEST"]["HIDDEN_LAYERS"][-1],
                q_dim = config["HOST"]["HIDDEN_LAYERS"][-1],
                h_dim = config["DECODER"]["IN_DIM"],
                h_out = config["BCN"]["HEADS"],
            ),
            name='h_mat', dim=None,
        )
        self.mlp_classifier = MLPDecoder(
            in_dim     = config["DECODER"]["IN_DIM"],
            hidden_dim = config["DECODER"]["HIDDEN_DIM"],
            out_dim    = config["DECODER"]["OUT_DIM"],
            binary     = config["DECODER"]["BINARY"],
        )

    def forward(self, guest_data, host_data, mode="train"):
        v_g = self.guest_extractor(guest_data, self.max_guest)
        v_h = self.host_extractor(host_data,   self.max_host)
        f, att = self.bcn(v_g, v_h)
        score  = self.mlp_classifier(f)
        if mode == "train":
            return v_g, v_h, f, score
        else:
            return v_g, v_h, score, att


class MolecularGCN(nn.Module):
    """GCN encoder using torch_geometric; featurizer identical to CDGNN."""
    def __init__(self, in_feats, dim_embedding=128, hidden_feats=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        layers, in_dim = [], dim_embedding
        for out_dim in hidden_feats:
            layers.append(GCNConv(in_dim, out_dim))
            in_dim = out_dim
        self.convs        = nn.ModuleList(layers)
        self.output_feats = hidden_feats[-1]

    def forward(self, data, max_nodes):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.init_transform(x)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        batch_size = batch.max().item() + 1
        out = torch.zeros(batch_size, max_nodes, self.output_feats, device=x.device)
        for b in range(batch_size):
            mask  = (batch == b)
            nodes = x[mask]
            n     = min(nodes.size(0), max_nodes)
            out[b, :n] = nodes[:n]
        return out


class MLPDecoder(nn.Module):
    """MLP decoder copied from DrugBAN."""
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        return self.fc4(x)
