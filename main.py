"""
CD-BAN main.py — training entry point.

Usage:
  python main.py --cfg configs/CDBAN.yaml
  python main.py --cfg configs/CDBAN.yaml --seed 42 --out results/seed_42
"""
import warnings, os, argparse
import torch
import pandas as pd
import yaml
from time import time
from torch.utils.data import DataLoader

from models     import CDBAN
from trainer    import Trainer
from dataloader import CDBinaryDataset, cd_collate_func
from utils      import set_seed, mkdir

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--cfg',  required=True, type=str)
parser.add_argument('--seed', default=None,  type=int, help='override YAML seed')
parser.add_argument('--out',  default=None,  type=str, help='override output directory')
args = parser.parse_args()


def main():
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg["SOLVER"]["SEED"] = args.seed
    if args.out is not None:
        cfg["RESULT"]["OUTPUT_DIR"] = args.out

    set_seed(cfg["SOLVER"]["SEED"])
    mkdir(cfg["RESULT"]["OUTPUT_DIR"])
    print(f"Seed   : {cfg['SOLVER']['SEED']}")
    print(f"Config : {args.cfg}")
    print(f"Device : {device}\n")

    data_dir = cfg["DATASET"]["DATA_PATH"]
    df_train = pd.read_csv(os.path.join(data_dir, cfg["DATASET"]["TRAIN"]))
    df_val   = pd.read_csv(os.path.join(data_dir, cfg["DATASET"]["VAL"]))
    df_test  = pd.read_csv(os.path.join(data_dir, cfg["DATASET"]["TEST"]))

    train_ds = CDBinaryDataset(df_train.index.values, df_train)
    val_ds   = CDBinaryDataset(df_val.index.values,   df_val)
    test_ds  = CDBinaryDataset(df_test.index.values,  df_test)

    params_train = dict(batch_size=cfg["SOLVER"]["BATCH_SIZE"],
                        shuffle=True, drop_last=True,
                        num_workers=cfg["SOLVER"]["NUM_WORKERS"],
                        collate_fn=cd_collate_func)
    params_eval  = dict(batch_size=cfg["SOLVER"]["BATCH_SIZE"],
                        shuffle=False, drop_last=False,
                        num_workers=cfg["SOLVER"]["NUM_WORKERS"],
                        collate_fn=cd_collate_func)

    train_loader = DataLoader(train_ds, **params_train)
    val_loader   = DataLoader(val_ds,  **params_eval)
    test_loader  = DataLoader(test_ds, **params_eval)

    model = CDBAN(**cfg).to(device)
    print(model)
    print(f"\nParams: {sum(p.numel() for p in model.parameters()):,}\n")

    optim = torch.optim.Adam(model.parameters(), lr=cfg["SOLVER"]["LR"])

    trainer = Trainer(model, optim, device,
                      train_loader, val_loader, test_loader, **cfg)
    result = trainer.train()

    with open(os.path.join(cfg["RESULT"]["OUTPUT_DIR"], "model_arch.txt"), "w") as f:
        f.write(str(model))

    print(f"\nOutput directory: {cfg['RESULT']['OUTPUT_DIR']}")
    return result


if __name__ == '__main__':
    t0 = time()
    result = main()
    print(f"\nTotal time: {round(time()-t0,1)}s")
