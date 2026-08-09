"""
CD-BAN Trainer — DrugBAN trainer adapted: domain adaptation removed, pos_weight added.
"""
import torch
import torch.nn as nn
import copy
import os
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              roc_curve, confusion_matrix,
                              precision_recall_curve, precision_score, f1_score)
from models import binary_cross_entropy
from prettytable import PrettyTable
from tqdm import tqdm


class Trainer:
    def __init__(self, model, optim, device,
                 train_loader, val_loader, test_loader,
                 **config):
        self.model        = model
        self.optim        = optim
        self.device       = device
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.epochs       = config["SOLVER"]["MAX_EPOCH"]
        self.n_class      = config["DECODER"]["BINARY"]
        self.pos_weight   = config.get("CLASSIFIER", {}).get("POS_WEIGHT", None)
        self.output_dir   = config["RESULT"]["OUTPUT_DIR"]
        self.config       = config

        self.best_model   = None
        self.best_epoch   = 0
        self.best_auroc   = 0.0
        self.current_epoch = 0

        self.train_loss_epoch = []
        self.val_loss_epoch   = []
        self.val_auroc_epoch  = []
        self.test_metrics     = {}

        self.val_table  = PrettyTable(["Epoch", "AUROC", "AUPRC", "Val_loss"])
        self.test_table = PrettyTable(["Best_Epoch", "AUROC", "AUPRC",
                                       "F1", "Sensitivity", "Specificity",
                                       "Accuracy", "Threshold", "Test_loss"])
        self.train_table = PrettyTable(["Epoch", "Train_loss"])

    def train(self):
        for epoch in range(self.epochs):
            self.current_epoch = epoch + 1
            train_loss = self._train_epoch()
            self.train_loss_epoch.append(train_loss)
            self.train_table.add_row([f"epoch {self.current_epoch}", f"{train_loss:.4f}"])

            auroc, auprc, val_loss = self._eval("val")
            self.val_loss_epoch.append(val_loss)
            self.val_auroc_epoch.append(auroc)
            self.val_table.add_row([f"epoch {self.current_epoch}",
                                    f"{auroc:.4f}", f"{auprc:.4f}", f"{val_loss:.4f}"])
            print(f"[Epoch {self.current_epoch:3d}]  train_loss={train_loss:.4f}"
                  f"  val_loss={val_loss:.4f}  AUROC={auroc:.4f}  AUPRC={auprc:.4f}")

            if auroc >= self.best_auroc:
                self.best_auroc  = auroc
                self.best_epoch  = self.current_epoch
                self.best_model  = copy.deepcopy(self.model)

        result = self._eval("test")
        auroc, auprc, f1, sens, spec, acc, test_loss, thr, prec = result
        self.test_table.add_row([f"epoch {self.best_epoch}",
                                  f"{auroc:.4f}", f"{auprc:.4f}", f"{f1:.4f}",
                                  f"{sens:.4f}", f"{spec:.4f}", f"{acc:.4f}",
                                  f"{thr:.4f}", f"{test_loss:.4f}"])
        print(f"\n[Test @ Best Epoch {self.best_epoch}]  AUROC={auroc:.4f}  AUPRC={auprc:.4f}"
              f"  F1={f1:.4f}  Sens={sens:.4f}  Spec={spec:.4f}  Acc={acc:.4f}")

        self.test_metrics = {
            "auroc": auroc, "auprc": auprc, "f1": f1,
            "sensitivity": sens, "specificity": spec, "accuracy": acc,
            "threshold": thr, "test_loss": test_loss, "best_epoch": self.best_epoch,
        }
        self._save()
        return self.test_metrics

    def _train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for v_g, v_h, labels in tqdm(self.train_loader, leave=False):
            v_g    = v_g.to(self.device)
            v_h    = v_h.to(self.device)
            labels = labels.float().to(self.device)
            self.optim.zero_grad()
            _, _, _, score = self.model(v_g, v_h, mode="train")
            _, loss = binary_cross_entropy(score, labels, self.pos_weight)
            loss.backward()
            self.optim.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def _eval(self, split="val"):
        loader = self.val_loader if split == "val" else self.test_loader
        mdl    = self.model if split == "val" else self.best_model
        mdl.eval()
        y_true, y_pred = [], []
        total_loss = 0.0
        with torch.no_grad():
            for v_g, v_h, labels in loader:
                v_g    = v_g.to(self.device)
                v_h    = v_h.to(self.device)
                labels = labels.float().to(self.device)
                _, _, score, _ = mdl(v_g, v_h, mode="eval")
                n, loss = binary_cross_entropy(score, labels, self.pos_weight)
                total_loss += loss.item()
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(n.cpu().tolist())

        auroc = roc_auc_score(y_true, y_pred)
        auprc = average_precision_score(y_true, y_pred)
        avg_loss = total_loss / len(loader)

        if split == "val":
            return auroc, auprc, avg_loss

        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_pred)
        precision_v = tpr / (tpr + fpr + 1e-8)
        f1_v = 2 * precision_v * tpr / (precision_v + tpr + 1e-8)
        thr_opt = thresholds[5:][np.argmax(f1_v[5:])] if len(f1_v) > 5 else 0.5

        y_pred_bin = [1 if p >= thr_opt else 0 for p in y_pred]
        cm = confusion_matrix(y_true, y_pred_bin)
        acc  = (cm[0,0] + cm[1,1]) / cm.sum()
        sens = cm[0,0] / (cm[0,0] + cm[0,1] + 1e-8)
        spec = cm[1,1] / (cm[1,0] + cm[1,1] + 1e-8)
        f1   = f1_score(y_true, y_pred_bin, zero_division=0)
        prec = precision_score(y_true, y_pred_bin, zero_division=0)
        return auroc, auprc, f1, sens, spec, acc, avg_loss, thr_opt, prec

    def _save(self):
        os.makedirs(self.output_dir, exist_ok=True)
        if self.config["RESULT"]["SAVE_MODEL"]:
            torch.save(self.best_model.state_dict(),
                       os.path.join(self.output_dir, f"best_model_epoch_{self.best_epoch}.pth"))
            torch.save(self.model.state_dict(),
                       os.path.join(self.output_dir, f"model_epoch_{self.current_epoch}.pth"))
        state = {
            "train_loss": self.train_loss_epoch,
            "val_loss":   self.val_loss_epoch,
            "val_auroc":  self.val_auroc_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config,
        }
        torch.save(state, os.path.join(self.output_dir, "result_metrics.pt"))
        for name, table in [("val", self.val_table),
                             ("test", self.test_table),
                             ("train", self.train_table)]:
            with open(os.path.join(self.output_dir, f"{name}_log.txt"), "w") as f:
                f.write(table.get_string())
