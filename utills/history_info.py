import os
import json


# 학습과정 모든 정보 저장 클래스
class HistoryManager:
    def __init__(self):
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "train_mf1": [],
            "train_miou": [],
            "train_oa": [],
            "val_loss": [],
            "val_acc": [],
            "val_mf1": [],
            "val_miou": [],
            "val_oa": [],
        }

    def update_history(self,
                       train_loss, train_cls_acc, train_seg_mf1, train_seg_miou, train_seg_oa,
                       val_loss, val_cls_acc, val_seg_mf1, val_seg_miou, val_seg_oa):
        self.history["train_loss"].append(float(train_loss))
        self.history["train_acc"].append(float(train_cls_acc))
        self.history["train_mf1"].append(float(train_seg_mf1))
        self.history["train_miou"].append(float(train_seg_miou))
        self.history["train_oa"].append(float(train_seg_oa))

        self.history["val_loss"].append(float(val_loss))
        self.history["val_acc"].append(float(val_cls_acc))
        self.history["val_mf1"].append(float(val_seg_mf1))
        self.history["val_miou"].append(float(val_seg_miou))
        self.history["val_oa"].append(float(val_seg_oa))

    def get(self):
        return self.history



def history_save(history, path = './checkpoint/history'):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)