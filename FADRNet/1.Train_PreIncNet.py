import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import warnings
import torch
import monai
import pandas as pd
import numpy as np

from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

# Project-specific modules
from model._PreIncrNet import PreIncrNet
import config
from dataLoder.incremental_datasets import Dataset_Train, Dataset_Test


# -----------------------------
# Metrics & helpers
# -----------------------------
def clf_metrics(predictions, targets, average='macro'):
    """
    Compute standard classification metrics and confusion matrix.
    """
    cm = confusion_matrix(targets, predictions).astype(int)
    f1 = f1_score(targets, predictions, average=average)
    precision = precision_score(targets, predictions, average=average)
    recall = recall_score(targets, predictions, average=average)
    acc = accuracy_score(targets, predictions)
    return acc, f1, precision, recall, cm


class AverageMeter:
    """
    Tracks the running average of a scalar (e.g., loss).
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def to_device(x, gpu=False):
    """
    Move tensor to CUDA (non_blocking) if gpu=True, else to CPU.
    """
    return x.cuda(non_blocking=True) if gpu else x.cpu()


def get_learning_rate(optimizer):
    """
    Return current LR from the first param group.
    """
    if optimizer.param_groups:
        return optimizer.param_groups[0]['lr']
    raise ValueError('No trainable parameters.')


def save_model(model, cfg):
    """
    Save model weights (state_dict) as .pth.
    NOTE: Filename summarizes F1 and global step; training behavior unchanged.
    """
    f1_new = cfg['Test_F1'] * 100
    name = f"{cfg['name']}_F1_{f1_new:.2f}_step_{cfg['global_step']}.pth"
    path = os.path.join(cfg['save_dir'], name)
    torch.save(model.state_dict(), path)
    print(f"Saved model state_dict to {path}")


# -----------------------------
# Validation / loaders
# -----------------------------
def validate(data_loader, model, split_tag, cls_name='IDH'):
    """
    Run evaluation loop on a dataloader.
    Returns: (avg_loss, acc, f1, precision, recall)
    """
    model.eval()
    ce_loss = CrossEntropyLoss(reduction='mean')
    targets_all, preds_all = [], []
    loss_meter = AverageMeter()

    for img_tensor, cls_label in data_loader:
        img_tensor = to_device(img_tensor, gpu=config.use_cuda)
        cls_label = to_device(cls_label, gpu=config.use_cuda)

        with torch.no_grad():
            logits = model.predictcls(img_tensor)
            loss = ce_loss(logits, cls_label)
            loss_meter.update(loss.item(), n=1)

            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            labels_np = cls_label.cpu().numpy()

            preds_all.extend(preds)
            targets_all.extend(labels_np)

    preds_all = np.asarray(preds_all, dtype=np.int32)
    targets_all = np.asarray(targets_all, dtype=np.int32)
    acc, f1, precision, recall, cm = clf_metrics(predictions=preds_all, targets=targets_all, average="macro")

    print(cm)
    if split_tag == 'Test':
        print(f'Test set {cls_name} | Accuracy {acc:.4f} | F1 {f1:.4f} | Precision {precision:.4f} | Recall {recall:.4f} |')
    else:
        print(f'Train set {cls_name} | Accuracy {acc:.4f} | F1 {f1:.4f} | Precision {precision:.4f} | Recall {recall:.4f} |')

    model.train()
    return loss_meter.avg, acc, f1, precision, recall


def valid_metrics_by_cls(model, data_loader_Test, cls_name):
    """
    Wrapper to evaluate and return [loss, acc, f1, precision, recall].
    """
    return list(validate(data_loader=data_loader_Test, model=model, split_tag='Test', cls_name=cls_name))


def get_cls_loder(val_transforms, basic_task):
    """
    Build training and validation loaders for a classification task.
    (Function name retained to avoid changing call sites.)
    """
    train_ds = Dataset_Train(transform=val_transforms, task=basic_task, read_type='train')
    train_loader = DataLoader(train_ds,
                              batch_size=config.batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=config.n_threads,
                              pin_memory=config.use_cuda)
    print(f"Classification | Number of train samples for {basic_task}: {len(train_ds)}")

    val_ds = Dataset_Test(transform=val_transforms, task=basic_task, read_type='test')
    val_loader = DataLoader(val_ds,
                            batch_size=1,
                            shuffle=False,
                            drop_last=False,
                            num_workers=config.n_threads,
                            pin_memory=config.use_cuda)
    print(f"Classification | Number of val samples for {basic_task}: {len(val_ds)}")
    return train_loader, val_loader


def f1_from_preds(preds, gts):
    """
    Convenience: compute macro F1 from lists/arrays of predictions and ground-truth labels.
    """
    preds = np.asarray(preds, dtype=np.int32)
    gts = np.asarray(gts, dtype=np.int32)
    _, f1, _, _, _ = clf_metrics(predictions=preds, targets=gts, average="macro")
    return f1

# -----------------------------
# Train loop
# -----------------------------
def train(model, basic_task, save_name):
    """
    Train model on a single basic task.
    Training behavior preserved:
      - Adam optimizer / CE loss
      - Each epoch uses 100 iterations (fixed)
      - Best model selection by F1 on validation set
    """
    val_transforms = monai.transforms.Compose([monai.transforms.ToTensor()])

    train_loader_cls, val_loader_cls = get_cls_loder(val_transforms, basic_task)

    # Minimal and consistent log schema
    log_cols = ['epoch', 'Train_loss_1', 'Train_F1_1', 'Test_loss_1', 'Test_F1_1']
    log_df = pd.DataFrame(columns=log_cols)

    if config.use_cuda:
        model.cuda()

    loss_ce = CrossEntropyLoss(reduction='mean')

    # Only optimize parameters that require grad
    optim_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(optim_params, lr=config.lr_self, weight_decay=config.weight_decay)

    global_step = 0
    best_score_save = -1.0
    path_save = os.path.join(config.data_dir, 'results', str(save_name))
    os.makedirs(path_save, exist_ok=True)

    model.train()
    iter_index_cls = 0
    iter_labeled_cls = iter(train_loader_cls)

    for epoch in range(config.epochs):
        train_loss_meter = AverageMeter()
        pred_cls, target_cls = [], []

        # Reset best comparison after save_epoch (kept as original behavior)
        if epoch == config.save_epoch:
            best_score_save = -1.0

        # Fixed number of iterations per epoch (preserved)
        for _ in range(100):
            iter_index_cls += 1
            # Recycle the iterator when reaching the end of the loader
            if iter_index_cls >= len(train_loader_cls):
                iter_labeled_cls = iter(train_loader_cls)
                iter_index_cls = 0

            # ---- Fetch a batch ----
            imgs, labels = next(iter_labeled_cls)
            imgs = to_device(imgs, gpu=config.use_cuda)
            labels = to_device(labels, gpu=config.use_cuda)

            # ---- Forward & backward ----
            loss, logits = model(imgs, labels, loss_ce)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ---- Bookkeeping ----
            train_loss_meter.update(loss.item(), n=1)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            pred_cls.extend(preds)
            target_cls.extend(labels.cpu().numpy())
            global_step += 1

        lr = get_learning_rate(optimizer)
        print(f'Global_Step {global_step} | Train Epoch: {epoch} | lr {lr:.2e} |')

        train_loss_avg = train_loss_meter.avg
        f1_train = f1_from_preds(pred_cls, target_cls)

        # ---- Validation on the same basic task ----
        # metrics: [loss, acc, f1, precision, recall]
        metrics_val = valid_metrics_by_cls(model, val_loader_cls, cls_name=basic_task)
        loss_val = metrics_val[0]
        f1_val = metrics_val[2]

        # ---- Logging ----
        row = pd.Series([epoch, train_loss_avg, f1_train, loss_val, f1_val], index=log_cols)
        log_df = pd.concat([log_df, row.to_frame().T], ignore_index=True)
        log_df.to_csv(os.path.join(path_save, 'log.csv'), index=False)

        # ---- Model selection by validation F1 ----
        compare = f1_val
        if epoch >= config.save_epoch and compare > best_score_save:
            save_cfg = {
                'name': save_name,
                'save_dir': path_save,
                'global_step': global_step,
                'Test_F1': f1_val,
            }
            save_model(model=model, cfg=save_cfg)
            best_score_save = compare

        if compare > best_score_save:
            best_score_save = compare


# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # Reproducibility
    seed = config.random_seed
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print("Available GPUs:", torch.cuda.device_count())

    basic_task_list = ['_1p19q', 'IDH', 'LHG']
    for basic_task in basic_task_list:
        model = PreIncrNet(basic_task=basic_task, in_channels=1)
        save_name = model.name
        print(save_name)
        train(model, basic_task, save_name)