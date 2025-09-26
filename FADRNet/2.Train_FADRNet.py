import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import warnings
import torch
import monai
import pandas as pd
import numpy as np

from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

# Project-specific modules
from model._FADRNet import FADRNet
from model._PreIncrNet import PreIncrNet

import config
from dataLoder.incremental_datasets import Dataset_Train_frequency, Dataset_Test_frequency
from torch.utils.data import DataLoader
from collections import OrderedDict

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
    Track running average of a scalar (e.g., loss).
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
    Save model state_dict with filename summarizing F1 (new & old) and step.
    """
    f1_new = cfg['Test_F1'] * 100
    f1_old = cfg['Test_F1_old'] * 100
    name = f"{cfg['name']}_F1_{f1_new:.2f}_F1_old_{f1_old:.2f}_step_{cfg['global_step']}.pth"
    path = os.path.join(cfg['save_dir'], name)
    torch.save(model.state_dict(), path)
    print(f"Saved model state_dict to {path}")


def load_infer_weights(model, infer_ckpt_path: str):
    """
    Load reduced *_infer.pth checkpoint for inference.
    Uses strict=False to avoid errors if keys are missing by design.
    Returns the load result object for inspection.
    """
    sd = torch.load(infer_ckpt_path, map_location='cpu')
    result = model.load_state_dict(sd, strict=False)

    # Print useful info for debugging
    missing = getattr(result, 'missing_keys', [])
    unexpected = getattr(result, 'unexpected_keys', [])
    if missing:
        print(f"[Load Infer] Missing keys (expected by model but not in checkpoint): "
              f"{missing[:8]}{' ...' if len(missing)>8 else ''}")
    if unexpected:
        print(f"[Load Infer] Unexpected keys (found in checkpoint but not used by model): "
              f"{unexpected[:8]}{' ...' if len(unexpected)>8 else ''}")
    print(f"[Load Infer] Loaded from {infer_ckpt_path}")
    return result

# -----------------------------
# Validation / loaders
# -----------------------------
def validate(data_loader, model, split_tag, cls_name='IDH', old=False):
    """
    Run evaluation loop on a dataloader.
    If old=True, call model.predictcls_old; otherwise call model.predictcls.
    Returns: (avg_loss, acc, f1, precision, recall)
    """
    model.eval()
    ce_loss = CrossEntropyLoss(reduction='mean')
    targets_all, preds_all = [], []
    loss_meter = AverageMeter()

    for imgs, labels in data_loader:
        # Move inputs to device; imgs is a list of frequency-conditioned inputs
        imgs_cuda = [to_device(img, gpu=config.use_cuda) for img in imgs]
        labels = to_device(labels, gpu=config.use_cuda)

        with torch.no_grad():
            logits = model.predictcls_old(imgs_cuda) if old else model.predictcls(imgs_cuda)
            loss = ce_loss(logits, labels)
            loss_meter.update(loss.item(), n=1)

            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()

            preds_all.extend(preds)
            targets_all.extend(labels_np)

    # Metrics
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


def eval_metrics_for_class(model, test_loader, cls_name, old=False):
    """
    Wrapper to evaluate and return [loss, acc, f1, precision, recall].
    """
    return list(validate(data_loader=test_loader, model=model, split_tag='Test', cls_name=cls_name, old=old))


def build_class_loaders(transforms, task_name, diameters_list):
    """
    Build training and validation loaders for a classification task (incremental task).
    """
    train_ds = Dataset_Train_frequency(transform=transforms, task=task_name,
                                       read_type='train', diameters_list=diameters_list)
    train_loader = DataLoader(train_ds,
                              batch_size=config.batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=config.n_threads,
                              pin_memory=config.use_cuda)
    print(f"Classification | Number of train samples for {task_name}: {len(train_ds)}")

    val_ds = Dataset_Test_frequency(transform=transforms, task=task_name,
                                    read_type='test', diameters_list=diameters_list)
    val_loader = DataLoader(val_ds,
                            batch_size=1,
                            shuffle=False,
                            drop_last=False,
                            num_workers=config.n_threads,
                            pin_memory=config.use_cuda)
    print(f"Classification | Number of val samples for {task_name}: {len(val_ds)}")
    return train_loader, val_loader


def build_basic_val_loader(transforms, task_name, diameters_list):
    """
    Build validation loader for the *basic* (old) task to evaluate forgetting.
    """
    val_ds = Dataset_Test_frequency(transform=transforms, task=task_name,
                                    read_type='test', diameters_list=diameters_list)
    val_loader = DataLoader(val_ds,
                            batch_size=1,
                            shuffle=False,
                            drop_last=False,
                            num_workers=config.n_threads,
                            pin_memory=config.use_cuda)
    print(f"Classification | Number of basic val samples for {task_name}: {len(val_ds)}")
    return val_loader


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
def train(model, teacher_model, diameters_list, basic_task, incre_task, save_name):
    """
    Train model for the given incremental task, while monitoring the basic task (old) performance.
    """
    val_tfms = monai.transforms.Compose([monai.transforms.ToTensor()])

    # Incremental task loaders
    train_loader, val_loader = build_class_loaders(val_tfms, incre_task, diameters_list)
    # Old/basic task loader for retention evaluation
    basic_val_loader = build_basic_val_loader(val_tfms, basic_task, diameters_list)

    # Minimal training log
    log_cols = ['epoch', 'Train_loss', 'Train_F1',
                'Test_loss', 'Test_F1', 'Test_F1_old', 'Mean_TrCls']
    log_df = pd.DataFrame(columns=log_cols)

    if config.use_cuda:
        model.cuda()
        teacher_model.cuda()

    loss_ce = CrossEntropyLoss(reduction='mean')

    # Only optimize parameters that require grad
    optim_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(optim_params, lr=config.lr_self, weight_decay=config.weight_decay)

    global_step = 0
    best_score_save = -1.0

    save_dir = os.path.join(config.data_dir, 'results', str(save_name))
    os.makedirs(save_dir, exist_ok=True)

    model.train()
    iter_cls_idx = 0
    train_iter = iter(train_loader)

    for epoch in range(config.epochs):
        train_loss_meter = AverageMeter()
        train_preds, train_gts = [], []

        # Reset best comparison after save_epoch (as in original logic)
        if epoch == config.save_epoch:
            best_score_save = -1.0

        # Fixed number of iterations per epoch (kept to preserve training behavior)
        for _ in range(100):
            iter_cls_idx += 1
            # Recycle the iterator when reaching the end of the loader
            if iter_cls_idx >= len(train_loader):
                train_iter = iter(train_loader)
                iter_cls_idx = 0

            # ---- Fetch a batch ----
            imgs, labels = next(train_iter)
            imgs_cuda = [to_device(img, gpu=config.use_cuda) for img in imgs]
            labels = to_device(labels, gpu=config.use_cuda)

            # ---- Forward & backward ----
            loss, logits = model(imgs_cuda, labels, loss_ce, teacher_model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ---- Bookkeeping ----
            train_loss_meter.update(loss.item(), n=1)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_gts.extend(labels.cpu().numpy())
            global_step += 1

        lr = get_learning_rate(optimizer)
        print(f'Global_Step {global_step} | Train Epoch: {epoch} | lr {lr:.2e} |')

        train_loss_avg = train_loss_meter.avg
        train_f1 = f1_from_preds(train_preds, train_gts)
        mean_tr_cls = train_f1  # keep original meaning

        # ---- Evaluate incremental (new) task & basic (old) task ----
        # Returns: [loss, acc, f1, precision, recall]
        metrics_new = eval_metrics_for_class(model, val_loader, cls_name=incre_task, old=False)
        metrics_old = eval_metrics_for_class(model, basic_val_loader, cls_name=basic_task, old=True)

        test_loss = metrics_new[0]
        f1_new = metrics_new[2]
        f1_old = metrics_old[2]

        # ---- Log CSV ----
        row = pd.Series([epoch, train_loss_avg, train_f1,
                         test_loss, f1_new, f1_old, mean_tr_cls],
                        index=log_cols)
        log_df = pd.concat([log_df, row.to_frame().T], ignore_index=True)
        log_df.to_csv(os.path.join(save_dir, 'log.csv'), index=False)

        # ---- Model selection: maximize (F1_new + F1_old) ----
        compare_score = (f1_new + f1_old)
        if epoch >= config.save_epoch and compare_score > best_score_save:
            save_cfg = {
                'name': save_name,
                'save_dir': save_dir,
                'global_step': global_step,
                'Test_F1': f1_new,
                'Test_F1_old': f1_old,
            }
            save_model(model=model, cfg=save_cfg)
            best_score_save = compare_score

        if compare_score > best_score_save:
            best_score_save = compare_score


def prepare_teacher_and_student(model, basic_task, basic_model_paths, in_channels):
    """
    Build a teacher model, load its weights (.pth state_dict), copy overlapping
    parameters to `model.student_model`, freeze teacher, enable student.

    Args:
        model: The FADRNet instance that has `student_model` (and optionally `teacher_model`).
        basic_task: Name of the basic task used to pick the teacher checkpoint.
        basic_model_paths: Dict mapping task -> checkpoint path (state_dict .pth).
        in_channels: Input channels for PreIncrNet (default 1).
    Returns:
        teacher_model: The instantiated and frozen teacher network.
        load_info: A dict with 'missing_keys' and 'unexpected_keys' from load_state_dict.
    """
    # 1) Instantiate teacher
    teacher_model = PreIncrNet(basic_task=basic_task, in_channels=in_channels)

    # 2) Load teacher weights (state_dict .pth)
    ckpt_path = basic_model_paths[basic_task]
    print(f"[Teacher] Loading state_dict for {basic_task} from {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))

    # 3) Load into teacher model
    load_result = teacher_model.load_state_dict(state_dict)

    # 4) Copy overlapping weights from teacher to student's state_dict
    student_sd = model.student_model.state_dict()
    teacher_sd = teacher_model.state_dict()
    overlap = {k: v for k, v in teacher_sd.items() if k in student_sd}
    student_sd.update(overlap)
    model.student_model.load_state_dict(student_sd)

    # 5) Freeze teacher; enable student
    for p in teacher_model.parameters():
        p.requires_grad = False
    for p in model.student_model.parameters():
        p.requires_grad = True

    return teacher_model

# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # Reproducibility: seeds & deterministic flags
    seed = config.random_seed
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print("Available GPUs:", torch.cuda.device_count())

    # Frequency radii/diameters for conditioning (kept as in original)
    diameters_list = [70, 110, 150, 999]
    diameters_len = len(diameters_list)

    # Task space
    basic_task_list = ['_1p19q', 'IDH', 'LHG']

    # Pretrained basic models to warm-start student branch
    basic_model_paths = {
        '_1p19q': './checkpoints/basic_model_1p19q.pth',
        'IDH': './checkpoints/basic_model_IDH.pth',
        'LHG': './checkpoints/basic_model_LHG.pth',
    }
    # Train all incremental pairs: for each basic task, iterate over the other tasks as incremental targets
    for basic_task in basic_task_list:
        incremental_tasks = [t for t in basic_task_list if t != basic_task]
        for incre_task in incremental_tasks:
            model = FADRNet(basic_task=basic_task, incre_task=incre_task,
                            diameters_len=diameters_len, in_channels=1, n_classes=2)
            # ---- Load basic (teacher) model state_dict and copy matching weights to student branch ----
            teacher_model = prepare_teacher_and_student(
                model=model, basic_task=basic_task, basic_model_paths=basic_model_paths, in_channels=1)
            # Train & save
            save_name = model.name
            print(save_name)
            train(model, teacher_model, diameters_list, basic_task, incre_task, save_name)