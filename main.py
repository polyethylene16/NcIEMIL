import sys
import glob
import os
import argparse
import numpy as np
import itertools
from scnet import CISMIL
from model import SCNet
# from models.ilra import ILRA
# from models.abmil import AbMIL
# from models.dsmil import DSMIL
# from models.clam import CLAM_MB, CLAM_SB
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import json
from sklearn.preprocessing import label_binarize

from pathlib import Path

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from torch.utils.tensorboard import SummaryWriter
from timm.utils import NativeScaler, get_state_dict, ModelEma
import pandas as pd
import matplotlib
import utils

matplotlib.use('Agg')
import matplotlib.pyplot as plt

class DataSet(data.Dataset):
    def __init__(self, data_dir, fold, state=None):
        self.data_dir = data_dir
        self.fold = fold
        self.slide_data = pd.read_csv(self.fold, index_col=0)

        if state == 'train':
            self.data = self.slide_data.loc[:, 'train_slide'].dropna()
            self.label = self.slide_data.loc[:, 'train_label'].dropna()
        if state == 'val':
            self.data = self.slide_data.loc[:, 'val_slide'].dropna()
            self.label = self.slide_data.loc[:, 'val_label'].dropna()
        # if state == 'test':
        #     self.data = self.slide_data.loc[:, 'test'].dropna()
        #     self.label = self.slide_data.loc[:, 'test_label'].dropna()


    def __getitem__(self, item):
        slide_id = self.data[item]
        label = int(self.label[item])
        slide_path = Path(self.data_dir) / f'{slide_id}.pth'
        data = torch.load(slide_path)

        return data, label

    def __len__(self):
        return len(self.data)


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = torch.zeros(1).to(device)
    train_loader = tqdm(train_loader, file=sys.stdout, ncols=160, colour='blue')

    for i, (data, label) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        label = label.to(device)

        # output, _ = model(data)
        # loss = criterion(output, label)

        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        total_loss = (total_loss * i + loss.detach()) / (i + 1)
        train_loader.desc = 'Train\t[epoch {}] lr: {}\tloss {}'.format(epoch, optimizer.param_groups[0]["lr"], round(total_loss.item(), 3))

    return total_loss.item()


@torch.no_grad()
def val_one_epoch(model, val_loader, device):
    model.eval()
    labels = torch.tensor([], device=device)
    preds = torch.tensor([], device=device)

    for i, (data, label) in enumerate(val_loader):

        data = data.to(device)
        label = label.to(device)
        output = model(data)
        labels = torch.cat([labels, label], dim=0)
        preds = torch.cat([preds, output.detach()], dim=0)

    return preds.cpu(), labels.cpu()


def calculate_metrics(logits: torch.Tensor, targets: torch.Tensor, num_classes, confusion_mat=False):
    targets = targets.numpy()
    _, pred = torch.max(logits, dim=1)
    pred = pred.numpy()
    acc = accuracy_score(targets, pred)
    f1 = f1_score(targets, pred, average='macro')

    probs = F.softmax(logits, dim=1)
    probs = probs.numpy()
    if len(np.unique(targets)) != num_classes:
        roc_auc = 0
    else:
        if num_classes == 2:
            fpr, tpr, _ = roc_curve(y_true=targets, y_score=probs[:, 1], pos_label=1)
            roc_auc = auc(fpr, tpr)
        else:
            binary_labels = label_binarize(targets, classes=[i for i in range(num_classes)])
            valid_classes = np.where(np.any(binary_labels, axis=0))[0]
            binary_labels = binary_labels[:, valid_classes]
            valid_cls_probs = probs[:, valid_classes]
            fpr, tpr, _ = roc_curve(y_true=binary_labels.ravel(), y_score=valid_cls_probs.ravel())
            roc_auc = auc(fpr, tpr)
    if confusion_mat:
        mat = confusion_matrix(targets, pred)
        return acc, f1, roc_auc, mat
    return acc, f1, roc_auc


def draw_metrics(tb_writer, name, num_class, acc, auc, mat, f1, step):
    tb_writer.add_scalar("{}/acc".format(name), acc, step)
    tb_writer.add_scalar("{}/auc".format(name), auc, step)
    tb_writer.add_scalar("{}/f1".format(name), f1, step)
    if mat is not None:
        tb_writer.add_figure("{}/confusion mat".format(name),
                             plot_confusion_matrix(cmtx=mat, num_classes=num_class), step)
        

def plot_confusion_matrix(cmtx, num_classes, class_names=None, title='Confusion matrix', normalize=False,
                          cmap=plt.cm.Blues):
    if normalize:
        cmtx = cmtx.astype('float') / cmtx.sum(axis=1)[:, np.newaxis]
    if class_names is None or type(class_names) != list:
        class_names = [str(i) for i in range(num_classes)]

    figure = plt.figure()
    plt.imshow(cmtx, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    fmt = '.2f' if normalize else 'd'
    threshold = cmtx.max() / 2.0
    for i, j in itertools.product(range(cmtx.shape[0]), range(cmtx.shape[1])):
        plt.text(j, i, format(cmtx[i, j], fmt), horizontalalignment="center",
                 color="white" if cmtx[i, j] > threshold else "black")

    plt.tight_layout()
    plt.ylabel("Ground Truth")
    plt.xlabel("Predict")

    return figure

def parse():
    parser = argparse.ArgumentParser('Training for MIL')
    parser.add_argument('--epochs', type=int, default=300)
    # Model parameters
    parser.add_argument('--in_dim', type=int, default=768, 
                        help="The dimension of instance-level representations")
    parser.add_argument('--drop', type=float, default=0.2, metavar='PCT',
                        help='Dropout rate (default: 0.1)')
    
    # Optimizer parameters
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adam"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='weight decay (default: 1e-5)')
    
    # Dataset parameters
    parser.add_argument('--data', type=str, default='', 
                        help='feature path')
    parser.add_argument('--k', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--flag', type=str, default='stomoch_1024_Si_256_2')
    parser.add_argument('--fold', type=str, default='/data_sda/sqh/MIL_v2/fold/fold.csv')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use for training / testing')
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seed', default=256, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--n_classes', type=int, default=4)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--output_dir', default='/data_sda/sqh/MIL_v2/results',
                        help='path where to save, empty for no saving')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')

    return parser.parse_args()


def main(args):
    print(args)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    max_score, max_acc, max_auc, max_f1 = 0.0, 0.0, 0.0, 0.0

    train_set = DataSet(data_dir=args.data, fold=args.fold, state="train")
    val_set = DataSet(data_dir=args.data, fold=args.fold, state="val")
    test_set = DataSet(data_dir=args.data, fold=args.fold, state="test")

    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True, pin_memory=True)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=False, pin_memory=True)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=False, pin_memory=True)

    model = CISMIL(in_dim=args.in_dim, in_chans=2 * args.k, n_classes=args.n_classes, attn_drop=args.drop, proj_drop=args.drop, 
                   conv_drop=args.drop, latent_dim=args.embed_dim).to(device)
    # model = SCNet(n_chans=2 * args.k, n_classes=4).to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    parameters_g = [param for param in model.parameters() if param.requires_grad]
    # optimizer = torch.optim.Adam(parameters_g, lr=2e-4, weight_decay=1e-5)
    optimizer = torch.optim.SGD(parameters_g, lr=2e-4, weight_decay=1e-5)
    
    criterion = torch.nn.CrossEntropyLoss()
    output_dir = Path(args.output_dir)
    weight_dir = output_dir / "weight"
    log_dir = output_dir / "log"
    if weight_dir.exists() is not True:
        weight_dir.mkdir()
    
    if log_dir.exists() is not True:
        log_dir.mkdir()

    log = log_dir / args.flag
    weight = weight_dir / args.flag

    if log.exists() is not True:
        log.mkdir()

    if weight.exists() is not True:
        weight.mkdir()

    tb_writer = SummaryWriter(log)
    print('Set Tensorboard: {}'.format(log))

    if args.resume == '':
        tmp = f"{weight}/checkpoint.pth"
        if os.path.exists(tmp):
            args.resume = tmp

    flag = os.path.exists(args.resume)
    if args.resume and flag:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
        if 'max_score' in checkpoint:
            max_score = checkpoint['max_score']

    print(f"Start training for {args.epochs} epochs")
    for epoch in range(args.start_epoch, args.epochs):
        loss = train_one_epoch(model=model, train_loader=train_loader, criterion=criterion, 
                                      optimizer=optimizer, device=device, epoch=epoch + 1)
        
        if args.output_dir:
            checkpoint_paths = [weight/ 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'max_score': max_score,
                }, checkpoint_path)
            if epoch % 10 == 0:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'max_score': max_score,
                }, f"{weight}/backup.pth")

        val_preds, val_labels = val_one_epoch(model=model, val_loader=val_loader, device=device)
        val_acc, val_f1, val_auc, val_mat = calculate_metrics(logits=val_preds, targets=val_labels, num_classes=args.n_classes, confusion_mat=True)
        draw_metrics(tb_writer, 'Val', args.n_classes, val_acc, val_auc, val_mat, val_f1, epoch + 1)
        print('Val\t[epoch {}] acc:{}\tauc:{}\tf1-score:{}'.format(epoch + 1, val_acc, val_auc, val_f1))
        max_score = max(max_score, 3 / (1 / val_acc + 1 / val_auc + 1 / val_f1))
        if max_score == 3 / (1 / val_acc + 1 / val_auc + 1 / val_f1):
            torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'max_score': max_score,
                }, f"{weight}/best.pth")
        
            # test_preds, test_labels = val_one_epoch(model=model, val_loader=test_loader, device=device)
            # test_acc, test_f1, test_auc, test_mat = calculate_metrics(logits=test_preds, targets=test_labels, num_classes=args.n_classes, confusion_mat=True)
            # print('Test\t[epoch {}] acc:{}\tauc:{}\tf1-score:{}'.format(epoch + 1, test_acc, test_auc, test_f1))
            max_acc = val_acc
            max_auc = val_auc
            max_f1 = val_f1

            # draw_metrics(tb_writer, 'Test', args.n_classes, val_acc, val, test_mat, test_f1, epoch + 1)
        print('Max val score: {:.4f}%'.format(max_score))
        print("Max val accuracy: {:.4f}%\tMax val AUC:{:.4f}%\tMax val F1:{:.4f}%".format(max_acc, max_auc, max_f1))


if __name__ == '__main__':
    opt = parse()
    main(opt)
