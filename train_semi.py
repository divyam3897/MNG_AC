import argparse
import copy
import logging
import math
import random
import sys
import time

import apex.amp as amp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from preact_resnet import resnet50 as ResNet50
from wideresnet import WideResNet
from evaluate import clamp, norms, norms_l1, norms_p
from evaluate import l1_dir_topk, proj_l1ball, proj_simplex
from torch.distributions import laplace

from torch_backend import *
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler
from collections import OrderedDict
import torch.nn.functional as F
from torch import autograd
from losses import trades_loss
from datasets import SemiSupervisedDataset, SemiSupervisedSampler, DATASETS

logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(asctime)s] - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.DEBUG)

cifar10_mean = (0.0, 0.0, 0.0)
cifar10_std = (1.0, 1.0, 1.0)
mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()
upper_limit = ((1 - mu) / std)
lower_limit = ((0 - mu) / std)


def initialize_weights(module):
  if isinstance(module, nn.Conv2d):
    n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
    module.weight.data.normal_(0, math.sqrt(2. / n))
    if module.bias is not None:
      module.bias.data.zero_()
  elif isinstance(module, nn.BatchNorm2d):
    module.weight.data.fill_(1)
    module.bias.data.zero_()
  elif isinstance(module, nn.Linear):
    module.bias.data.zero_()


def attack_pgd(model, X, y, opt, norm, dataset, params=None):
  delta = torch.zeros_like(X).cuda()
  if norm == "linf":
    if dataset == "cifar10" or dataset == "svhn":
      epsilon = (8 / 255.) / std
      attack_iters = 10
      alpha = (1. / 255.) / std
    else:
      epsilon = (4 / 255.) / std
      attack_iters = 10
      alpha = (1 / 255.) / std
    delta[:, 0, :, :].uniform_(-epsilon[0][0][0].item(),
                               epsilon[0][0][0].item())
    delta[:, 1, :, :].uniform_(-epsilon[1][0][0].item(),
                               epsilon[1][0][0].item())
    delta[:, 2, :, :].uniform_(-epsilon[2][0][0].item(),
                               epsilon[2][0][0].item())
  elif norm == "l2":
    epsilon = (80 / 255.) / std
    attack_iters = 10
    alpha = (25. / 255.) / std
    delta = torch.rand_like(X, requires_grad=True)
    delta.data *= (2.0 * delta.data - 1.0) * epsilon
    delta.data /= norms_p(
        delta.detach(), 2.0).clamp(min=epsilon.detach().cpu().numpy()[0][0][0])
  elif norm == "l1":
    epsilon = (2000 / 255.) / std
    attack_iters = 20
    alpha = (255. / 255.) / std
    ini = laplace.Laplace(loc=delta.new_tensor(0), scale=delta.new_tensor(1))
    delta.data = ini.sample(delta.data.shape)
    delta.data = (2.0 * delta.data - 1.0) * epsilon
    delta.data /= norms_l1(
        delta.detach()).clamp(min=epsilon.detach().cpu().numpy()[0][0][0])
  delta.requires_grad = True
  criterion_kl = nn.KLDivLoss(reduction='sum')
  for _ in range(attack_iters):
    X_adv = X + delta

    with torch.enable_grad():
      loss = criterion_kl(F.log_softmax(model(X_adv), dim=1),
                          F.softmax(model(X), dim=1))
    grad = torch.autograd.grad(loss, [X_adv])[0]
    if norm == "linf":
      delta.data = clamp(delta.data + alpha * torch.sign(grad), -epsilon,
                         epsilon)
    elif norm == "l2":
      delta.data = delta.data + alpha * grad / norms_p(grad, 2.0)
      delta.data *= epsilon / norms_p(delta.detach(), 2.0).clamp(
          min=epsilon.detach().cpu().numpy()[0][0][0])
    elif norm == "l1":
      k = 99
      delta.data = delta.data + alpha * l1_dir_topk(grad, delta.data, X, k)
      delta.data = proj_l1ball(delta.data,
                               epsilon=epsilon.detach().cpu().numpy()[0][0][0],
                               device=device)
    delta.data = clamp(delta.data, lower_limit - X, upper_limit - X)
  return delta.detach()


def get_loaders(dir_, batch_size, dataset, rst):
  if dataset == "cifar10":
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])
    test_transform = transforms.Compose([transforms.ToTensor()])
  elif dataset == "svhn":
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
  elif dataset == "tinyimagenet":
    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([transforms.ToTensor()])

  num_workers = 2
  if dataset == "svhn":
    if not rst:
      train_dataset = datasets.SVHN(dir_,
                                    split='train',
                                    transform=train_transform,
                                    download=True)
      test_dataset = datasets.SVHN(dir_,
                                   split='test',
                                   transform=test_transform,
                                   download=True)
    else:
      train_dataset = SemiSupervisedDataset(base_dataset=dataset,
                                            add_svhn_extra=True,
                                            root=dir_,
                                            train=True,
                                            download=True,
                                            transform=train_transform,
                                            aux_data_filename=None,
                                            add_aux_labels=True,
                                            aux_take_amount=None)
      test_dataset = SemiSupervisedDataset(base_dataset=dataset,
                                           root=dir_,
                                           train=False,
                                           download=True,
                                           transform=test_transform)

  elif dataset == "cifar10":
    if not rst:
      train_dataset = datasets.CIFAR10(dir_,
                                       train=True,
                                       transform=train_transform,
                                       download=True)
      test_dataset = datasets.CIFAR10(dir_,
                                      train=False,
                                      transform=test_transform,
                                      download=True)
    else:
      train_dataset = SemiSupervisedDataset(
          base_dataset=dataset,
          add_svhn_extra=False,
          root=dir_,
          train=True,
          download=True,
          transform=train_transform,
          aux_data_filename='ti_500K_pseudo_labeled.pickle',
          add_aux_labels=True,
          aux_take_amount=None)
      test_dataset = SemiSupervisedDataset(base_dataset=dataset,
                                           root=dir_,
                                           train=False,
                                           download=True,
                                           transform=test_transform)
  elif dataset == "tinyimagenet":
    train_dataset = torchvision.datasets.ImageFolder(root=dir_ + '/train',
                                                     transform=train_transform)
    test_dataset = torchvision.datasets.ImageFolder(root=dir_ + '/val',
                                                    transform=test_transform)

  train_loader = torch.utils.data.DataLoader(
      dataset=train_dataset,
      batch_size=batch_size,
      shuffle=True,
      pin_memory=True,
      num_workers=num_workers,
  )
  test_loader = torch.utils.data.DataLoader(
      dataset=test_dataset,
      batch_size=batch_size,
      shuffle=False,
      pin_memory=True,
      num_workers=2,
  )
  return train_loader, test_loader


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch-size', default=2048, type=int)
  parser.add_argument('--data-dir', default='../cifar-data', type=str)
  parser.add_argument('--dataset', default='cifar10', type=str)
  parser.add_argument('--epochs', default=30, type=int)
  parser.add_argument('--n_classes', default=10, type=int)
  parser.add_argument('--lr-schedule',
                      default='cyclic',
                      choices=['cyclic', 'piecewise'])
  parser.add_argument('--lr-max', default=0.21, type=float)
  parser.add_argument('--attack',
                      default='pgd',
                      type=str,
                      choices=['pgd', 'fgsm', 'free', 'none'])
  parser.add_argument(
      '--attack_type',
      default='none',
      type=str,
      choices=['none', 'random', 'max', 'avg', 'avg_loss', 'meta'])
  parser.add_argument('--norm', default='linf', type=str)
  parser.add_argument('--epsilon', default=8, type=int)
  parser.add_argument('--attack-iters', default=8, type=int)
  parser.add_argument('--restarts', default=1, type=int)
  parser.add_argument('--pgd-alpha', default=2, type=int)
  parser.add_argument('--fgsm-alpha', default=1.25, type=float)
  parser.add_argument('--fgsm-init',
                      default='random',
                      choices=['zero', 'random', 'previous'])
  parser.add_argument('--fname', default='cifar_model_free1', type=str)
  parser.add_argument('--seed', default=0, type=int)
  parser.add_argument('--width-factor', default=10, type=int)
  parser.add_argument('--model', default='WideResNet')
  parser.add_argument('--rst', default=False, type=bool)
  parser.add_argument('--overfit-check', action='store_true')
  return parser.parse_args()


def main():
  args = get_args()
  logger.info(args)

  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)

  args.data_dir = args.dataset + "-data"
  args.n_classes = 10

  start_start_time = time.time()
  train_loader, test_loader = get_loaders(args.data_dir, args.batch_size,
                                          args.dataset, args.rst)

  epsilon = (args.epsilon / 255.) / std
  pgd_alpha = (args.pgd_alpha / 255.) / std

  if args.model == 'resnet50':
    model = ResNet50().cuda()
  elif args.model == 'WideResNet':
    model = WideResNet(28, 10, widen_factor=args.width_factor,
                       dropRate=0.0).cuda()
  else:
    raise ValueError("Unknown model")
  model = torch.nn.DataParallel(model).cuda()

  model.apply(initialize_weights)
  model.train()
  opt = torch.optim.SGD(model.params(),
                        lr=args.lr_max,
                        momentum=0.9,
                        weight_decay=5e-4)
  criterion = nn.CrossEntropyLoss()
  epochs = args.epochs

  if args.lr_schedule == 'cyclic':
    lr_schedule = lambda t: np.interp(
        [t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]
  elif args.lr_schedule == 'piecewise':

    def lr_schedule(t):
      if t / args.epochs < 0.5:
        return args.lr_max
      elif t / args.epochs < 0.75:
        return args.lr_max / 10.
      else:
        return args.lr_max / 100.

  prev_robust_acc = 0.
  logger.info('Epoch \t Time \t LR \t \t Train Loss \t Train Acc')
  criterion_kl = nn.KLDivLoss(reduction='sum')

  for epoch in range(epochs):
    start_time = time.time()
    train_loss = 0
    train_acc = 0
    train_n = 0
    for i, (X, y) in enumerate(train_loader):
      X, y = X.cuda(), y.cuda()
      lr = lr_schedule(epoch + (i + 1) / len(train_loader))
      opt.param_groups[0].update(lr=lr)

      output = model(X)
      norms_list = ["linf", "l1"]
      curr_norm = random.choices(norms_list)
      delta = attack_pgd(model, X, y, opt, curr_norm[0], args.dataset)
      output = model(clamp(X + delta[:X.size(0)], lower_limit, upper_limit))
      x_adv = clamp(X + delta[:X.size(0)], lower_limit, upper_limit)

      x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
      logits_adv = F.log_softmax(model(x_adv), dim=1)
      logits = model(X)

      loss_natural = F.cross_entropy(logits, y, ignore_index=-1)
      p_natural = F.softmax(logits, dim=1)
      loss_robust = criterion_kl(logits_adv, p_natural) / args.batch_size
      loss = loss_natural + 6 * loss_robust

      opt.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), 0.5)
      opt.step()
      train_loss += loss.item() * y.size(0)
      train_acc += (output.max(1)[1] == y).sum().item()
      train_n += y.size(0)

    best_state_dict = copy.deepcopy(model.state_dict())

    train_time = time.time()
    print('%d \t %.1f \t %.4f \t %.4f \t %.4f' %
          (epoch, (train_time - start_time) / 60, lr, train_loss / train_n,
           train_acc / train_n))
  torch.save(best_state_dict, args.fname + '.pth')
  logger.info('Total train time: %.4f minutes',
              (train_time - start_start_time) / 60)


if __name__ == "__main__":
  main()
