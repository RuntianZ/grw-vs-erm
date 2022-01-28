"""
Training on CelebA

Algorithms (--alg):
erm     ERM
iw      IW
gdro    Group DRO
"""

import os
import argparse
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.sampler import WeightedRandomSampler

from torchvision.datasets import CelebA
from torchvision.models import resnet18
import torchvision.transforms as transforms

uniform_algs = ['iw']


def get_transform_celebA(augment, target_w=None, target_h=None):
  # Reference: https://github.com/kohpangwei/group_DRO/blob/master/data/celebA_dataset.py#L80
  orig_w = 178
  orig_h = 218
  orig_min_dim = min(orig_w, orig_h)
  if target_w is not None and target_h is not None:
    target_resolution = (target_w, target_h)
  else:
    target_resolution = (orig_w, orig_h)

  if not augment:
    transform = transforms.Compose([
      transforms.CenterCrop(orig_min_dim),
      transforms.Resize(target_resolution),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
  else:
    # Orig aspect ratio is 0.81, so we don't squish it in that direction any more
    transform = transforms.Compose([
      transforms.RandomResizedCrop(
        target_resolution,
        scale=(0.7, 1.0),
        ratio=(1.0, 1.3333333333333333),
        interpolation=2),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
  return transform


def main():
  parser = argparse.ArgumentParser()

  # Basic settings
  parser.add_argument('--data_root', type=str)
  parser.add_argument('--device', default='cuda', type=str)
  parser.add_argument('--seed', type=int)
  parser.add_argument('--save_file', type=str)
  parser.add_argument('--download', default=False, action='store_true')

  # Training settings
  parser.add_argument('--alg', type=str)
  parser.add_argument('--epochs', type=int)
  parser.add_argument('--batch_size', default=400, type=int)
  parser.add_argument('--lr', default=0.001, type=float)
  parser.add_argument('--wd', default=0, type=float)
  parser.add_argument('--scheduler', type=str)
  parser.add_argument('--dro_step_size', default=0.01, type=float)
  parser.add_argument('--test_train', default=False, action='store_true')

  args = parser.parse_args()
  print('Algorithm: {}'.format(args.alg))
  print('eps: {}'.format(args.eps))
  print('dro_step_size: {}'.format(args.dro_step_size))
  print('Batch size: {}'.format(args.batch_size))
  print('lr: {}'.format(args.lr))
  print('wd: {}'.format(args.wd))
  print('Epochs: {}'.format(args.epochs))
  print('Test train: {}'.format(args.test_train))

  data_root = args.data_root
  device = args.device
  if args.save_file is not None:
    d = os.path.dirname(os.path.abspath(args.save_file))
    if not os.path.isdir(d):
      os.makedirs(d)

  # Prepare dataset
  target_w = 224
  target_h = 224
  n_classes = 2
  transform_train = get_transform_celebA(True, target_w, target_h)
  transform_test = get_transform_celebA(False, target_w, target_h)

  dataset_test = CelebA(data_root, split='test', target_type='attr',
                        transform=transform_test, download=args.download)
  dataset_valid = CelebA(data_root, split='valid', target_type='attr',
                         transform=transform_test, download=False)
  target_idx = 9  # Blond

  # Domains

  train_domain_fn = [
      lambda t: (t[:, 20] == 1) & (t[:, 9] == 1),  # Male            Blond
      lambda t: (t[:, 20] == 1) & (t[:, 9] == 0),  # Male            Not-Blond
      lambda t: (t[:, 20] == 0) & (t[:, 9] == 1),  # Female          Blond
      lambda t: (t[:, 20] == 0) & (t[:, 9] == 0),  # Female          Not-Blond
  ]

  test_domain_fn = train_domain_fn

  label_id = lambda t: t[:, target_idx]
  dataset_train = CelebA(data_root, split='train', target_type='attr',
                         transform=transform_train)

  # Fix seed for reproducibility
  if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_deterministic(True)
    cudnn.benchmark = False
  else:
    cudnn.benchmark = True

  # Build model
  model = resnet18()
  d = model.fc.in_features
  model.fc = nn.Linear(d, n_classes)
  model = model.to(device)
  model = torch.nn.DataParallel(model)
  num_domains = len(train_domain_fn)
  model.probs = torch.ones((num_domains,), dtype=torch.float) / num_domains
  model.probs = model.probs.to(device)

  # Reweighting
  print('Number of training samples in each groups:')
  num_train_samples = np.zeros((num_domains,), dtype=np.int)
  for i in range(num_domains):
    num_train_samples[i] = sum(train_domain_fn[i](dataset_train.attr))
    print('Group {}: {}'.format(i, num_train_samples[i]))
  assert sum(num_train_samples) == len(dataset_train.attr)
  print('====')
  print('Total: {}'.format(len(dataset_train.attr)))
  rw_weights = len(dataset_train.attr) / num_train_samples * 0.1

  print(rw_weights)
  sampler = None

  rw_sample_weights = torch.zeros((len(dataset_train.attr),), dtype=torch.float)
  for i in range(num_domains):
    rw_sample_weights[train_domain_fn[i](dataset_train.attr)] = rw_weights[i]

  if args.alg in uniform_algs:
    sampler = WeightedRandomSampler(rw_sample_weights, len(dataset_train.attr), replacement=True)

  trainloader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=(sampler is None),
                           sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
  test_trainloader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False,
                                num_workers=4, pin_memory=True)
  testloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)
  validloader = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
  optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=args.wd)

  test_criterion = nn.CrossEntropyLoss(reduction='none')
  criterion = test_criterion

  scheduler = None
  if args.scheduler is not None:
    milestones = args.scheduler.split(',')
    milestones = [int(s) for s in milestones]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    print('Scheduler: {}'.format(milestones))

  # Training
  val_avg_acc = []
  val_avg_loss = []
  val_group_acc = []
  val_group_loss = []
  avg_acc = []
  avg_loss = []
  group_acc = []
  group_loss = []
  group_probs = [] # For group dro
  train_avg_acc = []
  train_avg_loss = []
  train_group_acc = []
  train_group_loss = []
  best_valid = 0.
  best_epoch = 0
  best_acc = 0.
  best_worst_acc = 0.

  # Main loop
  for epoch in range(args.epochs):
    print('===Train(epoch={})==='.format(epoch + 1))
    model.train()
    if args.alg == 'erm':
      erm_train(model, trainloader, optimizer, criterion, device, label_id)
    elif args.alg == 'gdro':
      group_dro_train(model, trainloader, optimizer, criterion, device,
                      train_domain_fn, label_id, args.dro_step_size)
    elif args.alg == 'iw':
      erm_train(model, trainloader, optimizer, criterion, device, label_id)
    else:
      raise NotImplementedError

    if scheduler is not None:
      scheduler.step()

    if args.test_train:
      print('===Test Train(epoch={})'.format(epoch + 1))
      a, b, c, d, c_rec = test(model, test_trainloader, test_criterion, device,
                        test_domain_fn, label_id)
      train_avg_acc.append(a)
      train_avg_loss.append(b)
      train_group_acc.append(c)
      train_group_loss.append(d)

    print('===Validation(epoch={})==='.format(epoch + 1))
    a, b, c, d, e, f, c_rec = test(model, validloader, test_criterion, device,
                            test_domain_fn, label_id, True)
    val_avg_acc.append(a)
    val_avg_loss.append(b)
    val_group_acc.append(c)
    val_group_loss.append(d)
    worst_acc = c.min()
    if worst_acc > best_valid:
      best_valid = worst_acc
      best_epoch = epoch + 1
    print('===Test(epoch={})==='.format(epoch + 1))
    a, b, c, d, c_rec = test(model, testloader, test_criterion, device,
                      test_domain_fn, label_id)
    worst_acc = c.min()
    if best_epoch == epoch + 1:
      best_acc = a
      best_worst_acc = worst_acc
    avg_acc.append(a)
    avg_loss.append(b)
    group_acc.append(c)
    group_loss.append(d)
    group_probs.append(model.probs.detach().cpu().numpy())

  # Print the results
  print('===Results===')
  print('                           Best Epoch: {}'.format(best_epoch))
  print('      Test Accuracy of the Best Epoch: {}'.format(best_acc))
  print('Worst-case Accuracy of the Best Epoch: {}'.format(best_worst_acc))

  # Save the results
  if args.save_file is not None:
    mat = {
      'test_avg_acc': np.array(avg_acc),
      'test_avg_loss': np.array(avg_loss),
      'test_group_acc': np.array(group_acc),
      'test_group_loss': np.array(group_loss),
      'train_avg_acc': np.array(train_avg_acc),
      'train_avg_loss': np.array(train_avg_loss),
      'train_group_acc': np.array(train_group_acc),
      'train_group_loss': np.array(train_group_loss),
      'val_avg_acc': np.array(val_avg_acc),
      'val_avg_loss': np.array(val_avg_loss),
      'val_group_acc': np.array(val_group_acc),
      'val_group_loss': np.array(val_group_loss),
      'best_epoch': best_epoch,
      'best_acc': best_acc,
      'best_worst_acc': best_worst_acc,
      'group_probs': np.array(group_probs),
    }
    sio.savemat(args.save_file, mat)


def test(model: Module, loader: DataLoader, criterion, device: str,
         domain_fn, label_id, need_cvar=False):
  """Test the avg and group acc of the model"""

  model.eval()
  total_correct = 0
  total_loss = 0
  total_num = 0
  num_domains = len(domain_fn)
  group_correct = np.zeros((num_domains,), dtype=np.int)
  group_loss = np.zeros((num_domains,), dtype=np.float)
  group_num = np.zeros((num_domains,), dtype=np.int)
  l_rec = []
  c_rec = []
  alpha = 0.1
  eps = 0.01

  with torch.no_grad():
    for _, (inputs, targets) in enumerate(loader):
      inputs, targets = inputs.to(device), targets.to(device)

      labels = label_id(targets)
      outputs = model(inputs)
      predictions = torch.argmax(outputs, dim=1)
      # print(predictions)

      c = (predictions == labels)
      c_rec.append(c.detach().cpu().numpy())
      correct = c.sum().item()
      l = criterion(outputs, labels).view(-1)
      if need_cvar:
        l_rec.append(l.detach().cpu().numpy())
      loss = l.sum().item()
      total_correct += correct
      total_loss += loss
      total_num += len(inputs)

      for i in range(num_domains):
        g = domain_fn[i](targets)
        group_correct[i] += c[g].sum().item()
        group_loss[i] += l[g].sum().item()
        group_num[i] += g.sum().item()

  print('Acc: {} ({} of {})'.format(total_correct / total_num, total_correct, total_num))
  print('Avg Loss: {}'.format(total_loss / total_num))
  for i in range(num_domains):
    print('Group {:2}\tAcc: {} ({} of {})'.format(i, group_correct[i] / group_num[i],
                                                group_correct[i], group_num[i]))
    print('Group {:2}\tAvg Loss: {}'.format(i, group_loss[i] / group_num[i]))

  c_rec = np.concatenate(c_rec)
  if need_cvar:
    l_vec = np.concatenate(l_rec)
    n = int(len(l_vec) * alpha)
    l = np.sort(l_vec)
    l1 = l[-n:]
    cvar_loss = l1.mean()
    print('CVaR loss: {}'.format(cvar_loss))

    n1 = int(len(l_vec) * (eps + alpha * (1 - eps)))
    n2 = int(len(l_vec) * eps)
    l2 = l[-n1:-n2]
    robust_cvar_loss = l2.mean()
    print('Robust CVaR loss: {}'.format(robust_cvar_loss))
    return total_correct / total_num, total_loss / total_num, \
           group_correct / group_num, group_loss / group_num, \
           cvar_loss, robust_cvar_loss, c_rec

  return total_correct / total_num, total_loss / total_num, \
         group_correct / group_num, group_loss / group_num, c_rec


########################################################

def erm_train(model, loader, optimizer, criterion, device,
              label_id):
  """Empirical Risk Minimization (ERM)"""

  for _, (inputs, targets) in enumerate(loader):
    inputs, targets = inputs.to(device), targets.to(device)
    targets = label_id(targets)

    outputs = model(inputs)
    loss = criterion(outputs, targets).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

def group_dro_train(model, loader, optimizer, criterion, device,
                    domain_fn, label_id, dro_step_size):
  
  num_domains = len(domain_fn)

  for _, (inputs, targets) in enumerate(loader):
    inputs, targets = inputs.to(device), targets.to(device)
    labels = label_id(targets)
    batch_size = len(inputs)

    # Determine group membership
    domain_id = -torch.ones((batch_size,), dtype=torch.long).to(device)
    for i in range(num_domains):
      g = domain_fn[i](targets)
      domain_id[g] = i
    assert not (domain_id == -1).any()

    optimizer.zero_grad()
    logits = model(inputs)
    loss = criterion(logits, labels)

    gmap = (domain_id == (torch.arange(num_domains).unsqueeze(1).long().to(device)))
    group_map = gmap.float()

    group_count = group_map.sum(1)
    group_denom = group_count + (group_count == 0).float()  # avoid nans
    group_loss = (group_map @ loss.view(-1)) / group_denom  # average loss

    model.probs *= torch.exp(dro_step_size * group_loss)
    model.probs /= model.probs.sum()
    model.probs = model.probs.detach()
    loss = group_loss @ model.probs

    loss.backward()
    optimizer.step()


if __name__ == '__main__':
  main()
