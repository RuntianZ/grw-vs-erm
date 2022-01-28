"""
Training on Waterbirds

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
from torchvision.models import resnet18

from datasets.waterbirds import WaterBirds

uniform_algs = ['iw']


def main():
  parser = argparse.ArgumentParser()

  # Basic settings
  parser.add_argument('--data_root', type=str)
  parser.add_argument('--device', default='cuda', type=str)
  parser.add_argument('--seed', type=int)
  parser.add_argument('--save_file', type=str)

  # Training settings
  parser.add_argument('--alg', type=str)
  parser.add_argument('--epochs', default=500, type=int)
  parser.add_argument('--batch_size', default=128, type=int)
  parser.add_argument('--lr', default=0.0001, type=float)
  parser.add_argument('--wd', default=0, type=float)
  parser.add_argument('--scheduler', type=str)
  parser.add_argument('--dro_step_size', default=0.01, type=float)
  parser.add_argument('--test_train', default=False, action='store_true')

  args = parser.parse_args()
  print('Algorithm: {}'.format(args.alg))
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

  dataset_train = WaterBirds(data_root, 'train', 'train')
  dataset_traintest = WaterBirds(data_root, 'train', 'test')
  dataset_valid = WaterBirds(data_root, 'val', 'test')
  dataset_test = WaterBirds(data_root, 'test', 'test')

  # Fix seed for reproducibility
  if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_deterministic(True)
    cudnn.benchmark = False
  else:
    cudnn.benchmark = True

  # Build model
  model = resnet18(pretrained=True)
  num_classes = 2
  num_domains = 4
  d = model.fc.in_features
  model.fc = nn.Linear(d, num_classes)
  model = model.to(device)
  model = torch.nn.DataParallel(model)
  model.probs = torch.ones((num_domains,), dtype=torch.float) / num_domains
  model.probs = model.probs.to(device)

  # Reweighting
  print('Number of training samples in each groups:')
  num_train_samples = np.zeros((num_domains,), dtype=np.int)
  for i in range(num_domains):
    num_train_samples[i] = sum(dataset_train.group_array == i)
    print('Group {}: {}'.format(i, num_train_samples[i]))
  assert sum(num_train_samples) == len(dataset_train)
  print('====')
  print('Total: {}'.format(len(dataset_train)))
  rw_weights = len(dataset_train) / num_train_samples * 0.1

  rw_sample_weights = torch.zeros((len(dataset_train),), dtype=torch.float)
  for i in range(num_domains):
    rw_sample_weights[dataset_train.group_array == i] = rw_weights[i]
  sampler = None
  if args.alg in uniform_algs:
    sampler = WeightedRandomSampler(rw_sample_weights, len(dataset_train), replacement=True)

  trainloader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=(sampler is None),
                           sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
  test_trainloader = DataLoader(dataset_traintest, batch_size=args.batch_size, shuffle=False,
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
    if args.alg == 'erm' or args.alg == 'iw':
      erm_train(model, trainloader, optimizer, criterion, device)
    elif args.alg == 'gdro':
      group_dro_train(model, trainloader, optimizer,
                      criterion, device, args.dro_step_size)
    else:
      raise NotImplementedError

    if scheduler is not None:
      scheduler.step()

    if args.test_train:
      print('===Test Train(epoch={})'.format(epoch + 1))
      a, b, c, d, c_rec = test(model, test_trainloader, test_criterion, device)
      train_avg_acc.append(a)
      train_avg_loss.append(b)
      train_group_acc.append(c)
      train_group_loss.append(d)
  
    print('===Validation(epoch={})==='.format(epoch + 1))
    a, b, c, d, c_rec = test(model, validloader, test_criterion, device)
    val_avg_acc.append(a)
    val_avg_loss.append(b)
    val_group_acc.append(c)
    val_group_loss.append(d)
    worst_acc = c.min()
    if worst_acc > best_valid:
      best_valid = worst_acc
      best_epoch = epoch + 1
    print('===Test(epoch={})==='.format(epoch + 1))
    a, b, c, d, c_rec = test(model, testloader, test_criterion, device)
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
      'avg_acc': np.array(avg_acc),
      'avg_loss': np.array(avg_loss),
      'group_acc': np.array(group_acc),
      'group_loss': np.array(group_loss),
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

########################################################
def test(model: Module, loader: DataLoader, criterion, device: str):
  """Test the avg and group acc of the model"""

  model.eval()
  total_correct = 0
  total_loss = 0
  total_num = 0
  num_domains = 4
  group_correct = np.zeros((num_domains,), dtype=np.int)
  group_loss = np.zeros((num_domains,), dtype=np.float)
  group_num = np.zeros((num_domains,), dtype=np.int)
  c_rec = []

  with torch.no_grad():
    for _, (inputs, targets, group_labels) in enumerate(loader):
      inputs, targets = inputs.to(device), targets.to(device)

      labels = targets
      outputs = model(inputs)
      predictions = torch.argmax(outputs, dim=1)

      c = (predictions == labels)
      c_rec.append(c.detach().cpu().numpy())
      correct = c.sum().item()
      l = criterion(outputs, labels).view(-1)
      loss = l.sum().item()
      total_correct += correct
      total_loss += loss
      total_num += len(inputs)

      for i in range(num_domains):
        g = (group_labels == i)
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
  return total_correct / total_num, total_loss / total_num, \
         group_correct / group_num, group_loss / group_num, c_rec


########################################################
def erm_train(model, loader, optimizer, criterion, device):
  """Empirical Risk Minimization (ERM)"""

  for _, (inputs, targets, group_labels) in enumerate(loader):
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, targets).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def group_dro_train(model, loader, optimizer, criterion, device,
                    dro_step_size):

  num_domains = 4

  for _, (inputs, targets, group_labels) in enumerate(loader):
    inputs, targets = inputs.to(device), targets.to(device)
    batch_size = len(inputs)

    # Determine group membership
    domain_id = -torch.ones((batch_size,), dtype=torch.long).to(device)
    for i in range(num_domains):
      g = (group_labels == i)
      domain_id[g] = i
    assert not (domain_id == -1).any()

    optimizer.zero_grad()
    logits = model(inputs)
    loss = criterion(logits, targets)

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