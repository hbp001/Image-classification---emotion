import argparse
from time import gmtime, strftime
import os
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from dataloarder import classification

from arch.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
import train
import val
import wandb
import cv2

# 해당 모듈이 임포트된 경우가 아닌 인터프리터에서 직접 실행된 경우에 if문 이하의 코드를 실행
if __name__ == '__main__':
    wandb.init(project='emotion')
    wandb.run.name = 'resnet18-7'
    wandb.run.save()
    # parsing:어떤 데이터를 원하는 모양으로 만들어내는 것
    # 인자값을 받을 수 있는 인스턴스 
    
    parser = argparse.ArgumentParser()
    # 입력 받을 인자값 등록
    parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18'])
    parser.add_argument('--lr_base', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr_drop_epochs', type=int, default=[30, 60, 90], nargs='+')
    parser.add_argument('--lr_drop_rate', type=float, default=0.1)
    # 입력 받은 인자값을 args에 저장
    args = parser.parse_args()
    wandb.config.update(args)
    
    # define model
    #startswich:문자열이 특정 문자로 시작하는 지 여부를 알려줌
    if args.arch.startswith('resnet'):
        if args.arch == 'resnet18':
            model = resnet18(num_classes=7)
        elif args.arch == 'resnet34':
            model = resnet34(num_classes=7)
        elif args.arch == 'resnet50':
            model = resnet50(num_classes=7)
        elif args.arch == 'resnet101':
            model = resnet101(num_classes=7)
        elif args.arch == 'resnet152':
            model = resnet152(num_classes=7)
        else:
            raise NotImplementedError(f"architecture {args.arch} is not implemented")
    else:
        raise NotImplementedError(f"architecture {args.arch} is not implemented")
    model = model.cuda()
    # 여러 GPU 병렬 처리
    model = torch.nn.parallel.DataParallel(model)
   # print(model)
    wandb.watch(model)
    
#     # define batch_manager - dataloader 여기 바꿔주면 될듯/batch_manager코드 분석
    # 
    transform_train = transforms.Compose([transforms.RandomRotation(40, expand=False), transforms.RandomPerspective(), transforms.ToTensor(), transforms.Normalize((0.6, 0.6, 0.6), (0.6, 0.6, 0.6))]) 
    transform_val = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.6, 0.6, 0.6), (0.6, 0.6, 0.6))])

#    train_dataset = ImageFolder(root = './data/emotion_kaggle/train')
#    validation_dataset = ImageFolder(root = './data/emotion_kaggle/test')

   
#     dataloader_train = DataLoader(train_dataset, shuffle=True, num_workers=10, batch_size=args.batch_size)
#     dataloader_val = DataLoader(validation_dataset, shuffle=False, num_workers=10, batch_size=args.batch_size)
    dataloader_train = DataLoader(classification(1, transform_train), shuffle=True, num_workers=10, batch_size=args.batch_size)
    dataloader_val = DataLoader(classification(0, transform_val), shuffle=False, num_workers=10, batch_size=args.batch_size)

   # print(f"dataloader_train: {dataloader_train}")
   # print(f"dataloader_val: {dataloader_val}")

    # LR schedule
    lr = args.lr_base
    lr_per_epoch = []
    for epoch in range(args.epochs):
        if epoch in args.lr_drop_epochs:
            lr *= args.lr_drop_rate
        lr_per_epoch.append(lr)

    # define loss and optimizer(손실 함수와 옵티마이저 SGD)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_base, momentum=0.9, weight_decay=5e-4)

    # save_path
    current_time = strftime('%Y-%m-%d_%H:%M', gmtime())
    save_dir = os.path.join(f'checkpoints/{current_time}')
    os.makedirs(save_dir,  exist_ok=True)

    # train and val
    best_perform, best_epoch = -100, -100
    for epoch in range(1, args.epochs-1):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_per_epoch[epoch]
        print(f"Training at epoch {epoch}. LR {lr_per_epoch[epoch]}")

        train.train(model, dataloader_train, criterion, optimizer, epoch=epoch)
        acc1, acc5 = val.val(model, dataloader_val, epoch=epoch)

        save_data = {'epoch': epoch,
                     'acc1': acc1,
                     'acc5': acc5,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
        
        torch.save(save_data, os.path.join(save_dir, f'{epoch:03d}.pth.tar'))
        if epoch > 1:
            os.remove(os.path.join(save_dir, f'{epoch-1:03d}.pth.tar'))
        if acc1 >= best_perform:
            torch.save(save_data, os.path.join(save_dir, 'best.pth.tar'))
            best_perform = acc1
            best_epoch = epoch
        print(f"best performance {best_perform} at epoch {best_epoch}")
        wandb.log({
        "val_acc1": acc1,
        "val_acc5": acc5
    })
