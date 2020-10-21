# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 17:26:00 2020

@author: Antony
"""

import os
import time
import math
import numpy as np
import argparse

# import torchsnooper
import torch
import torch.nn as nn
from torch.autograd import Variable
import resnet50

from food101 import get_data_loader
from model import InceptionV2 as model

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def main():
    
    parser = argparse.ArgumentParser()
    # basic config
    parser.add_argument('--f101_image_dir', type=str, default='data/food-101/images') # validate and test on food101
    parser.add_argument('--f101n_image_dir', type=str, default='data/Food-101N_release/images') # train on food101n
    parser.add_argument('--f101_train_label', type=str, default='data/food-101/meta/train.txt') # validation
    parser.add_argument('--f101n_train_label', type=str, default='data/Food-101N_release/meta/imagelist.tsv') # train
    parser.add_argument('--stage1_train_label', type=str, default='data/Food-101N_release/meta/list1_res.tsv') # train
    parser.add_argument('--test_label', type=str, default='data/food-101/meta/test.txt') # test
    parser.add_argument('--model_save_path', type=str, default='saved_model') 
    parser.add_argument('--task_name', type=str, required=True, help='choose from below: 101, 101n, stage and stage_att') 
    parser.add_argument('--model_save_date', type=str, required=True) # e.g. 'July29'
    parser.add_argument('--model_name', type=str, default='resnet50')
    parser.add_argument('--pretrained', type=str, default=True)
    parser.add_argument('--attention', type=str, default=False)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--num_classes', type=int, default=101)
    # training config
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--optimizer_mode', type=str, default='Adam')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--grad_accumulation', type=str, default=False)
    parser.add_argument('--lr_reduce_mode', type=str, default='step', help='step or adaptive')
    parser.add_argument('--lr_reduce_epoch', type=int, default=10, help='if args.ReduceLROnPlateau, this mean patience')
    parser.add_argument('--lr_reduce_ratio', type=float, default=0.1)
    parser.add_argument('--num_epoch', type=int, default=200)
    
    args = parser.parse_args()
    print(str(args),'\n')
    
    if args.model_name == 'inception_v2':
        net = model(args)
    elif args.model_name == 'resnet50':
        net = resnet50.resnet50(pretrained=args.pretrained, attention=args.attention) 
    
    train_op(net, args)
    

def train_op(net, args):

    use_gpu = torch.cuda.is_available()
    print('use_gpu:', use_gpu)
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    net = net.to(device)
    
    if args.task_name == '101': 
        train_loader, val_loader = get_data_loader(args.f101_image_dir, args.f101_train_label, validation_split=args.val_ratio, batch_size=args.batch_size, is_split=True)
    elif args.task_name == '101n':
        train_loader = get_data_loader(args.f101n_image_dir, args.f101n_train_label, validation_split=args.val_ratio, batch_size=args.batch_size, is_split=False)
        val_loader = get_data_loader(args.f101_image_dir, args.f101_train_label, batch_size=args.batch_size, is_split=False)
    elif args.task_name == 'stage':
        train_loader = get_data_loader(args.f101n_image_dir, args.stage1_train_label, validation_split=args.val_ratio, batch_size=args.batch_size, is_split=False)
        val_loader = get_data_loader(args.f101_image_dir, args.f101_train_label, batch_size=args.batch_size, is_split=False)
    test_loader = get_data_loader(args.f101_image_dir, args.test_label, batch_size=args.batch_size, is_split=False)
    print('numbers of training set: {}'.format(len(train_loader)*args.batch_size))
    print('numbers of validation set: {}'.format(len(val_loader)*args.batch_size))
    print('numbers of test set: {}\n'.format(len(test_loader)*args.batch_size))
    
    # define evaluation metric for early stopping
    best_top1_acc, best_top5_acc, best_epoch, adapt_epoch = 0, 0, 0, 0

    if args.optimizer_mode == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=args.learning_rate, momentum=0.9, weight_decay=1e-3)
    elif args.optimizer_mode == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(),
                                     lr=args.learning_rate)

    #if args.StepLR:
    #    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_reduce_epoch*len(train_loader), gamma=args.lr_reduce_ratio)  # the step_size refers the index of the current loop
    #else:
    #    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.lr_reduce_ratio, patience=args.lr_reduce_epoch*len(train_loader), verbose=True)

    for epoch in range(args.num_epoch):
        start_time = time.clock()
        print('training epoch starts at: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
        for batch_ind, (data, target) in enumerate(train_loader):

            data, target = Variable(data), Variable(target)
            data, target = data.to(device), target.to(device)
            torch.set_grad_enabled(True)
            net.train()
            
            output = net(data)
            target = target.long()
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target)          

            if args.grad_accumulation:
                # grad accumulation
                loss = loss / 4
                loss.backward()
                if batch_ind % 4 == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss = loss.cpu() if use_gpu else loss
            # compute train correct number in epoch
            t_pred = torch.max(output, dim=1)[1].cpu().numpy()
            t_label = target.data.cpu().numpy()
            t_pred_num = (t_pred == t_label).sum()

            if batch_ind % (len(train_loader)//5) == 0:
                print('epoch: {}, iteration: {}, loss: {:.6f}, predict/correct: {}/{}'.format(epoch, batch_ind, loss, t_pred_num, args.batch_size))

        epoch_training_time = time.clock() - start_time
        print('training epoch ends at: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
        print('epoch: {}, loss: {:.6f}, epoch training time: {:.2f} min'.format(epoch, loss, epoch_training_time/60))
        print('current learning rate:', optimizer.param_groups[0]['lr'])

        if epoch % 1 == 0:
            val_start_time = time.clock()
            val_loss_list_local = []
            top1_val_correct = 0  # add correct predicted number to it to calculate accuracy
            top5_val_correct = 0  # add correct predicted number to it to calculate accuracy
            torch.set_grad_enabled(False)
            net.eval()
            for val_data, val_target in val_loader:
                val_data, val_target = Variable(val_data), Variable(val_target)
                if use_gpu:
                    val_data, val_target = val_data.to(device), val_target.to(device)
                    val_output = net(val_data)
                    val_target = val_target.long()

                    # compute validation loss
                    val_loss = criterion(val_output, val_target)
                    val_loss = val_loss.cpu() if use_gpu else val_loss
                    val_loss = val_loss.data.numpy()
                    val_loss_list_local.append(val_loss)

                    # compute validation correct in epoch
                    correct_top1, correct_top5 = compute_correct_number(val_output, val_target)
                    top1_val_correct += correct_top1
                    top5_val_correct += correct_top5
            # compute total accuracy
            top1_val_acc = top1_val_correct / (args.batch_size * len(val_loader))
            top5_val_acc = top5_val_correct / (args.batch_size * len(val_loader))
            val_avg_loss = np.mean(val_loss_list_local)
            val_end_time = time.clock() - val_start_time
            
            if top1_val_acc > best_top1_acc:
                best_top1_acc, best_epoch, adapt_epoch = top1_val_acc, epoch, epoch
            
            print('validation ends at: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
            # display the best validation performance
            print('The best epoch is {}, with accuracy {:8f}, adapt epoch is {}.'.format(best_epoch, best_top1_acc[0], adapt_epoch))
            print('epoch: {}, (valid) top1 acc: {:.8f}, (valid) top5 acc: {:.8f}, (valid) loss: {:.4f}, validation time: {:.2f} min\n'.format(epoch, top1_val_acc[0], top5_val_acc[0],  val_avg_loss, val_end_time/60))

            # learning rate adjustmentm (some logical errors in adaptive method?)
            lr = optimizer.param_groups[0]['lr']
            if args.lr_reduce_mode == 'step' and (epoch+1) % args.lr_reduce_epoch == 0:
                optimizer.param_groups[0]['lr'] = args.lr_reduce_ratio * lr
            elif args.lr_reduce_mode == 'adaptive' and epoch - args.lr_reduce_epoch > adapt_epoch:
                optimizer.param_groups[0]['lr'] = args.lr_reduce_ratio * lr
                adapt_epoch= epoch
            elif args.lr_reduce_mode == 'milestone':
                ms_list = [20, 25, 30, 35, 40, 45, 50]
                if (epoch+1) in ms_list:
                    optimizer.param_groups[0]['lr'] = args.lr_reduce_ratio * lr

            # early stopping
            if epoch - 20 > best_epoch:
                # compute test set performance
                print('test starts at: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
                test_start_time = time.clock()
                torch.set_grad_enabled(False)
                top1_test_correct = 0   
                top5_test_correct = 0   
                net.eval()
                for test_data, test_target in test_loader:
                    test_data, test_target = Variable(test_data), Variable(test_target)
                    if use_gpu:
                        test_data, test_target = test_data.to(device), test_target.to(device)
                        test_output = net(test_data)
                        test_target = test_target.long()

                        # compute test correct in epoch
                        correct_top1, correct_top5 = compute_correct_number(test_output, test_target)
                        top1_test_correct += correct_top1
                        top5_test_correct += correct_top5
                # compute total accuracy
                top1_test_acc = top1_test_correct / (args.batch_size * len(test_loader))
                top5_test_acc = top5_test_correct / (args.batch_size * len(test_loader))

                test_end_time = time.clock() - test_start_time
                print('test ends at: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
                print('epoch: {}, (test) top1 acc: {:.8f}, (test) top5 acc: {:.8f}, test time: {:.2f} min'.format(best_epoch, top1_test_acc[0], top5_test_acc[0], test_end_time/60))

                # model saving
                state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': best_epoch}
                print('Saving model ...')
                model_saved_path = os.path.join(args.model_save_path, args.task_name, args.model_save_date)
                if not os.path.exists(model_saved_path ):
                    os.makedirs(model_saved_path)
                torch.save(state, os.path.join(model_saved_path, '{}_epoch_{}_acc_{:.3f}_{:.3f}.pkl'.format(args.model_name, best_epoch, top1_test_acc[0], top5_test_acc[0])))
                print('Model was just saved in {}'.format(model_saved_path))
                break

def compute_correct_number(output, label):
    _, pred = output.topk(5, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))
    correct_1 = correct[:1].view(-1).float().sum(0, keepdim=True)
    correct_5 = correct[:5].view(-1).float().sum(0, keepdim=True)
    return correct_1.cpu().numpy(), correct_5.cpu().numpy()


if __name__ == '__main__':
    main()

    # print(net)


