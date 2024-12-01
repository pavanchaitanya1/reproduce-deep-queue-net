from util.argparser import argparser
import os
from config import BaseConfig
from DQN_dataset import DataFromH5File
import torch
import torch.utils.data as data
from model import DeepQueueNet
import torch.nn as nn
from train import train_DQN
import numpy as np
from val import val_DQN
import shutil

def save_checkpoint(state, is_best, checkpoint,filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def weight_init(m):
    if isinstance(m, nn.Linear):
        # nn.init.xavier_normal_(m.weight)
        # nn.init.constant_(m.bias, 0)
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LSTM):
        nn.init.orthogonal_(m.weight_ih_l0)
        nn.init.orthogonal_(m.weight_hh_l0)
        nn.init.constant_(m.bias_ih_l0, 0)
        nn.init.constant_(m.bias_hh_l0, 0)


if __name__ == "__main__":
    params = argparser()
    os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    params['save_dir'] = "../checkpoints"
    params['train_file'] = "../data/train.h5"
    params['test1_file'] = "../data/test1.h5"
    params['test2_file'] = "../data/test2.h5"
    params['mode'] = 1
    config = BaseConfig()
    torch.manual_seed(config.seed)
    if not os.path.exists(params['save_dir']):
        os.mkdir(params['save_dir'])
    if not os.path.exists("../logs"):
        os.mkdir("../logs")
    if params['mode']==1: #intial training
        train_file = params['train_file']
        test1_file = params['test1_file']
        test2_file = params['test2_file']

        print('Data Loading')

        epochs = config.epochs
        trainset = DataFromH5File(train_file)
        testset_1 = DataFromH5File(test1_file)
        testset_2 = DataFromH5File(test2_file)

        train_loader = data.DataLoader(dataset=trainset, batch_size=config.BATCH_SIZE, shuffle=True,  num_workers=0)
        test_loader_1 = data.DataLoader(dataset=testset_1, batch_size=config.BATCH_SIZE, shuffle=True,  num_workers=0)
        test_loader_2 = data.DataLoader(dataset=testset_2, batch_size=config.BATCH_SIZE, shuffle=True,  num_workers=0)

        model = DeepQueueNet(config, device)
        model.to(device)
        '''
        for name, parameters in model.named_parameters():
            print(name, ':', parameters.size())        
        '''

        model.apply(weight_init)

        print('Start Training Epochs: {}'.format(epochs))

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'],weight_decay=config.l2)
        best_loss = float("inf")

        train_log = "../logs/train_100epochs_l2_bilstm.log"
        test1_log = "../logs/test1_100epochs_l2_bilstm.log"
        test2_log = "../logs/test2_100epochs_l2_bilstm.log"
        for epoch in range(1,epochs+1):
            print('Epoch: {}'.format(epoch))
            train_loss = train_DQN(train_loader,device,model,loss_fn,optimizer,epoch,train_log)
            test1_loss = val_DQN(test_loader_1,device,model,loss_fn,epoch, test1_log)
            test2_loss = val_DQN(test_loader_1,device,model,loss_fn,epoch, test2_log)
            is_best = test1_loss+test2_loss < best_loss
            best_loss = min(test1_loss+test2_loss, best_loss)
            state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'loss': test1_loss+test2_loss,
            'optimizer': optimizer.state_dict(),
            }
            if is_best:
                filename="best_epoch_{}.ckpt".format(epoch)
                filepath = os.path.join(params['save_dir'], filename)
                torch.save(state, filepath)

                print('Best loss:')
                print(test1_loss+test2_loss)
            print_str = 'Epoch: [{0}]\t Train loss {1}\t' \
                            ' Test_1 loss {2}\t' \
                            ' Test_2 loss {3}\t'\
                    .format(
                    epoch,
                    train_loss,
                    test1_loss,
                    test2_loss
                )
            print(print_str)
