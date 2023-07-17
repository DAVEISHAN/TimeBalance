from cmath import e
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
import time
import os, sys
import numpy as np

from model import build_r3d_backbone, build_r3d50_mlp, load_r3d50_mlp

import paths as cfg
import sys, traceback
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from torch.utils.tensorboard import SummaryWriter
import cv2
from torch.utils.data import DataLoader
import math
import argparse
import random
from contrastive_loss.nt_xent_original import *
from contrastive_loss.global_local_temporal_contrastive import global_local_temporal_contrastive
from torch.cuda.amp import autocast, GradScaler
import warnings
warnings.filterwarnings("ignore")
# if torch.cuda.is_available(): 
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.backends.cudnn.benchmark = True


def train_epoch(scaler, run_id, learning_rate2, epoch, criterion, data_loader, model, optimizer, writer, use_cuda, criterion2):
    print('train at epoch {}'.format(epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate2
        writer.add_scalar('Learning Rate', learning_rate2, epoch)  

        print("Learning rate is: {}".format(param_group['lr']))
  
    losses = []
    losses_gsr_gdr, losses_ic2, losses_ic1, losses_local_local = [], [], [], []
    losses_global_local = []

    model.train()
    
    for i, (sparse_clip, dense_clip0, dense_clip1, dense_clip2, dense_clip3, a_sparse_clip, \
            a_dense_clip0, a_dense_clip1, a_dense_clip2, a_dense_clip3,_,_ ,_,_) in enumerate(data_loader):
        
        optimizer.zero_grad()

        sparse_clip = sparse_clip.permute(0,2,1,3,4) #aug_DL output is [120, 16, 3, 112, 112]], model expects [8, 3, 16, 112, 112]
        dense_clip0 = dense_clip0.permute(0,2,1,3,4)        
        dense_clip1 = dense_clip1.permute(0,2,1,3,4)
        dense_clip2 = dense_clip2.permute(0,2,1,3,4)
        dense_clip3 = dense_clip3.permute(0,2,1,3,4)

        # out_sparse will have output in this order: [sparse_clip[5], augmented_sparse_clip]
        # one element from the each of the list has 5 elements: see MLP file for details
        out_sparse = []
        # out_dense will have output in this order : [d0,d1,d2,d3,a_d0,...]
        out_dense = [[],[]]

        a_sparse_clip = a_sparse_clip.permute(0,2,1,3,4) #aug_DL output is [120, 16, 3, 112, 112]], model expects [8, 3, 16, 112, 112]
        a_dense_clip0 = a_dense_clip0.permute(0,2,1,3,4)        
        a_dense_clip1 = a_dense_clip1.permute(0,2,1,3,4)
        a_dense_clip2 = a_dense_clip2.permute(0,2,1,3,4)
        a_dense_clip3 = a_dense_clip3.permute(0,2,1,3,4)

        with autocast():

            out_sparse.append(model((sparse_clip.cuda(),'s')))
            out_sparse.append(model((a_sparse_clip.cuda(),'s')))

            out_dense[0].append(model((dense_clip0.cuda(),'d')))
            out_dense[0].append(model((dense_clip1.cuda(),'d')))
            out_dense[0].append(model((dense_clip2.cuda(),'d')))
            out_dense[0].append(model((dense_clip3.cuda(),'d')))

            out_dense[1].append(model((a_dense_clip0.cuda(),'d')))
            out_dense[1].append(model((a_dense_clip1.cuda(),'d')))
            out_dense[1].append(model((a_dense_clip2.cuda(),'d')))
            out_dense[1].append(model((a_dense_clip3.cuda(),'d')))


            criterion = NTXentLoss(device = 'cuda', batch_size=out_sparse[0][0].shape[0], temperature=params.temperature, use_cosine_similarity = False).cuda()
            criterion_local_local = NTXentLoss(device = 'cuda', batch_size=4, temperature=params.temperature, use_cosine_similarity = False).cuda()
            
            # Instance contrastive losses with the global clips (sparse clips)

            loss_ic2 = criterion(out_sparse[0][0], out_sparse[1][0])

            loss_ic1 = 0
            
            # Instance contrastive losses with the local clips (dense clips)
            for ii in range(2):
                for jj in range(2):
                    for chunk in range(1,5):
                        for chunk1 in range(1,5):
                            if (ii == jj and chunk == chunk1):
                                continue
                            loss_ic1 += criterion(out_dense[ii][chunk-1],out_dense[jj][chunk1-1])
            
            loss_ic1 /= 4 #scaling over ii and jj

            loss_local_local = 0
            # print(out_dense[0][0].shape) # this prints shape of [4,128]
            # print(torch.stack(out_dense[0],dim=1).shape) # this prints shape of [BS, 4, 128]
            # exit()
            for ii in range(out_dense[0][0].shape[0]): #for loop in the batch size
                loss_local_local += criterion_local_local(torch.stack(out_dense[0],dim=1)[ii], torch.stack(out_dense[1],dim=1)[ii])
            
            loss_global_local=0
            for ii in range(2):
                for jj in range(2):
                    loss_global_local += criterion2(torch.stack(out_sparse[ii][1:],dim=1), torch.stack(out_dense[jj],dim=1), params.temperature)

            loss = params.ic_weight*loss_ic2 + params.ic_weight*loss_ic1 + params.tcl_weight*loss_local_local + params.tcg_weight*loss_global_local


        loss_unw = loss_ic2.item()+ loss_ic1.item() + loss_local_local.item() + loss_global_local.item()
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        losses_local_local.append(loss_local_local.item())
        losses_global_local.append(loss_global_local.item())

        losses_ic1.append(loss_ic1.item())
        losses_ic2.append(loss_ic2.item())

        
        if (i+1) % 25 == 0: 
            print(f'Training Epoch {epoch}, Batch {i}, Loss: {np.mean(losses) :.5f}')
            print(f'Training Epoch {epoch}, Batch {i}, losses_local_local: {np.mean(losses_local_local) :.5f}')
            print(f'Training Epoch {epoch}, Batch {i}, losses_global_local: {np.mean(losses_global_local) :.5f}')
            print(f'Training Epoch {epoch}, Batch {i}, losses_ic2: {np.mean(losses_ic2) :.5f}')
            print(f'Training Epoch {epoch}, Batch {i}, losses_ic1: {np.mean(losses_ic1) :.5f}')

        # exit()
    print('Training Epoch: %d, Loss: %.4f' % (epoch,  np.mean(losses)))
    writer.add_scalar('Training Loss', np.mean(losses), epoch)
    writer.add_scalar('losses_local_local', np.mean(losses_local_local), epoch)
    writer.add_scalar('losses_global_local', np.mean(losses_global_local), epoch)
    writer.add_scalar('losses_ic2', np.mean(losses_ic2), epoch)
    writer.add_scalar('losses_ic1', np.mean(losses_ic1), epoch)
    
    del out_sparse, out_dense, loss, loss_ic2, loss_ic1, losses_local_local, loss_global_local

    return model, np.mean(losses), scaler


def train_classifier(run_id, restart, saved_model, linear, params, devices):   
# def train_classifier(run_id, restart):
    use_cuda = True
    best_score = 0
    writer = SummaryWriter(os.path.join(cfg.logs, str(run_id)))
    
    writer.add_text('Run ID', str(run_id), 0)
    writer.add_text('Backbone', str(params.backbone), 0)
    writer.add_text('RGB', str(params.RGB), 0)
    writer.add_text('Normalize', str(params.normalize), 0)
    writer.add_text('Optimizer', str(params.opt_type), 0)
    writer.add_text('Frozen Backbone', str(params.frozen_bb), 0)
    writer.add_text('Frozen BatchNorm', str(params.frozen_bn), 0)
    writer.add_scalar('Learning Rate', params.learning_rate)
    writer.add_scalar('Batch Size', params.batch_size)
    writer.add_scalar('Patience', params.scheduler_patience)
    for item in dir(params):
        if '__' not in item:
            # if (saved_model is not None) and ('pretrained_checkpoint' in item):
            #     continue
            print(f'{item} =  {params.__dict__[item]}') 


    save_dir = os.path.join(cfg.saved_models_dir, run_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if restart:
        saved_model_file = save_dir + '/model_temp.pth'
        
        try:
            model = load_r3d50_mlp(saved_model_file= saved_model_file)
            epoch0 = torch.load(saved_model_file)['epoch']
            learning_rate1 = torch.load(saved_model_file)['learning_rate']
            best_score= torch.load(saved_model_file)['best_score'] 
            scheduler_epoch = torch.load(saved_model_file)['scheduler_epoch'] 
            scaler = torch.load(saved_model_file)['amp_scaler'] 

        except:
            print(f'No such model exists: {saved_model_file} :(')
            epoch0 = 0 
            model = build_r3d50_mlp()
            scheduler_epoch = 0
            best_score = 10000
            learning_rate1 = params.learning_rate
            scaler = GradScaler()
    
    
    else:
        epoch0 = 0 
        if saved_model is not None: 
            model = load_r3d50_mlp(saved_model_file= saved_model, avoid_layer4 = params.avoid_layer4, avoid_layer3 = params.avoid_layer3)
        else:
            model = build_r3d50_mlp()
        scheduler_epoch = 0
        best_score = 10000
        scaler = GradScaler()

        learning_rate1 = params.learning_rate
    print(f'Starting learning rate {learning_rate1}')
    print(f'Scheduler_epoch {scheduler_epoch}')
    print(f'Best score till now is {best_score}')

    criterion = NTXentLoss(device = 'cuda', batch_size=params.batch_size, temperature=params.temperature, use_cosine_similarity = False)

    device_name =  'cuda:' + str(devices[0]) # This is used to move the data no matter if it is multigpu
    print(f'Device name is {device_name}')
    if len(devices)>1:
        print(f'Multiple GPUS found!')
        # model=nn.DataParallel(model)
        model = torch.nn.DataParallel(model, device_ids=devices)
        
        model.cuda()

    else:
        print('Only 1 GPU is available')
        
        model.to(device=torch.device(device_name))
        
        # model.cuda()
        criterion.to(device=torch.device(device_name))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate1, weight_decay = params.weight_decay)

    if restart:
        saved_model_file = save_dir + '/model_temp.pth'
        
        # try:
        if os.path.exists(saved_model_file):
            optimizer.load_state_dict(torch.load(saved_model_file)['optimizer'])
        else:
            print(f'File {saved_model_file} dne')
        # except:

        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    train_dataset = dl_ssl_gen(params = params, dataset = params.dataset, shuffle = True, data_percentage = params.data_percentage)
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn2)
    print(f'Train dataset length: {len(train_dataset)}')
    print(f'Train dataset steps per epoch: {len(train_dataset)/params.batch_size}')
   
    learning_rate2 = learning_rate1 
    scheduler_step = 1         

    for epoch in range(epoch0, params.num_epochs):
        print(f'Epoch {epoch} started')
        if epoch < params.warmup:
            learning_rate2 = params.warmup_array[epoch]*params.learning_rate

        if scheduler_epoch == params.scheduler_patience:
            print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
            print(f'Dropping learning rate to {learning_rate2/params.drop_factor} for epoch')
            print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
            learning_rate2 = learning_rate1/(params.drop_factor**scheduler_step)
            scheduler_epoch = 0
            scheduler_step += 1


        start=time.time()
        try:
            model, loss, scaler = train_epoch(scaler, run_id, learning_rate2, epoch, criterion, train_dataloader, model, optimizer, writer, use_cuda, criterion2 = global_local_temporal_contrastive)
                        
            if loss < best_score:
                print('++++++++++++++++++++++++++++++')
                print(f'Epoch {epoch} is the best model till now for {run_id}!')
                print('++++++++++++++++++++++++++++++')
                save_dir = os.path.join(cfg.saved_models_dir, run_id)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_file_path = os.path.join(save_dir, 'model_best_e{}_loss_{}.pth'.format(epoch, str(loss)[:6]))
                states = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'learning_rate': learning_rate2,
                    'amp_scaler': scaler,

                }
                torch.save(states, save_file_path)
                best_score = loss
                scheduler_epoch = 0
            elif epoch % 1 == 0:
                save_dir = os.path.join(cfg.saved_models_dir, run_id)
                save_file_path = os.path.join(save_dir, 'model_e{}_loss_{}.pth'.format(epoch, str(loss)[:6]))
                states = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'learning_rate': learning_rate2,
                    'amp_scaler': scaler,
                }
                torch.save(states, save_file_path)

            if loss > best_score:
                scheduler_epoch += 1
            
            save_dir = os.path.join(cfg.saved_models_dir, run_id)
            save_file_path = os.path.join(save_dir, 'model_temp.pth')
            states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate2,
                'best_score': best_score,
                'scheduler_epoch': scheduler_epoch,
                'amp_scaler': scaler,
            }
            torch.save(states, save_file_path)
        except:
            print("Epoch ", epoch, " failed")
            print('-'*60)
            traceback.print_exc(file=sys.stdout)
            print('-'*60)
            continue

        train_dataset = dl_ssl_gen(params = params, dataset = params.dataset, shuffle = True, data_percentage = params.data_percentage)
        train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn2)
        print(f'Train dataset length: {len(train_dataset)}')
        print(f'Train dataset steps per epoch: {len(train_dataset)/params.batch_size}')
       
        taken = time.time()-start
        print(f'Time taken for Epoch-{epoch} is {taken}')
        print()

if __name__ == '__main__':
    import argparse, importlib

    parser1 = argparse.ArgumentParser(description='Script to do self-supervised pretraining of teachers or student')

    parser1.add_argument("--run_id", dest='run_id', type=str, required=False, default= "dummy",
                        help='run_id')
    parser1.add_argument("--restart", action='store_true')
    parser1.add_argument("--saved_model", dest='saved_model', type=str, required=False, default= None,
                        help='run_id')
    parser1.add_argument("--linear", action='store_true')
    parser1.add_argument("--config", dest='config_file_location', type=str, required=True, default= "conf_ssl",
                        help='config_file_location')
    parser1.add_argument("--devices", dest='devices', action='append', type =int, required=False, default=None,
                        help='devices should be a list even when it is single')
      
    args = parser1.parse_args()
    print(f'Restart {args.restart}')
    
    config_filename = args.config_file_location.replace('.py', '')
    if os.path.exists(config_filename + '.py'):
        params = importlib.import_module(config_filename)
        print(f' {config_filename} is loaded as params')
    else:
        print(f'{config_filename} dne, give it correct path!')
        
    from dataloaders_july22.dl_ssl import *
    
    
    run_id = args.run_id
    saved_model = args.saved_model
    linear = args.linear
    devices = args.devices
    if devices is None: 
        devices = list(range(torch.cuda.device_count()))
    
    print(f'devices are {devices}') 
    # exit()
    if saved_model is not None and len(saved_model):
        saved_model = '/' +saved_model
        
    else:
        saved_model = params.pretrained_checkpoint

    # if saved_model is not None and len(saved_model):
        # saved_model = saved_model.replace('-symlink', '')
        # saved_model = saved_model.replace('some_unneccessary_string', '')

    train_classifier(str(run_id), args.restart, saved_model, linear, params, devices)




        


