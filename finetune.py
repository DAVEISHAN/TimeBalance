import torch, torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
import time
import os
import numpy as np
import sys
    
import paths as cfg
import sys, traceback
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from torch.utils.tensorboard import SummaryWriter

import cv2
from torch.utils.data import DataLoader
import math
import argparse
import itertools

from models_july22.model import *
from torch.cuda.amp import autocast, GradScaler


# if torch.cuda.is_available(): 
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.backends.cudnn.benchmark = True


def train_epoch(run_id, learning_rate2,  epoch, data_loader, model, criterion, optimizer, writer, use_cuda, scaler,device_name):
    print('train at epoch {}'.format(epoch))
    for param_group in optimizer.param_groups:
        param_group['lr']=learning_rate2
        writer.add_scalar('Learning Rate', learning_rate2, epoch)  

        print("Learning rate is: {}".format(param_group['lr']))
  
  
    losses, weighted_losses = [], []
    loss_mini_batch = 0
    predictions, gt = [], []


    model.train()

    for i, (inputs, label, vid_path, frameids) in enumerate(data_loader):

        optimizer.zero_grad()

        inputs = inputs.permute(0,2,1,3,4)
        if params.RGB or params.normalize:
            inputs = torch.flip(inputs, [1]) 
        
        if params.normalize: 

            inputs = inputs.permute(0,2,1,3,4)
            
            inputs_shape = inputs.shape
            inputs = inputs.reshape(inputs_shape[0]*inputs_shape[1], inputs_shape[2], inputs_shape[3], inputs_shape[4])
            inputs = torchvision.transforms.functional.normalize(inputs, mean =(0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
            inputs = inputs.reshape(inputs_shape)
            
            inputs = inputs.permute(0,2,1,3,4)

       
        if use_cuda:
            inputs = inputs.to(device=torch.device(device_name))
            label = torch.from_numpy(np.asarray(label)).to(device=torch.device(device_name))
            frameids = frameids.to(device=torch.device(device_name))
        
        with autocast(): 

            output = model(inputs)
            
            loss = criterion(output,label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            predictions.extend(torch.max(output, axis=1).indices.cpu().numpy())
            gt.extend(label.cpu().numpy())
        losses.append(loss.item())
        if i % 24 == 0: 
            print(f'Training Epoch {epoch}, Batch {i}, Loss: {np.mean(losses) :.5f}', flush=True)
        
    print('Training Epoch: %d, Loss: %.4f' % (epoch,  np.mean(losses)))
    writer.add_scalar('Training Loss', np.mean(losses), epoch)
    
    predictions = np.asarray(predictions)
    gt = np.asarray(gt)
        
        
    accuracy = ((predictions == gt).sum())/np.size(predictions)
    print(f'Training Accuracy at Epoch {epoch} is {accuracy*100 :0.3f}')
    writer.add_scalar('Training Accuracy', accuracy, epoch)
    
    

    del loss, inputs, output, label, frameids

    return model, np.mean(losses), scaler

def val_epoch(run_id, epoch,mode, crop_fac, pred_dict,label_dict, data_loader, model, criterion, writer, use_cuda,device_name):
    print(f'validation at epoch {epoch} - mode {mode} ')
    
    model.eval()

    losses = []
    predictions, ground_truth = [], []
    vid_paths = []

    for i, (inputs, label, vid_path, frameids) in enumerate(data_loader):
        vid_paths.extend(vid_path)
        ground_truth.extend(label)
        if len(inputs.shape) != 1:

            inputs = inputs.permute(0,2,1,3,4)
            if params.RGB or params.normalize:
                inputs = torch.flip(inputs, [1]) 


            if params.normalize:
                inputs = inputs.permute(0,2,1,3,4)

                inputs_shape = inputs.shape
                inputs = inputs.reshape(inputs_shape[0]*inputs_shape[1], inputs_shape[2], inputs_shape[3], inputs_shape[4])
                inputs = torchvision.transforms.functional.normalize(inputs, mean =(0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
                inputs = inputs.reshape(inputs_shape)
            
                inputs = inputs.permute(0,2,1,3,4)
            if use_cuda:
                inputs = inputs.to(device=torch.device(device_name))
                label = torch.from_numpy(np.asarray(label)).to(device=torch.device(device_name))
                frameids = frameids.to(device=torch.device(device_name))

        
            with torch.no_grad():
                # with autocast(): 

                output = model(inputs)

                loss = criterion(output,label)

            losses.append(loss.item())


            predictions.extend(nn.functional.softmax(output, dim = 1).cpu().data.numpy())


            if i+1 % 45 == 0:
                print("Validation Epoch ", epoch , "mode", mode, "crop_fac", crop_fac, " Batch ", i, "- Loss : ", np.mean(losses))
        
    del inputs, output, label, loss 

    ground_truth = np.asarray(ground_truth)
    pred_array = np.flip(np.argsort(predictions,axis=1),axis=1) 
    c_pred = pred_array[:,0] 

    for entry in range(len(vid_paths)):
        if str(vid_paths[entry].split('/')[-1]) not in pred_dict.keys():
            pred_dict[str(vid_paths[entry].split('/')[-1])] = []
            pred_dict[str(vid_paths[entry].split('/')[-1])].append(predictions[entry])

        else:
            # print('yes')
            pred_dict[str(vid_paths[entry].split('/')[-1])].append(predictions[entry])

    for entry in range(len(vid_paths)):
        if str(vid_paths[entry].split('/')[-1]) not in label_dict.keys():
            label_dict[str(vid_paths[entry].split('/')[-1])]= ground_truth[entry]

    print_pred_array = []

    
    
    correct_count = np.sum(c_pred==ground_truth)
    accuracy = float(correct_count)/len(c_pred)
    
    # print(f'Correct Count is {correct_count}')
    print(f'Epoch {epoch}, mode {mode}, crop_fac {crop_fac}, Accuracy: {accuracy*100 :.3f}')
    return pred_dict, label_dict, accuracy, np.mean(losses)
    
def train_classifier(run_id, restart, saved_model, linear, params, devices):
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
            print(f'{item} =  {params.__dict__[item]}')  

    
    save_dir = os.path.join(cfg.saved_models_dir, run_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    
    
    if saved_model is not None:
        saved_model_file = saved_model
    else:
        saved_model_file = params.pretrained_checkpoint
    

    
    if restart:
        if params.backbone == 'R3D18':
            model = load_r3d_classifier(arch = params.backbone, num_classes = params.num_classes, saved_model_file = saved_model_file)
    else:
        if params.backbone == 'videomae_ucf101':
            model = videomae_vit(pretraining = 'ucf101', num_classes= params.num_classes, retrieval = False, num_frames = params.num_frames, num_segments = 1)
        elif params.backbone == 'videomae_hmdb51':
            model = videomae_vit(pretraining = 'hmdb51', num_classes= params.num_classes, retrieval = False, num_frames = params.num_frames, num_segments = 1)    
        else:
            model = build_r3d_classifier(arch = params.backbone, saved_model_file = saved_model_file, num_classes = params.num_classes)

    
    scaler = GradScaler()
    
        
    if params.frozen_bn:
        frozen_bn(model)
    
    if params.frozen_bb:
        for name, param in model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
    
    if params.linear:
        print('Its linear evaluation')
        for name, param in model.named_parameters():
            
            if not ('final_class_fc' in name or 'twoD_fc' in name or 'twoThree_fc' in name  or 'model.head' in name): ########### 'head' addition require a verification later for linear 
                param.requires_grad = False
            else:
                print(f'Kept unfrozen {name}')
    
    
    epoch0 = 0
    

    learning_rate1 = params.learning_rate
    best_score = 10000

    
    criterion= nn.CrossEntropyLoss()
     
    
    # if torch.cuda.device_count()>1:
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
        criterion.to(device=torch.device(device_name))
    
    if params.opt_type == 'adam':
        optimizer = optim.Adam(model.parameters(),lr=params.learning_rate)
    elif params.opt_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9)
    elif params.opt_type == 'adamW':
        optimizer = optim.AdamW(model.parameters(),lr=params.learning_rate, weight_decay=1e-8)
    else:
        raise NotImplementedError(f"not supporting {params.opt_type}")
    
    train_dataset = baseline_dataloader_train_strong(params = params, shuffle = False, data_percentage = params.data_percentage, dl_mode= params.dl_mode)

    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn_train)
    print(f'Train dataset length: {len(train_dataset)}')
    print(f'Train dataset steps per epoch: {len(train_dataset)/params.batch_size}')
    
    #We will do validation only at epochs mentioned in the following array
   
    val_array =[x*2 for x in range(0,params.num_epochs)]
    if params.lr_scheduler == "patience_based":
        # val_array = [10,20,30] + [x*2 for x in range(30,100)]
        val_array =[x*2 for x in range(0,params.num_epochs)]

    if params.linear:
        val_array = list(range(0,20,5)) + list(range(20,params.num_epochs,5))
    # val_array = [x*params.val_low_freq for x in range(0,int(params.dense_val_after_epoch/params.val_low_freq))]
    
    
    if params.data_percentage != 1.0:
        val_array = [x*50 for x in range(0,50)]

    val_array = params.val_array

    modes = list(range(params.num_modes))
    crop_facs = params.cropping_facs# [0.7, 0.8, 1.0] #because of low cpu memory, val features itself toakes 75G!

   
    print(f'Num modes {len(modes)}')
   
    accuracy = 0
    best_acc = 0 

    learning_rate2 = learning_rate1 
    scheduler_step = 1  
    scheduler_epoch = 0
    train_loss = 1000


    for epoch in range(epoch0, params.num_epochs):
        
        print(f'Epoch {epoch} started')
        start=time.time()

        # try:
        if params.lr_scheduler == "cosine":
            learning_rate2 = params.cosine_lr_array[epoch]*learning_rate1
        elif params.warmup and epoch < len(params.warmup_array):
            learning_rate2 = params.warmup_array[epoch]*learning_rate1
        elif params.lr_scheduler == "loss_based":
            if train_loss < 0.8 and train_loss>=0.5:
                learning_rate2 = learning_rate1/2
            elif train_loss <0.4:
                learning_rate2 = learning_rate1/10
            elif train_loss <0.25:
                learning_rate2 = learning_rate1/20    
            elif train_loss <0.20:
                learning_rate2 = learning_rate1/100
            elif train_loss <0.05:
                learning_rate2 = learning_rate1/1000                            
        elif params.lr_scheduler == "patience_based":
            if scheduler_epoch == params.scheduler_patience:
                print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
                print(f'Dropping learning rate to {learning_rate2/params.lr_reduce_factor} for epoch')
                print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
                learning_rate2 = learning_rate1/(params.lr_reduce_factor**scheduler_step)
                scheduler_epoch = 0
                scheduler_step += 1
        

        model, train_loss, scaler = train_epoch(run_id, learning_rate2,  epoch, train_dataloader, model, criterion, optimizer, writer, use_cuda, scaler,device_name)
        if train_loss < best_score:
            # scheduler_epoch += 1

            best_score = train_loss
            scheduler_epoch = 0
        else:
            scheduler_epoch+=1
            

        if epoch in val_array:
            pred_dict = {}
            label_dict = {}
            val_losses =[]
            val_iter = 0
            for crop_fac in crop_facs:
                for mode in modes:
                    try:
                        validation_dataset = multi_baseline_dataloader_val_strong(params = params, shuffle = True, data_percentage = params.data_percentage,\
                            mode = mode, cropping_factor = crop_fac, total_num_modes = params.num_modes)
                        validation_dataloader = DataLoader(validation_dataset, batch_size=params.v_batch_size, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn2)
                        if val_iter ==0:
                            print(f'Validation dataset length: {len(validation_dataset)}')
                            print(f'Validation dataset steps per epoch: {len(validation_dataset)/params.v_batch_size}')    
                        pred_dict, label_dict, accuracy, loss = val_epoch(run_id, epoch,mode, crop_fac, pred_dict, label_dict, validation_dataloader, model, criterion, writer, use_cuda,device_name)
                        val_losses.append(loss)

                        predictions1 = np.zeros((len(list(pred_dict.keys())), params.num_classes))
                        ground_truth1 = []
                        entry = 0
                        for key in pred_dict.keys():
                            predictions1[entry] = np.mean(pred_dict[key], axis =0)
                            entry+=1

                        for key in label_dict.keys():
                            ground_truth1.append(label_dict[key])

                        pred_array1 = np.flip(np.argsort(predictions1,axis=1),axis=1) # Prediction with the most confidence is the first element here
                        c_pred1 = pred_array1[:,0]

                        correct_count1 = np.sum(c_pred1==ground_truth1)
                        accuracy11 = float(correct_count1)/len(c_pred1)


                        print(f'Running Avg Accuracy is for epoch {epoch}, mode {mode}, crop_fac {crop_fac}, is {accuracy11*100 :.3f}% ')  
                    except:
                        print(f'Failed epoch {epoch}, mode {mode}, crop_fac {crop_fac}, is {accuracy11*100 :.3f}% ')  
                    val_iter+=1

            val_loss = np.mean(val_losses)
            predictions = np.zeros((len(list(pred_dict.keys())),params.num_classes))
            ground_truth = []
            entry = 0
            for key in pred_dict.keys():
                predictions[entry] = np.mean(pred_dict[key], axis =0)
                entry+=1

            for key in label_dict.keys():
                ground_truth.append(label_dict[key])

            pred_array = np.flip(np.argsort(predictions,axis=1),axis=1) # Prediction with the most confidence is the first element here
            c_pred = pred_array[:,0]

            correct_count = np.sum(c_pred==ground_truth)
            accuracy1 = float(correct_count)/len(c_pred)
            print(f'Val loss for epoch {epoch} is {np.mean(val_losses)}')
            print(f'Correct Count is {correct_count} out of {len(c_pred)}')
            writer.add_scalar('Validation Loss', np.mean(val_loss), epoch)
            writer.add_scalar('Validation Accuracy', np.mean(accuracy1), epoch)
            print(f'Overall Accuracy is for epoch {epoch} is {accuracy1*100 :.3f}% ')

            accuracy = accuracy1
        
        save_dir = os.path.join(cfg.saved_models_dir, run_id)

        if accuracy > best_acc:
            print('++++++++++++++++++++++++++++++')
            print(f'Epoch {epoch} is the best model till now for {run_id}!')
            print('++++++++++++++++++++++++++++++')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_file_path = os.path.join(save_dir, 'model_{}_bestAcc_{}.pth'.format(epoch, str(accuracy)[:6]))
            states = {
                'epoch': epoch + 1,
                # 'arch': params.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp_scaler': scaler,
            }
            torch.save(states, save_file_path)
            best_acc = accuracy
        # else:
        # if linear:
        #     save_dir = os.path.join('linear', run_id)
        save_file_path = os.path.join(save_dir, 'model_temp.pth')
        
        states = {
            'epoch': epoch + 1,
            # 'arch': params.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'amp_scaler': scaler,
        }
        torch.save(states, save_file_path)
            
        # except:
        #     print("Epoch ", epoch, " failed")
        #     print('-'*60)
        #     traceback.print_exc(file=sys.stdout)
        #     print('-'*60)
        #     continue
        taken = time.time()-start
        print(f'Time taken for Epoch-{epoch} is {taken}')
        print()
        train_dataset = baseline_dataloader_train_strong(params = params, shuffle = False, data_percentage = params.data_percentage)
        train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn_train)
        print(f'Train dataset length: {len(train_dataset)}')
        print(f'Train dataset steps per epoch: {len(train_dataset)/params.batch_size}')
        if (params.lr_scheduler != 'cosine') and learning_rate2 < 1e-10 and epoch > 10:
            print(f'Learning rate is very low now, ending the process...s')
            exit()


if __name__ == '__main__':
    import argparse, importlib

    parser1 = argparse.ArgumentParser(description='Script to do linear evaluation ')

    parser1.add_argument("--run_id", dest='run_id', type=str, required=False, default= "dummy_linear",
                        help='run_id')
    parser1.add_argument("--restart", action='store_true')
    parser1.add_argument("--saved_model", dest='saved_model', type=str, required=False, default= None,
                        help='run_id')
    parser1.add_argument("--linear", action='store_true')
    parser1.add_argument("--config", dest='conf_file_location', type=str, required=True, default= "parameters_BL",
                        help='conf_file_location')
    parser1.add_argument("--devices", dest='devices', action='append', type =int, required=False, default=None,
                        help='devices should be a list even when it is single')


    args = parser1.parse_args()
    print(f'Restart {args.restart}')
    
    params_filename = args.conf_file_location.replace('.py', '')
    if os.path.exists(params_filename + '.py'):
        params = importlib.import_module(params_filename)
        print(f' {params_filename} is loaded as params')
    else:
        print(f'{params_filename} dne, give it correct path!')
        
    from dataloaders_july22.dl_linear_frameids import *
    
    
    run_id = args.run_id
    saved_model = args.saved_model
    linear = args.linear
    devices = args.devices
    if devices is None: 
        devices = list(range(torch.cuda.device_count()))
    
    print(f'devices are {devices}') 

    if saved_model is not None and len(saved_model):
        saved_model = '/' +saved_model
        
    else:
        saved_model = params.pretrained_checkpoint

    if saved_model is not None and len(saved_model):
        saved_model = saved_model.replace('-symlink', '')

    train_classifier(str(run_id), args.restart, saved_model, linear, params, devices)


        


