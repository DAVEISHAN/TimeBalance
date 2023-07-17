from turtle import fd
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

from model import *
from torch.cuda.amp import autocast, GradScaler

# if torch.cuda.is_available(): 
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly = True


def train_epoch(run_id, learning_rate2,  epoch, data_loader_labeled, data_loader_unlabeled, models ,criterions , optimizer, writer, use_cuda, scaler, device_name):
    print('train at epoch {}'.format(epoch))
    for param_group in optimizer.param_groups:
        param_group['lr']=learning_rate2
        writer.add_scalar('Learning Rate', learning_rate2, epoch)  

        print("Learning rate is: {}".format(param_group['lr']))
  
    criterion, criterion_teacher = criterions
    model, model_dist, model_inv = models


    losses, losses_teacher_unlabeled, losses_teacher_labeled, losses_supervised = [], [], [], []
    loss_mini_batch = 0
    predictions, gt = [], []


    model.train()
    model_dist.eval()
    model_inv.eval()

    for i, (data1, data2) in enumerate(zip(data_loader_labeled, data_loader_unlabeled)):
    
        optimizer.zero_grad()

        inputs_labeled = data1[0] 
        label = data1[1]
    
        inputs_unlabeled = data2[0] 

        fd_score_labeled = data1[-1]
        fd_score_unlabeled = data2[-1]


        inputs_labeled = inputs_labeled.permute(0,2,1,3,4)
        inputs_unlabeled = inputs_unlabeled.permute(0,2,1,3,4)



        if params.RGB or params.normalize:
            inputs_labeled = torch.flip(inputs_labeled, [1]) 
            inputs_unlabeled = torch.flip(inputs_unlabeled, [1]) 
        
        
        if params.normalize: 
            # print('convert this input to normalized', inputs.shape) #[8, 3, 16, 224, 224])           

            inputs = inputs.permute(0,2,1,3,4)
            
            inputs_shape = inputs.shape
            inputs = inputs.reshape(inputs_shape[0]*inputs_shape[1], inputs_shape[2], inputs_shape[3], inputs_shape[4])
            inputs = torchvision.transforms.functional.normalize(inputs, mean =(0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
            inputs = inputs.reshape(inputs_shape)
            
            inputs = inputs.permute(0,2,1,3,4)
   
       
        if use_cuda:
            inputs_labeled = inputs_labeled.to(device=torch.device(device_name))
            inputs_unlabeled = inputs_unlabeled.to(device=torch.device(device_name))

            label = torch.from_numpy(np.asarray(label)).to(device=torch.device(device_name))

            fd_score_labeled = torch.Tensor(fd_score_labeled).unsqueeze(1).to(device=torch.device(device_name))

            fd_score_unlabeled = torch.Tensor(fd_score_unlabeled).unsqueeze(1).to(device=torch.device(device_name))

        
        with autocast(): 
            output_labeled = model(inputs_labeled)
            output_unlabeled = model(inputs_unlabeled)


            ## since dist and inv models require bgr input
            inputs_labeled = torch.flip(inputs_labeled, [1])
            inputs_unlabeled = torch.flip(inputs_unlabeled, [1]) 


            output_dist_labeled = nn.functional.softmax(model_dist(inputs_labeled), dim=1)
            output_dist_unlabeled = nn.functional.softmax(model_dist(inputs_unlabeled), dim=1)

            output_inv_labeled = nn.functional.softmax(model_inv(inputs_labeled), dim=1)
            output_inv_unlabeled = nn.functional.softmax(model_inv(inputs_unlabeled), dim=1)



            labeled_teacher_prediction = (1-fd_score_labeled)*output_dist_labeled + (fd_score_labeled)*output_inv_labeled

            unlabeled_teacher_prediction = (1-fd_score_unlabeled)*output_dist_unlabeled + (fd_score_unlabeled)*output_inv_unlabeled
            

            # output = model(inputs)
            # print('output1shape', output1.shape)
            # print('outputshape', output.shape)



            superivsed_loss = criterion(output_labeled,label)


            if params.teacher_supervision == 'kld':
                labeled_teacher_loss = criterion_teacher(nn.functional.log_softmax(output_labeled, dim=1), labeled_teacher_prediction)

                unlabeled_teacher_loss = criterion_teacher(nn.functional.log_softmax(output_unlabeled, dim=1), unlabeled_teacher_prediction)

            elif params.teacher_supervision == 'l2':
                labeled_teacher_loss = criterion_teacher(nn.functional.softmax(output_labeled, dim=1), labeled_teacher_prediction)

                unlabeled_teacher_loss = criterion_teacher(nn.functional.softmax(output_unlabeled, dim=1), unlabeled_teacher_prediction)


            loss = params.supervised_weight*superivsed_loss + params.labeled_teacher_loss_weight*labeled_teacher_loss + params.unlabeled_teacher_loss_weight*unlabeled_teacher_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            predictions.extend(torch.max(output_labeled, axis=1).indices.cpu().numpy())
            gt.extend(label.cpu().numpy())


        losses.append(loss.item())
        losses_teacher_unlabeled.append(unlabeled_teacher_loss.item())
        losses_teacher_labeled.append(labeled_teacher_loss.item())
        losses_supervised.append(superivsed_loss.item())


        if i % 24 == 0: 
            print(f'Training Epoch {epoch}, Batch {i}, Loss: {np.mean(losses) :.5f}, Supervised Labeled: {np.mean(losses_supervised) :.5f}, Teacher Labeled: {np.mean(losses_teacher_labeled) :.5f}, Teacher UnLoss: {np.mean(losses_teacher_unlabeled) :.5f}', flush=True)

        
    print(f'Training Epoch {epoch}, Loss: {np.mean(losses) :.5f}, Supervised Labeled: {np.mean(losses_supervised) :.5f}, Teacher Labeled: {np.mean(losses_teacher_labeled) :.5f}, Teacher UnLoss: {np.mean(losses_teacher_unlabeled) :.5f}', flush=True)
    
    writer.add_scalar('Training Loss', np.mean(losses), epoch)
    writer.add_scalar('Teacher UnLoss', np.mean(losses_teacher_unlabeled), epoch)
    writer.add_scalar('Teacher LLoss', np.mean(losses_teacher_labeled), epoch)
    writer.add_scalar('Supervised Loss', np.mean(losses_supervised), epoch)

    predictions = np.asarray(predictions)
    gt = np.asarray(gt)
        
        
    accuracy = ((predictions == gt).sum())/np.size(predictions)
    print(f'Training Accuracy at Epoch {epoch} is {accuracy*100 :0.3f}')
    writer.add_scalar('Training Accuracy', accuracy, epoch)
    
    

    del loss, label

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

    if params.ssl_method == 'tclr_ssl':
        model = load_r3d_classifier(saved_model_file = params.checkpoint_tclr_checkpoint, num_classes = params.num_classes, arch = params.backbone)
    
    model_dist = load_r3d_classifier(saved_model_file = params.checkpoint_dist, num_classes = params.num_classes,arch = params.backbone)

    model_inv = load_r3d_classifier(saved_model_file = params.checkpoint_inv, num_classes = params.num_classes,arch = params.backbone)


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
            
            if not ('final_class_fc' in name or 'twoD_fc' in name or 'twoThree_fc' in name or 'fc' in name):
                param.requires_grad = False
            else:
                print(f'Kept unfrozen {name}')
    
    
    
    epoch0 = 0
    learning_rate1 = params.learning_rate
    best_score = 10000

    
    criterion= nn.CrossEntropyLoss()
    if params.teacher_supervision == 'kld':
        criterion_teacher = nn.KLDivLoss(size_average=None, reduce=None, reduction='mean', log_target=False)
    elif params.teacher_supervision == 'l2':
        criterion_teacher = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    

    device_name =  'cuda:' + str(devices[0]) 
    print(f'Device name is {device_name}')
    if len(devices)>1:
        print(f'Multiple GPUS found!')
        # model=nn.DataParallel(model)
        model = torch.nn.DataParallel(model, device_ids=devices)
        model_dist = torch.nn.DataParallel(model_dist, device_ids=devices)
        model_inv = torch.nn.DataParallel(model_inv, device_ids=devices)

        model.cuda()
        model_dist.cuda()
        model_inv.cuda()


    else:
        print('Only 1 GPU is available')
        
        model.to(device=torch.device(device_name))
        model_dist.to(device=torch.device(device_name))
        model_inv.to(device=torch.device(device_name))

        criterion.to(device=torch.device(device_name))
    
    if params.opt_type == 'adam':
        optimizer = optim.Adam(model.parameters(),lr=params.learning_rate)
    elif params.opt_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9)
    elif params.opt_type == 'adamW':
        optimizer = optim.AdamW(model.parameters(),lr=params.learning_rate, weight_decay=1e-8)
    else:
        raise NotImplementedError(f"not supporting {params.opt_type}")
    
    train_dataset_labeled = baseline_dataloader_train_strong(params = params, shuffle = False, data_percentage = params.data_percentage, dl_mode= '20p', fd_score= True)

    train_dataloader_labeled = DataLoader(train_dataset_labeled, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn_train)
    print(f'Labeled Train dataset length: {len(train_dataset_labeled)}')
    print(f'Labeled Train dataset steps per epoch: {len(train_dataset_labeled)/params.batch_size}')

    train_dataset_unlabeled = baseline_dataloader_train_strong(params = params, shuffle = True, data_percentage = params.data_percentage, dl_mode= '20pUnlabeled', fd_score= True)

    train_dataloader_unlabeled = DataLoader(train_dataset_unlabeled, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn_train)
    print(f'Unlabeled Train dataset length: {len(train_dataset_unlabeled)}')
    print(f'Unlabeled Train dataset steps per epoch: {len(train_dataset_unlabeled)/params.batch_size}')


    
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
    crop_facs = params.cropping_facs# [0.7, 0.8, 1.0] #because of low cpu memory, val features itself takes 75G!

   
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
                print(f'Dropping learning rate to {learning_rate2/10} for epoch')
                print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\')
                learning_rate2 = learning_rate1/(params.lr_reduce_factor**scheduler_step)
                scheduler_epoch = 0
                scheduler_step += 1
        

        model, train_loss, scaler = train_epoch(run_id, learning_rate2,  epoch, train_dataloader_labeled, train_dataloader_unlabeled, [model, model_dist, model_inv], [criterion, criterion_teacher], optimizer, writer, use_cuda, scaler,device_name)
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

                        predictions1 = np.zeros((len(list(pred_dict.keys())),params.num_classes))
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

        train_dataloader_labeled = DataLoader(train_dataset_labeled, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn_train)
        print(f'Labeled Train dataset length: {len(train_dataset_labeled)}')
        print(f'Labeled Train dataset steps per epoch: {len(train_dataset_labeled)/params.batch_size}')

        train_dataset_unlabeled = baseline_dataloader_train_strong(params = params, shuffle = True, data_percentage = params.data_percentage, dl_mode= '20pUnlabeled', fd_score= True)

        train_dataloader_unlabeled = DataLoader(train_dataset_unlabeled, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, collate_fn=collate_fn_train)
        print(f'Unlabeled Train dataset length: {len(train_dataset_unlabeled)}')
        print(f'Unlabeled Train dataset steps per epoch: {len(train_dataset_unlabeled)/params.batch_size}')

        if (params.lr_scheduler != 'cosine') and learning_rate2 < 1e-10 and epoch > 10:
            print(f'Learning rate is very low now, ending the process...s')
            exit()



        

if __name__ == '__main__':
    import argparse, importlib

    parser1 = argparse.ArgumentParser(description='Script to do semi supervised training of student')

    parser1.add_argument("--run_id", dest='run_id', type=str, required=False, default= "dummy_semi",
                        help='run_id')
    parser1.add_argument("--restart", action='store_true')
    parser1.add_argument("--saved_model", dest='saved_model', type=str, required=False, default= None,
                        help='run_id')
    parser1.add_argument("--linear", action='store_true')
    parser1.add_argument("--config", dest='config_file_location', type=str, required=True, default= "conf_semisup",
                        help='config_file_location')
    parser1.add_argument("--devices", dest='devices', action='append', type =int, required=False, default=None,
                        help='devices should be a list even when it is single')

    args = parser1.parse_args()
    print(f'Restart {args.restart}')
    
    params_filename = args.config_file_location.replace('.py', '')
    if os.path.exists(params_filename + '.py'):
        # import args.params_file_location as params
        params = importlib.import_module(params_filename)
        print(f' {params_filename} is loaded as params')
    else:
        print(f'{params_filename} dne, give it correct path!')
        
    from dataloaders.dl_semisup_2clips import *
    
    
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

    if saved_model is not None and len(saved_model):
        saved_model = saved_model.replace('-symlink', '')


    train_classifier(str(run_id), args.restart, saved_model, linear, params, devices)


        


