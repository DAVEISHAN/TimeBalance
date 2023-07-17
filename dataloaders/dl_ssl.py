import enum
import os, sys, traceback
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# sys.path.insert(0, "/sensei-fs/users/idave/ucf101_exp_with_VTN/")
sys.path.insert(0, '/home/ishan/self_supervised/r3d/ssl_for_semi/')

import config as cfg
import random
import pickle
# import parameters_BL as params
import json
import math
import cv2
# from tqdm import tqdm
import time
import torchvision.transforms as trans


class dl_ssl_gen(Dataset):

    def __init__(self, params, dataset='ucf101', shuffle = True, data_percentage = 1.0, split = 1):

        self.dataset= dataset
        self.params = params
        if self.dataset == 'ucf101':
            path_file = 'ucfTrainTestlist/trainlist0' + str(split) +'.txt'
            self.all_paths = open(os.path.join(cfg.path_folder, path_file),'r').read().splitlines()

            self.classes= json.load(open(cfg.ucf_class_mapping))['classes']


        elif self.dataset == 'hmdb51':
            file_name = 'hmdb_train_' + str(split) + '.txt'
            self.all_paths = open(os.path.join(cfg.path_folder,file_name),'r').read().splitlines()
            self.classes= json.load(open(cfg.hmdb_mapping))

        elif self.dataset == 'k400':
            self.all_paths = open('/sensei-fs/users/idave/data/k400train_full_path_resized_annos.txt','r').read().splitlines()    

        else:
            print(f'{self.dataset} dne')
               
        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(self.all_paths)
        
        self.data_percentage = data_percentage
        self.data_limit = int(len(self.all_paths)*self.data_percentage)
        self.data = self.all_paths[0: self.data_limit]
        self.PIL = trans.ToPILImage()
        self.TENSOR = trans.ToTensor()

    def __len__(self):
        return len(self.data)  
    
    def __getitem__(self,index):
        sparse_clip, dense_clip0, dense_clip1, dense_clip2, dense_clip3, \
                a_sparse_clip, a_dense_clip0, a_dense_clip1, a_dense_clip2, a_dense_clip3, list_sparse, list_dense, label, vid_path = self.process_data(index)
        return sparse_clip, dense_clip0, dense_clip1, dense_clip2, dense_clip3, \
                a_sparse_clip, a_dense_clip0, a_dense_clip1, a_dense_clip2, a_dense_clip3, list_sparse, list_dense, label, vid_path

    def process_data(self, idx):

        if self.dataset == 'ucf101':
            vid_path = cfg.path_folder + '/UCF-101/' + self.data[idx].split(' ')[0]
            
            # self.data[idx].split(' ')[0][41:]

            label = self.classes[vid_path.split('/')[-2]]
        elif self.dataset == 'hmdb51':
            vid_path = self.data[idx]
            label = self.classes[vid_path.split(' ')[1]]
            vid_path = cfg.path_folder.replace('/paths_crcv_node', '')  + '/HMDB51/' + vid_path.split(' ')[1]+ '/' + vid_path.split(' ')[0]
        elif self.dataset == 'k400':
            vid_path = cfg.path_folder + '/Kinetics/' + vid_path.split(' ')[0]
            label = int(vid_path.split(' ')[1])

        sparse_clip, dense_clip0, dense_clip1, dense_clip2, dense_clip3, a_sparse_clip, a_dense_clip0, a_dense_clip1, a_dense_clip2, a_dense_clip3, list_sparse, list_dense = self.build_clip(vid_path)

        return sparse_clip, dense_clip0, dense_clip1, dense_clip2, dense_clip3, a_sparse_clip, a_dense_clip0, a_dense_clip1, a_dense_clip2, a_dense_clip3, list_sparse, list_dense, label, vid_path 
    
    def build_clip(self, vid_path):
        # print(os.path.exists(vid_path))
        cap = cv2.VideoCapture(vid_path)
        cap.set(1, 0)
        frame_count = cap.get(7)
        self.ori_reso_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.ori_reso_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


        if frame_count <= 56:
            # print(f'Video {vid_path} has insufficient frames {frame_count}')
            return None, None, None, None, None, None, None, None, None, None, None, None

        ############################# frame_list maker start here#################################
        min_temporal_span_sparse = self.params.num_frames*self.params.sr_ratio
        if frame_count > min_temporal_span_sparse:
            start_frame = np.random.randint(0,frame_count-min_temporal_span_sparse)
            sr_sparse = 4
        else:
            start_frame = 0
            sr_sparse = 4
        sr_dense = int(sr_sparse/4)
        
        frames_sparse = [start_frame] + [start_frame + i*sr_sparse for i in range(1,self.params.num_frames)]

        frames_dense = [[frames_sparse[j*4]]+[frames_sparse[j*4] + i*sr_dense for i in range(1,self.params.num_frames)] for j in range(4)]            

        sparse_clip = []
        dense_clip0 = []
        dense_clip1 = []
        dense_clip2 = []
        dense_clip3 = []

        a_sparse_clip = []
        a_dense_clip0 = []
        a_dense_clip1 = []
        a_dense_clip2 = []
        a_dense_clip3 = []

        list_sparse = []
        list_dense = [[] for i in range(4)]
        count = -1

        random_array = np.random.rand(10,9)
        x_erase = np.random.randint(0,self.params.reso_h, size = (10,))
        y_erase = np.random.randint(0,self.params.reso_w, size = (10,))


        cropping_factor1 = np.random.uniform(self.params.augmentations_dict['cropping_fac'] , 1, size = (10,)) # on an average cropping factor is 80% i.e. covers 64% area

        x0 = [np.random.randint(0, self.ori_reso_w - self.ori_reso_w*cropping_factor1[ii] + 1) for ii in range(10)]          
        y0 = [np.random.randint(0, self.ori_reso_h - self.ori_reso_h*cropping_factor1[ii] + 1) for ii in range(10)]

        contrast_factor1 = np.random.uniform(0.75,1.25, size = (10,))
        hue_factor1 = np.random.uniform(-0.1,0.1, size = (10,))
        saturation_factor1 = np.random.uniform(0.75,1.25, size = (10,))
        brightness_factor1 = np.random.uniform(0.75,1.25,size = (10,))
        gamma1 = np.random.uniform(0.75,1.25, size = (10,))


        erase_size1 = np.random.randint(int((self.ori_reso_h/6)*(self.params.reso_h/224)),int((self.ori_reso_h/3)*(self.params.reso_h/224)), size = (10,))

        erase_size2 = np.random.randint(int((self.ori_reso_w/6)*(self.params.reso_h/224)),int((self.ori_reso_w/3)*(self.params.reso_h/224)), size = (10,))
        random_color_dropped = np.random.randint(0,3,(10))

        while(cap.isOpened()): 
            count += 1
            ret, frame = cap.read()
            if ((count not in frames_sparse) and (count not in frames_dense[0]) \
                and (count not in frames_dense[1]) and (count not in frames_dense[2]) \
                and (count not in frames_dense[3])) and (ret == True): 
                continue
            if ret == True:
                if (count in frames_sparse):
                    sparse_clip.append(self.augmentation(frame, random_array[0], x_erase[0], y_erase[0], cropping_factor1[0],\
                            x0[0], y0[0], contrast_factor1[0], hue_factor1[0], saturation_factor1[0], brightness_factor1[0],\
                            gamma1[0],erase_size1[0],erase_size2[0], random_color_dropped[0]))
                    a_sparse_clip.append(self.augmentation(frame, random_array[1], x_erase[1], y_erase[1], cropping_factor1[1],\
                            x0[1], y0[1], contrast_factor1[1], hue_factor1[1], saturation_factor1[1], brightness_factor1[1],\
                            gamma1[1],erase_size1[1],erase_size2[1], random_color_dropped[1]))
                    list_sparse.append(count)
                    # print(f'Lenght of list is {len(list_sparse)}')
                    # print(f'Length of clip is {len(sparse_clip)}')
                if (count in frames_dense[0]):
                    dense_clip0.append(self.augmentation(frame, random_array[2], x_erase[2], y_erase[2], cropping_factor1[2],\
                            x0[2], y0[2], contrast_factor1[2], hue_factor1[2], saturation_factor1[2], brightness_factor1[2],\
                            gamma1[2],erase_size1[2],erase_size2[2], random_color_dropped[2]))
                    a_dense_clip0.append(self.augmentation(frame, random_array[3], x_erase[3], y_erase[3], cropping_factor1[3],\
                            x0[3], y0[3], contrast_factor1[3], hue_factor1[3], saturation_factor1[3], brightness_factor1[3],\
                            gamma1[3],erase_size1[3],erase_size2[3], random_color_dropped[3]))
                    list_dense[0].append(count)
                if (count in frames_dense[1]):
                    dense_clip1.append(self.augmentation(frame, random_array[4], x_erase[4], y_erase[4], cropping_factor1[4],\
                            x0[4], y0[4], contrast_factor1[4], hue_factor1[4], saturation_factor1[4], brightness_factor1[4],\
                            gamma1[4],erase_size1[4],erase_size2[4], random_color_dropped[4]))
                    a_dense_clip1.append(self.augmentation(frame, random_array[5], x_erase[5], y_erase[5], cropping_factor1[5],\
                            x0[5], y0[5], contrast_factor1[5], hue_factor1[5], saturation_factor1[5], brightness_factor1[5],\
                            gamma1[5],erase_size1[5],erase_size2[5], random_color_dropped[5]))
                    list_dense[1].append(count)
                if (count in frames_dense[2]):
                    dense_clip2.append(self.augmentation(frame, random_array[6], x_erase[6], y_erase[6], cropping_factor1[6],\
                            x0[6], y0[6], contrast_factor1[6], hue_factor1[6], saturation_factor1[6], brightness_factor1[6],\
                            gamma1[6],erase_size1[6],erase_size2[6], random_color_dropped[6]))
                    a_dense_clip2.append(self.augmentation(frame, random_array[7], x_erase[7], y_erase[7], cropping_factor1[7],\
                            x0[7], y0[7], contrast_factor1[7], hue_factor1[7], saturation_factor1[7], brightness_factor1[7],\
                            gamma1[7],erase_size1[7],erase_size2[7], random_color_dropped[7]))
                    list_dense[2].append(count)
                if (count in frames_dense[3]):
                    dense_clip3.append(self.augmentation(frame, random_array[8], x_erase[8], y_erase[8], cropping_factor1[8],\
                            x0[8], y0[8], contrast_factor1[8], hue_factor1[8], saturation_factor1[8], brightness_factor1[8],\
                            gamma1[8],erase_size1[8],erase_size2[8], random_color_dropped[8]))
                    a_dense_clip3.append(self.augmentation(frame, random_array[9], x_erase[9], y_erase[9], cropping_factor1[9],\
                            x0[9], y0[9], contrast_factor1[9], hue_factor1[9], saturation_factor1[9], brightness_factor1[9],\
                            gamma1[9],erase_size1[9],erase_size2[9], random_color_dropped[9]))
                    list_dense[3].append(count)

            else:
                break

        if len(sparse_clip) < self.params.num_frames and len(sparse_clip)>13:
            # if params.num_frames - len(sparse_clip) >= 1:
            #     print(f'sparse_clip {vid_path} is missing {params.num_frames - len(sparse_clip)} frames')
            remaining_num_frames = self.params.num_frames - len(sparse_clip)
            sparse_clip = sparse_clip + sparse_clip[::-1][1:remaining_num_frames+1]
            a_sparse_clip = a_sparse_clip + a_sparse_clip[::-1][1:remaining_num_frames+1]

        if len(dense_clip3) < self.params.num_frames and len(dense_clip3)>7:
            
            # if params.num_frames - len(dense_clip3) >= 1:
            #     print(f'dense_clip3 {vid_path} is missing {params.num_frames - len(dense_clip3)} frames')
            remaining_num_frames = self.params.num_frames - len(dense_clip3)
            dense_clip3 = dense_clip3 + dense_clip3[::-1][1:remaining_num_frames+1]    
            a_dense_clip3 = a_dense_clip3 + a_dense_clip3[::-1][1:remaining_num_frames+1]  
        
        try:
            assert(len(sparse_clip)== self.params.num_frames)
            assert(len(dense_clip0)== self.params.num_frames)
            assert(len(dense_clip1)== self.params.num_frames)
            assert(len(dense_clip2)== self.params.num_frames)
            assert(len(dense_clip3)== self.params.num_frames)

            return sparse_clip, dense_clip0, dense_clip1, dense_clip2, dense_clip3, \
                    a_sparse_clip, a_dense_clip0, a_dense_clip1, a_dense_clip2, a_dense_clip3, list_sparse, list_dense
        except:
            print(f'Clip {vid_path} has some frames reading issue, failed')
            return None, None, None, None, None, None, None, None, None, None, None, None

    def augmentation(self, image, random_array, x_erase, y_erase, cropping_factor1,\
        x0, y0, contrast_factor1, hue_factor1, saturation_factor1, brightness_factor1,\
        gamma1,erase_size1,erase_size2, random_color_dropped):
        
        image = self.PIL(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = trans.functional.resized_crop(image,y0,x0,int(self.ori_reso_h*cropping_factor1),int(self.ori_reso_w*cropping_factor1),(self.params.reso_h, self.params.reso_w))


        if random_array[0] < 0.125:
            image = trans.functional.adjust_contrast(image, contrast_factor = contrast_factor1) #0.75 to 1.25
        if random_array[1] < 0.3 :
            image = trans.functional.adjust_hue(image, hue_factor = hue_factor1) # hue factor will be between [-0.25, 0.25]*0.4 = [-0.1, 0.1]
        if random_array[2] < 0.3 :
            image = trans.functional.adjust_saturation(image, saturation_factor = saturation_factor1) # brightness factor will be between [0.75, 1,25]
        if random_array[3] < 0.3 :
            image = trans.functional.adjust_brightness(image, brightness_factor = brightness_factor1) # brightness factor will be between [0.75, 1,25]
        if random_array[0] > 0.125 and random_array[0] < 0.25:
            image = trans.functional.adjust_contrast(image, contrast_factor = contrast_factor1) #0.75 to 1.25
        if random_array[4] > 0.70:
            if random_array[4] < 0.875:
                image = trans.functional.to_grayscale(image, num_output_channels = 3)
                if random_array[5] > 0.25:
                    image = trans.functional.adjust_gamma(image, gamma = gamma1, gain=1) #gamma range [0.8, 1.2]
            else:
                image = trans.functional.to_tensor(image)
                image[random_color_dropped,:,:] = 0
                image = self.PIL(image)

        if random_array[6] > 0.5:
            image = trans.functional.hflip(image)

        image = trans.functional.to_tensor(image)
        
        if self.params.augmentations_dict["blur"] and random_array[8] > 0.5:

            kernel_size= int(15*(self.params.reso_w/112))
            if kernel_size%2 ==0:
                kernel_size+=1

            image = trans.functional.gaussian_blur(image, kernel_size=kernel_size, sigma = 0.2)


        if random_array[7] < 0.7 :
            image = trans.functional.erase(image, x_erase, y_erase, erase_size1, erase_size2, v=0) 

        return image        

def collate_fn2(batch):

    sparse_clip, dense_clip0, dense_clip1, dense_clip2, dense_clip3, a_sparse_clip, \
    a_dense_clip0, a_dense_clip1, a_dense_clip2, a_dense_clip3, \
    list_sparse, list_dense, label, vid_path = [], [], [], [], [], [], [], [], [], [], [], [], [], []
    # print(len(batch))
    for item in batch:
        if not (None in item):
            # f_clip.append(torch.from_numpy(np.asarray(item[0],dtype='f')))
            # t_clip.append(torch.from_numpy(np.asarray(item[1],dtype='f')))
            # print(len(item[0]))
            sparse_clip.append(torch.stack(item[0],dim=0)) # I might need to convert this tensor to torch.float
            dense_clip0.append(torch.stack(item[1],dim=0))
            dense_clip1.append(torch.stack(item[2],dim=0))
            dense_clip2.append(torch.stack(item[3],dim=0))
            dense_clip3.append(torch.stack(item[4],dim=0))

            a_sparse_clip.append(torch.stack(item[5],dim=0)) # I might need to convert this tensor to torch.float
            a_dense_clip0.append(torch.stack(item[6],dim=0))
            a_dense_clip1.append(torch.stack(item[7],dim=0))
            a_dense_clip2.append(torch.stack(item[8],dim=0))
            a_dense_clip3.append(torch.stack(item[9],dim=0))


            list_sparse.append(np.asarray(item[10]))
            list_dense.append(np.asarray(item[11]))
            label.append(item[12])
            vid_path.append(item[13])
        # else:
            # print('oh no2')
    # print(len(f_clip))
    sparse_clip = torch.stack(sparse_clip, dim=0)
    dense_clip0 = torch.stack(dense_clip0, dim=0)
    dense_clip1 = torch.stack(dense_clip1, dim=0)
    dense_clip2 = torch.stack(dense_clip2, dim=0)
    dense_clip3 = torch.stack(dense_clip3, dim=0)

    a_sparse_clip = torch.stack(a_sparse_clip, dim=0)
    a_dense_clip0 = torch.stack(a_dense_clip0, dim=0)
    a_dense_clip1 = torch.stack(a_dense_clip1, dim=0)
    a_dense_clip2 = torch.stack(a_dense_clip2, dim=0)
    a_dense_clip3 = torch.stack(a_dense_clip3, dim=0)

    return sparse_clip, dense_clip0, dense_clip1, dense_clip2, dense_clip3, a_sparse_clip, \
            a_dense_clip0, a_dense_clip1, a_dense_clip2, a_dense_clip3, \
            list_sparse, list_dense, label, vid_path

if __name__ == '__main__':
    import params_ssl as params
    import torchvision
    from PIL import Image, ImageDraw, ImageFont
    dataset = 'hmdb51'#'ucf101'
    visualize = True
    run_id = 'hmdb_try1'#'try4_224_ssl'#'casia_try1' + casia_split 
    vis_output_path = '/home/c3-0/ishan/semisup_saved_models/some_visualization/ssl_dl/' + run_id


    train_dataset = dl_ssl_gen(params = params, dataset= dataset, shuffle = True, data_percentage = 1.0)

    train_dataloader = DataLoader(train_dataset, batch_size=16, \
        shuffle=False, num_workers=8, collate_fn=collate_fn2)

    print(f'Length of dataset: {len(train_dataset)}')
    print(f'Step involved: {len(train_dataset)/params.batch_size}')
    t=time.time()

    for i, (sparse_clip, dense_clip0, dense_clip1, dense_clip2, dense_clip3, a_sparse_clip, \
            a_dense_clip0, a_dense_clip1, a_dense_clip2, a_dense_clip3, \
            list_sparse, list_dense, label, vid_path) in enumerate(train_dataloader):
        if (i+1)%1 == 0:
            print(sparse_clip.shape)
            print(dense_clip3.shape)
            # print(f'Sparse frames list {list_sparse}')
            # print(f'Dense frames list {list_dense}')
            print()

        if visualize:
            if dataset == 'ucf101':
                classes = json.load(open(cfg.ucf_class_mapping))['classes']
            elif dataset == 'hmdb51':
                classes= json.load(open(cfg.hmdb_mapping))
            inv_map = {v: k for k, v in classes.items()}

            if not os.path.exists(vis_output_path):
                os.makedirs(vis_output_path)

            counter = 0

            # for kk2, clip in enumerate([sparse_clip, a_sparse_clip, dense_clip0, dense_clip3, a_dense_clip0, a_dense_clip3]):

            for kk in range(sparse_clip.shape[0]):

                clip = torch.stack([sparse_clip[kk], a_sparse_clip[kk], dense_clip0[kk], dense_clip3[kk], a_dense_clip0[kk], a_dense_clip3[kk]], dim=0)
                print(clip.shape)
                clip_shape = clip.shape
                clip = clip.reshape(clip_shape[0]*clip_shape[1], clip_shape[2], clip_shape[3], clip_shape[4])
                print(clip.shape)

                clip *= 255
                clip = clip.to(torch.uint8)#.permute(0,3,1,2) 
                image = torchvision.utils.make_grid(clip, nrow = params.num_frames)

                filename =  vis_output_path +'/' + inv_map[label[kk]] + str(counter) + '.png' 

                torchvision.io.write_png(image, filename)
                counter +=1
            exit()


    print(f'Time taken to load data is {time.time()-t}')