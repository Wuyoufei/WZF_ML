#  __        __  _____  _____ 
#  \ \      / / |__  / |  ___|
#   \ \ /\ / /    / /  | |_   
#    \ V  V /    / /_  |  _|  
#     \_/\_/    /____| |_|    
#    XXNet--->prediction             Note: (This script can only run on one GPU!)       
           
import torch
from torch.utils import data
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import numpy as np
import sys
import os
import time
import torch.distributed as dist
import torch.multiprocessing as mp
import warnings
#--->import ZF_ML
from ZF_ML.ZF_utils import DDP_Config, train_or_predict, performance_plot
from ZF_ML.ZF_DataSet import PFN_DataSet,PN_DataSet
#--->Maybe more functions

################################################################################################################
train_or_predict.cheems()#means start...
try:
    NET,SUFFIX=sys.argv[1].split('/')   #net/suffix
    if NET.lower() not in ['pfn', 'pn', 'pt']:
        warnings.warn('argv[1] supposed to be "net/suffix", if you changed of position of them, Please ignore this warning...')
        NET,SUFFIX=SUFFIX,NET
except:
    raise Exception('Please input argument: "net/suffix" ')
finally:
################################################################################################################
#Please change the parameters here...
    GPU_NUM=torch.cuda.device_count()

    ROOT_DIR_PATH=r'/hpcfs/cepc/higgsgpu/wuzuofei/Backup/Higgs/train_sample'
    TREE_NAME=r'tree'
    NET_DIR_PATH=None #--->None means use default relative dir
    HYPERPARAMS_DIR_PATH=None #--->None means use default relative dir
    PROCESSES=['ggH', 'VBFH']
    FEATURES=['jet_eta','jet_m','jet_phi','jet_pt','jet_DL_btag','jet_DL_ctag','jet_DL_utag','jet_GN_btag','jet_GN_ctag','jet_GN_utag']
    POINTS_FEATURES=['jet_eta','jet_phi']#--->PN need
    NUM_DATA_EACH_CLASS=40000
    BATCHSIZE=512
    NUM_CLASSES=len(PROCESSES)
#########################################################################################################
#print info
print(f'''
--->outside script:
      
suffix:         {SUFFIX}
# gpu:          {GPU_NUM}

--->loading data:

net:            {NET}
num_classes:    {NUM_CLASSES}
processes:      {PROCESSES}
features:       {FEATURES}
points_features:{POINTS_FEATURES if NET.lower()=='pn' else 'Not using PN...'}
# each class:   {NUM_DATA_EACH_CLASS}
batchsize:      {BATCHSIZE}
        ''')

def main(local_rank, test_set, GPU_mode=True):
    if GPU_mode:
        DDP_Config.init_ddp(local_rank)

    #########################################################################################################    
                                                #dataloader
    test_set=data.DataLoader(test_set,batch_size=BATCHSIZE,shuffle=False)
    #########################################################################################################


    net=train_or_predict.load_net(net=NET, suffix=SUFFIX, hyperparams_dir_path=HYPERPARAMS_DIR_PATH, net_dir_path=NET_DIR_PATH)
    
    loss=nn.CrossEntropyLoss(reduction='none') #none is essential！

    if GPU_mode:
        loss.to(local_rank)
        net.to(local_rank)

    net=nn.parallel.DistributedDataParallel(net,device_ids=[local_rank]) if GPU_mode else net
    
    if local_rank==0 and GPU_mode:
        torch.cuda.synchronize()
        start=time.time()
    else:
        start=time.time()


    if local_rank==0:
        scores,true_label,loss_test,acc_test=train_or_predict.evaluate_accuracy_GPUorCPU(net,loss,test_set,GPU_mode,test=True)
        print('{:-^80}'.format(f'loss in test is {loss_test:.3f}, accuracy in test is {acc_test:.3f},'))
        pred_label=torch.argmax(scores,dim=1)
        true_label=true_label.cpu().numpy()     #1d
        pred_label=pred_label.cpu().numpy()     #1d 
        scores=scores.cpu().numpy()             #2d

################################################################################################################################### 
#--->   performance_plot settings 
        performance_plot.plot_confusion_matrix(suffix=SUFFIX, num_classes=NUM_CLASSES, true_label=true_label, pred_label=pred_label, processes=PROCESSES)

        performance_plot.plot_roc(suffix=SUFFIX, num_classes=NUM_CLASSES, true_label=true_label, scores=scores, processes=PROCESSES)

        performance_plot.plot_scores_hist(suffix=SUFFIX, processes=PROCESSES, true_label=true_label, scores=scores, bins=20)#bin=20 by default

###################################################################################################################################

    if local_rank==0 and GPU_mode:
        torch.cuda.synchronize()
        end=time.time()
        print('Total time in prediction is ',end-start)
    else:
        end=time.time()
        print('Total time in prediction is ',end-start)

    if GPU_mode:
        dist.destroy_process_group()

################################################################################################################################### 

if __name__=='__main__':

    fimename=[i +'.root' for i in PROCESSES]
    if NET.lower() == 'pfn':
        test_set=PFN_DataSet.PFN_Dataset(fimename, num_data=NUM_DATA_EACH_CLASS, features=FEATURES, tree_name=TREE_NAME, root_dir_path=ROOT_DIR_PATH)
    elif NET.lower()=='pn':
        test_set=PN_DataSet.PN_Dataset(fimename, num_data=NUM_DATA_EACH_CLASS, features=FEATURES, points_features=POINTS_FEATURES, tree_name=TREE_NAME, root_dir_path=ROOT_DIR_PATH)

    if GPU_NUM>0:
        mp.spawn(main,args=(test_set,), nprocs=GPU_NUM)
    else:
        main(0, test_set, GPU_mode=False)

    print('Program Is Over !!!')


















