#  __        __  _____  _____ 
#  \ \      / / |__  / |  ___|
#   \ \ /\ / /    / /  | |_   
#    \ V  V /    / /_  |  _|  
#     \_/\_/    /____| |_|    
#    ParticleFlowNet--->training                Note: This script can run on the CPU or any GPU(s), for testing purposes only.
#           2023-12-1                           After testing, Please Use GPUonly script to sbatch               
import torch
from torch.utils import data
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import tqdm
from torchsummary import summary
import numpy as np
import sys
import os
import time
import torch.distributed as dist
import torch.multiprocessing as mp
#--->import ZF_ML
from ZF_ML.ZF_utils import DDP_Config, train_or_predict, performance_plot
from ZF_ML.ZF_DataSet import PFN_DataSet
from ZF_ML.ZF_Net import ParticleFlowNet
#--->Maybe more functions

################################################################################################################
train_or_predict.cheems()#means start...
try:
    SUFFIX=sys.argv[1]
except:
    default_suffix=time.strftime('%Y-%m-%d_%H_%M_%S',time.localtime())
    print(f"Seems like you didn't type a suffix, so the [tag_'localtime({default_suffix})'] has been used as the suffix") 
    SUFFIX='tag_'+default_suffix
finally:
################################################################################################################
#Please change the hyperparameters here...
    GPU_NUM=torch.cuda.device_count()
    #loading data
    ROOT_DIR_PATH=r'/hpcfs/cepc/higgsgpu/wuzuofei/Backup/Higgs/train_sample'
    TREE_NAME=r'tree'
    PROCESSES=['ggH','VBFH']
    FEATURES=['jet_eta','jet_m','jet_phi','jet_pt','jet_DL_btag','jet_DL_ctag','jet_DL_utag','jet_GN_btag','jet_GN_ctag','jet_GN_utag']
    NUM_DATA_EACH_CLASS=400_00
    TRAIN_VAL_TEST=[0.8,0.1,0.1]
    #training procedure
    EPOCHS=5
    BATCHSIZE=512
    LR=0.001
    #net config
    USE_BN=True
    PHI_SIZES=(64, 64, 50)
    F_SIZES=(64, 64, 40)
    #others
    NUM_CLASSES=len(PROCESSES)
#################################################################################################################
#print info
print(f'''
--->outside script:
      
suffix:         {SUFFIX}
# gpu           {GPU_NUM}

--->loading data:

root dir path:  {ROOT_DIR_PATH}
tree name:      {TREE_NAME}
num_classes:    {NUM_CLASSES}
processes:      {PROCESSES}
features:       {FEATURES}
# each class:   {NUM_DATA_EACH_CLASS}
train_val_test: {TRAIN_VAL_TEST}

--->training procedure:

epochs:         {EPOCHS}
batchsize:      {BATCHSIZE}
lr:             {LR}

--->net config

use_bn:        {USE_BN}
Phi_sizes:     {PHI_SIZES}
F_sizes:       {F_SIZES}
        ''')
def main(local_rank, train_set, val_set, test_set, GPU_mode=True):
    if GPU_mode:
        DDP_Config.init_ddp(local_rank)
    #########################################################################################################
                                                #dataloader
    features_shape_for_one_event=train_set[0][0].shape
    feature_dim=features_shape_for_one_event[0]

    train_sampler=data.DistributedSampler(train_set) if GPU_mode else None
    val_sampler=data.DistributedSampler(val_set) if GPU_mode else None
    
    train_set=data.DataLoader(train_set,batch_size=BATCHSIZE,sampler=train_sampler,
                              shuffle=not GPU_mode)
    val_set=data.DataLoader(val_set,batch_size=BATCHSIZE,sampler=val_sampler,
                            shuffle=not GPU_mode)
    test_set=data.DataLoader(test_set,batch_size=BATCHSIZE,shuffle=False)
    #########################################################################################################

    net=ParticleFlowNet.ParticleFlowNetwork(num_classes=NUM_CLASSES, 
                                            input_dims=feature_dim, 
                                            Phi_sizes=PHI_SIZES,
                                            F_sizes=F_SIZES,
                                            use_bn=USE_BN)
    
    #net.apply(my_tools.initialize_pfn)        #kaiming_init by default

    loss=nn.CrossEntropyLoss(reduction='none') #none is essential！
    if GPU_mode:
        loss.to(local_rank)
        net.to(local_rank)

    if local_rank==0:
        writer=SummaryWriter(f'output/log_tensorboard/log_{SUFFIX}')
        print(f'TensorBoard log path is {os.getcwd()}/output/log_tensorboard/log_{SUFFIX}')
        writer.add_graph(net,input_to_model=torch.rand(1,*features_shape_for_one_event).to(local_rank) if GPU_mode else torch.rand(1,*features_shape_for_one_event))
        print(summary(net,input_size=features_shape_for_one_event))

    net=nn.parallel.DistributedDataParallel(net,device_ids=[local_rank]) if GPU_mode else net

    optimizer=torch.optim.NAdam(net.parameters(),lr=LR)#After the  "net.to(device=device)"


    if local_rank==0 and GPU_mode:
        torch.cuda.synchronize()
        start=time.time()
    else:
        start=time.time()

    for epoch in range(EPOCHS):
        if GPU_mode:
            train_set.sampler.set_epoch(epoch) #Essential!!! Otherwize each GPU only get same ntuples every epoch
            #train_sampler.set_epoch(epoch)    #Same as the above, both are correct
        loss_train,acc_train=train_or_predict.train_procedure_in_each_epoch_GPUorCPU(net,train_set,loss,optimizer,GPU_mode,local_rank)
        loss_val,acc_val=train_or_predict.evaluate_accuracy_GPUorCPU(net,loss,val_set,GPU_mode,test=False)

        if local_rank==0:
            writer.add_scalars('Metric',{'loss_train':loss_train,
                                         'acc_train':acc_train,
                                         'loss_val':loss_val,
                                         'acc_val':acc_val},epoch)
            print(f'''epoch: {epoch} | acc_train: {acc_train:.3f} | acc_val: {acc_val:.3f} | loss_train: {loss_train:.3f} | loss_val: {loss_val:.3f} | 
            ''')

    if local_rank==0 and GPU_mode:
        torch.cuda.synchronize()
        end=time.time()
        print('Total time in training is ',end-start)
    else:
        end=time.time()
        print('Total time in training is ',end-start)

    if local_rank==0:
        train_or_predict.save_net(net,SUFFIX)
        train_or_predict.save_net_hyperparameters(suffix=SUFFIX, 
                                                  num_classes=NUM_CLASSES, 
                                                  input_dims=feature_dim, 
                                                  Phi_sizes=PHI_SIZES,
                                                  F_sizes=F_SIZES,
                                                  use_bn=USE_BN)
        scores,true_label,loss_test,acc_test=train_or_predict.evaluate_accuracy_GPUorCPU(net,loss,test_set,GPU_mode,test=True)
        print('{:-^80}'.format(f'loss in test is {loss_test:.3f}, accuracy in test is {acc_test:.3f},'))
        pred_label=torch.argmax(scores,dim=1)
        true_label=true_label.cpu().numpy()     #1d
        pred_label=pred_label.cpu().numpy()     #1d
        scores=scores.cpu().numpy()             #2d
        performance_plot.plot_confusion_matrix(suffix=SUFFIX, num_classes=NUM_CLASSES, true_label=true_label, pred_label=pred_label, processes=PROCESSES)

        performance_plot.plot_roc(suffix=SUFFIX, num_classes=NUM_CLASSES, true_label=true_label, scores=scores, processes=PROCESSES)

        performance_plot.plot_scores_hist(suffix=SUFFIX, processes=PROCESSES, true_label=true_label, scores=scores)#bin=20 by default
    if GPU_mode:
        dist.destroy_process_group()
################################################################################################################################### 

if __name__=='__main__':
    fimename=[i +'.root' for i in PROCESSES]
    dataset=PFN_DataSet.PFN_Dataset( fimename
                                    ,num_data=NUM_DATA_EACH_CLASS
                                    ,features=FEATURES
                                    ,tree_name=TREE_NAME
                                    ,root_dir_path=ROOT_DIR_PATH
                                    ,map_PID=False
                                    ,PID_idx=None
                                        )
    train_set,val_set,test_set=data.random_split(dataset=dataset,lengths=TRAIN_VAL_TEST)
    if GPU_NUM!=0:
        mp.spawn(main,args=(train_set, val_set, test_set,True), nprocs=GPU_NUM)
    else:
        main(0,train_set, val_set, test_set,GPU_mode=False)
    print('Program Is Over !!!')









