#  __        __  _____  _____ 
#  \ \      / / |__  / |  ___|
#   \ \ /\ / /    / /  | |_   
#    \ V  V /    / /_  |  _|  
#     \_/\_/    /____| |_|          2023-12-1
#                                    
#   Tools for training or predict procedure  
#   For now support: ParticleFlowNet ParticleNet ParticleNeXt   
#   
'''
Note:
Please directly use Linear output when you want to use the net to predict the events.
softmax has already been used in the evaluate_accuracy_xx function

__Future__: PT in load_net...
'''

import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import torch.distributed as dist
import json
import collections
from ZF_ML.ZF_Net import ParticleFlowNet,ParticleNet

###########################################################################################################################################################################

def cheems():
    print(f'''
        ⠀⠀⠀⠀⠀⣀⣀⣀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⡀⣯⡭⠀⢟⢿⣷⣄⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⢠⣼⠏⠴⠶⠈⠘⠻⣘⡆⠀⠀⠀⠀⠀⠀
        ⠀⠀⣠⣾⡟⠁⡀⠀⠀⠀⡼⠡⠛⡄⠀⠀⠀⠀⠀
        ⠀⠀⠙⠻⢴⠞⠁⠀⠊⠀⠀⠀⠈⠉⢄⠀⠀⠀⠀
        ⠀⠀⠀⠀⢀⠀⠀⠀⢃⠄⠂⠀⠀⢀⠞⢣⡀⠀⠀           
        ⠀⠀⠀⠀⡌⠁⠀⠀⠀⢀⠀⠐⠈⠀⠀⡺⠙⡄⠀   
        ⠀⠀⠀⠀⡿⡀⠀⠀⠀⠁⠀⠴⠁⠀⠚⠀⡸⣷⠀
        ⠀⠀⠀⠀⢹⠈⠀⠀⠀⠀⠔⠁⠀⢀⠄⠀⠁⢻⣧
        ⠀⠀⠀⠀⣸⠀⢠⣇⠀⢘⣬⠀⠀⣬⣠⣦⣼⣿⠏
        ⡠⠐⢂⡡⠾⢀⡾⠋⠉⠉⡇⠀⢸⣿⣿⣿⡿⠃⠀
        ⠉⢉⡠⢰⠃⠸⠓⠒⠂⠤⡇⠀⡿⠟⠛⠁⠀⠀⠀
        ⠘⢳⡞⣅⡰⠁⠀⠀⠀⢀⠇⠀⡇⠀⠀⠀⠀⠀⠀
        ⠀⠀⠉⠉⠀⠀⠀⠀⢀⣌⢀⢀⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠘⠊⠀⠀⠀⠀⠀⠀⠀
    Cheems shows up and tells you that the program has started!!!
        ''')

###########################################################################################################################################################################

def try_gpu(i=0):  
    
    if (torch.cuda.device_count()) >= i + 1:     
        print(f'The name of GPU[{i}] is {torch.cuda.get_device_name(i)}')
        return torch.device(f'cuda:{i}')
    else:
        print(f'NO GPU{i}, have to use CPU...')
        return torch.device('cpu')

###########################################################################################################################################################################

def initialize_pfn(m):
    #if isinstance(m,torch.nn.Conv1d) or isinstance(m,torch.nn.Linear):
    torch.nn.init.kaiming_uniform_(m.weight.data)

###########################################################################################################################################################################

def one_hot_encoding(true_label): #1D
    r'''
    true_label supposed to starts from zero !!!
    '''
    one_hot=torch.zeros(num:=len(true_label),true_label.max()+1) #python version>=3.8
    one_hot[range(num),true_label]=1
    return one_hot

###########################################################################################################################################################################

class Accumulator:  
    #accumulate in n variables
    def __init__(self, n):
        self.data = [torch.tensor(0.0)] * n

    def add(self, *args):
        self.data = [a + b for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [torch.tensor(0.0)] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

###########################################################################################################################################################################

def accuracy(y_hat, y):  
    #calculate the number of correct predictions
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return cmp.type(y.dtype).sum()

###########################################################################################################################################################################

def train_procedure_in_each_epoch_GPU(net, tran_set, loss, optimizer, local_rank):
    #sum_of_loss_of_all_events, sum_of_the_number_of_correct_predictions, all_events
    metric=Accumulator(3)
    if local_rank==0:
        pbar=tqdm(tran_set)
    net.train()
    for batch in tran_set:
        batch = [bat.cuda() for bat in batch]
        optimizer.zero_grad()
        y_hat=net(*batch[:-1]) 
        l=loss(y_hat,batch[-1])
        l.mean().backward()
        optimizer.step()
        ################################################
        b_loss=l.sum()
        b_acc=accuracy(y_hat, batch[-1])
        b_num=torch.tensor(batch[-1].numel()).cuda()
        dist.reduce(b_loss,0,op=dist.ReduceOp.SUM)
        dist.reduce(b_acc,0,op=dist.ReduceOp.SUM)
        dist.reduce(b_num,0,op=dist.ReduceOp.SUM)
        metric.add(b_loss, b_acc, b_num)
        if local_rank==0:
            pbar.set_description(f'batch_acc:{float(b_acc/b_num):.3f}, batch_loss:{float(b_loss/b_num):.3f}')
            pbar.update(1)
    #loss_train, acc_train
    return metric[0] / metric[2], metric[1] / metric[2]

###########################################################################################################################################################################

def evaluate_accuracy_GPU(net, loss, data_iter,test=False):  
    #caculate the accuracy on specified dataset
    if isinstance(net, torch.nn.Module):
        net.eval()  #Essential!
    metric = Accumulator(3)  #sum_of_loss_of_all_events, sum_of_the_number_of_correct_predictions, all_events
    y_pred_test=[]
    y_true_test=[]
    with torch.no_grad():
        for batch in data_iter:
            batch = [bat.cuda() for bat in batch]
            y_hat=net(*batch[:-1]) 
            l=loss(y_hat,batch[-1])
            metric.add(l.sum(), accuracy(y_hat, batch[-1]), torch.tensor(batch[-1].numel()).cuda())
            if test:
                y_pred_test.append(F.softmax(y_hat,dim=1))
                y_true_test.append(batch[-1])
        if test:
            y_pred_test=torch.cat(y_pred_test,dim=0)
            y_true_test=torch.cat(y_true_test,dim=0)
    if test:
        #score, true, loss, acc 
        return y_pred_test,y_true_test,metric[0] / metric[2], metric[1] / metric[2]
    else:
        dist.reduce(metric[0],0,op=dist.ReduceOp.SUM)
        dist.reduce(metric[1],0,op=dist.ReduceOp.SUM)
        dist.reduce(metric[2],0,op=dist.ReduceOp.SUM)
        #loss, acc 
        return metric[0] / metric[2], metric[1] / metric[2]

###########################################################################################################################################################################

def train_procedure_in_each_epoch_GPUorCPU(net, train_set, loss, optimizer, GPU_mode, local_rank):
    #sum_of_loss_of_all_events, sum_of_the_number_of_correct_predictions, all_events
    metric=Accumulator(3)
    if local_rank==0:
        pbar=tqdm(train_set)
    net.train()
    for batch in train_set:
        batch = [bat.cuda() for bat in batch] if GPU_mode else batch
        optimizer.zero_grad()
        y_hat=net(*batch[:-1]) 
        l=loss(y_hat,batch[-1])
        l.mean().backward()
        optimizer.step()
        ################################################
        b_loss=l.sum()
        b_acc=accuracy(y_hat, batch[-1])
        if GPU_mode:
            b_num=torch.tensor(batch[-1].numel()).cuda()
            dist.reduce(b_loss,0,op=dist.ReduceOp.SUM)
            dist.reduce(b_acc,0,op=dist.ReduceOp.SUM)
            dist.reduce(b_num,0,op=dist.ReduceOp.SUM)
        else:
            b_num=torch.tensor(batch[-1].numel())
        metric.add(b_loss, b_acc, b_num)
        if local_rank==0:
            pbar.set_description(f'batch_acc:{float(b_acc/b_num):.3f}, batch_loss:{float(b_loss/b_num):.3f}')
            pbar.update(1)
    #loss_train, acc_train
    return metric[0] / metric[2], metric[1] / metric[2]

###########################################################################################################################################################################

def evaluate_accuracy_GPUorCPU(net, loss, data_iter, GPU_mode,test=False):  
    #caculate the accuracy on specified dataset
    if isinstance(net, torch.nn.Module):
        net.eval()  #Essential!
    metric = Accumulator(3)  #sum_of_loss_of_all_events, sum_of_the_number_of_correct_predictions, all_events
    y_pred_test=[]
    y_true_test=[]
    with torch.no_grad():
        for batch in data_iter:
            batch = [bat.cuda() for bat in batch] if GPU_mode else batch
            y_hat=net(*batch[:-1]) 
            l=loss(y_hat,batch[-1])
            metric.add(l.sum(), accuracy(y_hat, batch[-1]), torch.tensor(batch[-1].numel()).cuda() if GPU_mode else torch.tensor(batch[-1].numel()))
            if test:
                y_pred_test.append(F.softmax(y_hat,dim=1))
                y_true_test.append(batch[-1])
        if test:
            y_pred_test=torch.cat(y_pred_test,dim=0)
            y_true_test=torch.cat(y_true_test,dim=0)
    if test:
        #score, true, loss, acc 
        return y_pred_test,y_true_test,metric[0] / metric[2], metric[1] / metric[2]
    else:
        if GPU_mode:
            dist.reduce(metric[0],0,op=dist.ReduceOp.SUM)
            dist.reduce(metric[1],0,op=dist.ReduceOp.SUM)
            dist.reduce(metric[2],0,op=dist.ReduceOp.SUM)
        #loss, acc 
        return metric[0] / metric[2], metric[1] / metric[2]

###########################################################################################################################################################################
 
def save_net(net,suffix):
    if not os.path.exists('./output/net_params/'):
        os.system('mkdir -p ./output/net_params')
    torch.save(net.state_dict(),f'./output/net_params/net_{suffix}.params')
    print(' save_net info '.center(81,'*'),end='\n\n')
    print(f'net has been saved in [{os.getcwd()}/output/net_params/net_{suffix}.params]\n')

###########################################################################################################################################################################

def save_net_hyperparameters(suffix,**kwargs):
    if not os.path.exists('./output/net_hyperparameters/'):
        os.system('mkdir -p ./output/net_hyperparameters')
    assert isinstance(kwargs,dict), 'Please input hyperparameters in key word only fomat... '
    with open(f'./output/net_hyperparameters/net_{suffix}.json','w') as f:
        json.dump(kwargs, f, indent=4)
    print(' save_net_hyperparameters info '.center(81,'*'),end='\n\n')
    print(f'net has been saved in [{os.getcwd()}/output/net_hyperparameters/net_{suffix}.json]\n')
        
###########################################################################################################################################################################

def load_net(net=None,suffix=None,hyperparams_dir_path=None,net_dir_path=None):
    """Note: require net.params ---> net_{suffix}.params
             require hyperparams ---> net_{suffix}.json

    Args:
        net (str): 'pfn' 'pn' 'pt'
        suffix (str): suffix in that round training
        hyperparams_dir_path (str): dir path where net_{suffix}.json locate
        net_dir_path (str): dir path where net_{suffix}.params locate

    Raises:
        Exception: Please check dir_path should not have slash in the end. 
    """
    net=net.lower()
    assert net in ['pfn','pn','pt'], 'net choice should be pfn、pn、pt !!!'

    if net=='pfn':
        net_dir_path=net_dir_path if net_dir_path is not None else r'../ParticleFlowNet/output/net_params/'  
        net_path=net_dir_path+f'/net_{suffix}.params'
        hyperparams_dir_path = hyperparams_dir_path if hyperparams_dir_path is not None else r'../ParticleFlowNet/output/net_hyperparameters'
        hyperparams_path=hyperparams_dir_path+f'/net_{suffix}.json'
        with open(hyperparams_path,'r') as f:
            hyperparams=json.load(f)
        net=ParticleFlowNet.ParticleFlowNetwork(**hyperparams)

    elif net=='pn':
        net_dir_path=net_dir_path if net_dir_path is not None else r'../ParticleNet/output/net_params/'  
        net_path=net_dir_path+f'/net_{suffix}.params'
        hyperparams_dir_path = hyperparams_dir_path if hyperparams_dir_path is not None else r'../ParticleNet/output/net_hyperparameters'
        hyperparams_path=hyperparams_dir_path+f'/net_{suffix}.json'
        with open(hyperparams_path,'r') as f:
            hyperparams=json.load(f)
        net=ParticleNet.ParticleNet(**hyperparams)

    else:
        raise Exception('Dont have ParticleTransformer yet...')
    
    weights_candid=torch.load(net_path)
    weights=collections.OrderedDict()
    for k,v in weights_candid.items():
        k=k.replace('module.', '')#DDP method will add "module" in each layer
        weights[k]=v
    net.load_state_dict(weights)
    print('load net successfully'.center(81,'*'))
    print()
    return net

###########################################################################################################################################################################
































