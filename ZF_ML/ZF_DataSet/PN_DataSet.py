#  __        __  _____  _____ 
#  \ \      / / |__  / |  ___|
#   \ \ /\ / /    / /  | |_   
#    \ V  V /    / /_  |  _|  
#     \_/\_/    /____| |_|    
#
#   ParticleNet--->DataSet 
# Please Read the description before running 
#                               ---2023-12-1
#
import torch
from torch.utils import data
import uproot4 as uproot
import numpy as np
from ZF_ML.ZF_utils import data_process

class PN_Dataset(data.Dataset):
    def __init__(self, filename, num_data, features, points_features, tree_name, root_dir_path, map_PID=False, PID_idx=None) -> None:
        """ParticleNet dataset class, one file represents a unique category !!!

        Args:
            filename (a list of str): a list of root files

            num_data (int or a list of int): number of events of each category, 
                                             if int, all the category will be the same, 
                                             if list, len(list) should be equal to len(filename)

            features (a list of str): a list of features

            points_features (a list of str): coordinates features

            tree_name (str or a list of str): tree name in root file, 
                                             if str, all the root file will be the same, 
                                             if list, len(list) should be equal to len(filename)

            root_dir_path (regexp, optional): file path.

            map_PID (bool): whether to map PID to float, 
                            if True, please input PID_idx as a int number in the features list.

            PID_idx (int): the idx of PID in the features list.
        """
        super().__init__()
        self.tree_name=tree_name
        if map_PID==True and (not isinstance(PID_idx,int)):
            raise Exception("Please input the index of PID in the list of features if you want to map PID, and NOTE the type should be int.")
        
        self.points,self.features,self.labels=self.load_data(filename, num_data, features, points_features, root_dir_path)

        if map_PID==True:
            data_process.remap_pids(self.features, pid_i=PID_idx, error_on_unknown=False)

        #preprocess
        #self.Acos(idx=0)#cosTheta -> Theta
        #my_tools_PN.remap_pids(self.features,pid_i=0,error_on_unknown=False)#optional, remap pid to a float number

        self.points=torch.from_numpy(self.points).type(torch.float32)
        self.features=torch.from_numpy(self.features).type(torch.float32)
        self.labels=torch.from_numpy(self.labels).type(torch.long)
    
    def __getitem__(self, index):
        return self.points[index],self.features[index],self.labels[index]
    
    def __len__(self):
        return len(self.features)


    def load_data(self,filename,num_data,features,points_features,root_dir_path):
        dataset_features=[]
        dataset_points_features=[]
        dataset_labels=[]
        ic=0
        stepsize=1_000
        for idx,fn in enumerate(filename):
            num=0
            events = uproot.open(root_dir_path+"/"+fn+f":{self.tree_name if isinstance(self.tree_name,str) else self.tree_name[idx]}")
            for array in events.iterate(step_size=stepsize, entry_stop=num_data if isinstance(num_data,int) else num_data[idx]):
                event_number=len(array[features[0]])
                num+=event_number
                step_features=[]
                step_points_features=[]
                for feature in features:
                    step_features.append(np.array(array[feature]).reshape(event_number,1,-1))
                for points_feature in points_features:
                    step_points_features.append(np.array(array[points_feature]).reshape(event_number,1,-1))

                dataset_features.append(np.concatenate(step_features, axis=1))
                dataset_points_features.append(np.concatenate(step_points_features, axis=1))
                dataset_labels.append(np.ones(event_number)*ic)

                if num%10000==0:
                    print('{:*^80}'.format(f'load tuples successfully ----->{ic}, {fn}, {num}'))

            ic+=1
        dataset_features=np.concatenate(dataset_features,axis=0)
        dataset_points_features=np.concatenate(dataset_points_features,axis=0)
        dataset_labels=np.concatenate(dataset_labels,axis=0)

        #just shuffle
        idx=np.random.permutation(len(dataset_labels))
        dataset_points_features=dataset_points_features[idx]
        dataset_features=dataset_features[idx]
        dataset_labels=dataset_labels[idx]
        return dataset_points_features, dataset_features, dataset_labels
    
    def Acos(self,idx):
        MASK=(np.abs(self.features).sum(axis=1,keepdims=False) != 0)
        self.points[:,idx,:]=np.arccos(self.points[:,idx,:])*MASK
        print('Done! cosTheta -> Theta')

if __name__=='__main__':
    print('qwe')


    













































