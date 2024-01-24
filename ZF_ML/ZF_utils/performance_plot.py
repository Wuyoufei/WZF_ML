#  __        __  _____  _____ 
#  \ \      / / |__  / |  ___|
#   \ \ /\ / /    / /  | |_   
#    \ V  V /    / /_  |  _|  
#     \_/\_/    /____| |_|          2023-12-1
#                                    
#   Tools to plot to indicate the model's performance   
#     

import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, auc
from numpy import interp
import itertools
import numpy as np
import os
import sys
import torch.distributed as dist
import time
import mplhep as hep
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import colors
    matplotlib.use('Agg')
except:
    print('please install matploltib in order to make plots')
    plt = False

###########################################################################################################################################################################

def one_hot_encoding(true_label): #1D
    r'''
    true_label supposed to starts from zero !!!
    Pytorch的torch.nn.functional.one_hot无法处理ndarray，本函数可以接收ndarray or tensor
    返回tensor
    '''
    one_hot=torch.zeros(num:=len(true_label),true_label.max()+1) #python version>=3.8
    one_hot[range(num),true_label]=1
    return one_hot

###########################################################################################################################################################################

def plot_confusion_matrix(suffix, num_classes, true_label, pred_label, processes="", normalize=True, cmap=plt.cm.Greens):
    """plot confusion matrix

    Args:
        suffix (str): suffix
        num_classes (int): num_classes
        true_label (ndarray): 1D (num_events, )
        pred_label (ndarray): 1D (num_events, )
        processes (list of str): PROCESSES
        normalize (bool, optional): Defaults to True.
        cmap (plt.cm.xxx): color, Defaults to plt.cm.Greens.
    """
    plt.figure(figsize=(10, 10))
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('white')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.autolayout'] = True
    cm = confusion_matrix(true_label, pred_label)
    cm=cm.astype('float')
    num_of_each_mode=np.sum(cm,axis=1)
    for i in range(len(cm)):
        cm[i]/=float(num_of_each_mode[i])
    with np.printoptions(precision=4, suppress=True):
        print(cm)
        print("%8.4f" % np.linalg.det(cm))
    if not os.path.exists('./output/conf_mat_npy/'):
        os.system('mkdir -p ./output/conf_mat_npy')
    np.save('./output/conf_mat_npy/conf_%s.npy'% suffix, cm)

    label_font = {'size': '14'}
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title("Confusion Matrix of Higgs decays", fontdict=label_font)
    plt.colorbar()
    if processes == "":
        processes = ["cc", "bb", r"$\mu \mu$", r"$\tau \tau$",
                   "gg", r"$\gamma\gamma$", "ZZ", "WW", r"$\gamma Z$"]
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, processes, rotation=45)
    plt.yticks(tick_marks, processes)
    fmt = '.2%' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True', fontdict=label_font)
    plt.xlabel('Predicted', fontdict=label_font)
    if not os.path.exists('./output/figs/'):
        os.system('mkdir -p ./output/figs')
    plt.savefig("./output/figs/conf_%s.pdf" % suffix, dpi=800)
    print(' plot_confusion_matrix info '.center(81,'*'),end='\n\n')
    print(f'conf_matrix.npy has been saved in {os.getcwd()}/output/conf_mat_npy/conf_{suffix}.npy')
    print(f'conf_matrix.pdf has been saved in {os.getcwd()}/output/figs/conf_{suffix}.pdf\n')

###########################################################################################################################################################################

def plot_roc(suffix, num_classes, true_label, scores, processes=''):
    """roc curve

    Args:
        suffix (str): SUFFIX
        num_classes (int): num_classes
        true_label (ndarray): 1D (num_events,)
        scores (ndarray): 2D (num_events, num_classes)
        processes (list of str): 1D processes, Defaults to ''.
    """
    lw=2
    true_label=one_hot_encoding(true_label)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(true_label[:,i], scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(true_label.ravel(), scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= num_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(10, 10))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.autolayout'] = True
    plt.plot(tpr["micro"], 1-fpr["micro"],
             label='Micro-average ROC (AUC = {0:0.3f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(tpr["macro"], 1-fpr["macro"],
             label='Macro-average ROC (AUC = {0:0.3f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    COLORS = itertools.cycle(['black', 'red', 'blue', 'darkorange', 'green',
                    'brown', 'cyan', 'purple', 'darkblue', 'pink'])
    for i, color in zip(range(num_classes), COLORS):
        plt.plot(tpr[i], 1-fpr[i], color=color, lw=lw,
                 label=r'ROC of {0} (AUC = {1:0.3f})'
                 ''.format(processes[i], roc_auc[i]))

    plt.plot([0, 1], [1, 0], 'k--', lw=lw)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Signal eff.')
    plt.ylabel('Bkgrnd rej.')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower left")
    if not os.path.exists('./output/figs/'):
        os.system('mkdir -p ./output/figs')
    plt.savefig("./output/figs/roc_%s.pdf" % suffix)
    print(' plot_roc info '.center(81,'*'),end='\n\n')
    print(f'ROC.pdf has been saved in {os.getcwd()}/output/figs/roc_{suffix}.pdf\n')

###########################################################################################################################################################################

def plot_scores_hist(suffix,processes, true_label, scores, bins=20):
    """在验证模型时，把测试集所有事例在 H→bb 类上的得分带标签地画在直方图中，可以得到如图4-7所示的分布图-----摘自本人的本科大便

    Args:
        processes (list of str): PROCESSES
        label (ndarray): 1D (batchsize,)
        scores (ndarray): 2D (batchsize,num_classes)
        bins (int):  Defaults to 20.
    """
    #true_label=(batchsize,)
    #scores=(batchsize,num_classes)

    if not os.path.exists('./output/figs/scores_hist/'):
        os.system('mkdir -p ./output/figs/scores_hist')

    print(' plot_scores_hist info '.center(81,'*'),end='\n\n')

    for class_idx, process in enumerate(processes):
        print(f'processing plot {process}...')
        list_of_scores=[np.array([]) for _ in processes]
        for entry_idx in range(len(true_label)):
            print(f'processing plot {process}--> accounting scores -->event_idx:{entry_idx}\r',end='')
            list_of_scores[true_label[entry_idx]]=np.append(list_of_scores[true_label[entry_idx]], scores[entry_idx][class_idx])
        print()
        print(f'processing plot {process}-->filling hist and drawing...')
        list_of_hists=[np.histogram(list_of_scores[i], bins=bins,range=(0.,1.) ) for i in range(len(processes))]
        fig,ax=plt.subplots()
        plt.yscale('log')
        ax.set_xlim(-0.1,1.1)
        ax.set_xlabel("Score")
        ax.set_ylabel("Events")
        ax.text(0.75, 0.85, process, transform=ax.transAxes, fontsize=22)
        for plt_class_idx, plt_process in enumerate(processes):
            hep.histplot(list_of_hists[plt_class_idx],label=plt_process)
        ax.legend(handlelength=0.5, ncol=3,loc='upper center')
        fig.savefig(f"./output/figs/scores_hist/{process}_{suffix}.pdf")
        print(f'{process}_{suffix}.pdf has been saved in {os.getcwd()}/output/figs/scores_hist/{process}_{suffix}.pdf\n')

###########################################################################################################################################################################




















