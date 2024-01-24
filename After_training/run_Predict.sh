#! /bin/bash
###### Part 1 ######
#SBATCH --partition=gpu
#SBATCH --account=higgsgpu
#SBATCH --job-name=predict
#SBATCH --qos=debug
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=4096
#SBATCH --gres=gpu:v100:2
#SBATCH --output=/hpcfs/cepc/higgsgpu/wuzuofei/WZF_ML/After_training/output/my_log/predict_zf/pfn.log

###### Part 2 ######

 ulimit -d unlimited
 ulimit -f unlimited
 ulimit -l unlimited
 ulimit -n unlimited
 ulimit -s unlimited
 ulimit -t unlimited
 srun -l hostname

 /usr/bin/nvidia-smi -L

 echo "Allocate GPU cards : ${CUDA_VISIBLE_DEVICES}"
 nvidia-smi
 # >>> conda initialize >>>
 # !! Contents within this block are managed by 'conda init' !!
 __conda_setup="$('/hpcfs/cepc/higgsgpu/wuzuofei/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
 if [ $? -eq 0 ]; then
     eval "$__conda_setup"
 else    
     if [ -f "/hpcfs/cepc/higgsgpu/wuzuofei/miniconda3/etc/profile.d/conda.sh" ]; then
         . "/hpcfs/cepc/higgsgpu/wuzuofei/miniconda3/etc/profile.d/conda.sh"
     else   
         export PATH="/hpcfs/cepc/higgsgpu/wuzuofei/miniconda3/bin:$PATH"
     fi      
 fi
 unset __conda_setup
 # <<< conda initialize <<<
 
 conda activate weaver

 which python 

python -u script_Predict/apply_and_predict.py zf/pfn
