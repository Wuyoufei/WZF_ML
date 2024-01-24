Please append file "ZF_ML" to your Environment Variable to run these scripts.

PFN and PN file are the same, but hyperparameters are different so I just split them to two files.

IF the GPU is easily accessible for you, just focus on the [GPUonly] file, since my laptop has no GPU, CPU version also provided---> GPUorCPU

GPUorCPU file can run on GPU or CPU, but the training procedure is slower than GPUonly... So they are only applied for test.

========================================================================

To run the code:

Change infomation of root/hyperparameters of net in training in my_train_DDP.py

Just ./start_GPU.sh

Check the settings in sbatch script

Input the suffix for this round training

Input "l" or "s" where "l" means running locally(for test), "s" means sbatch

Note:--> for GPUonly script, just input "s"

========================================================================

ParticleNeXt still has some bugs need to be fixed.
ParticleTransformer 老板这周不让我玩,下次再说



















