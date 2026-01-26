cd FNFM-main/FNFM
CUDA_VISIBLE_DEVICES=0 python 1Dmain_torchcfm.py --expIndex 1 --basemodel v_GWN --targetDataset collab --mode AE_CFM --denoise='cfmTransformer' --train_objective='cfm' --ae_arch layer_transformer --cfm_epochs 3000 --ae_epochs 1000
