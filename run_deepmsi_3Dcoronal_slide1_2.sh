#!/bin/bash
export PATH=%PATH:/mnt/c/Users/Richard/Anaconda3/envs/deepmsi

out_dir="E:/3DCoronal/decoded_data_new"
path_files="./file_dir_coronal_3D_slide1_2.json"
decoder_dir="D:/Work/projects/DeepMSI/saved_model/coronal3D_latent32_epoch10_decoder"
regressor_dir="D:/Work/projects/DeepMSI/saved_model/coronal3D_latent32_epoch10_regressor"
recon_ROI=(
    "R06"
    "R07"
    "R08"
)
mz_range="150 1100"
if_process_raw="True"
if_simu='False'
embedding='True'

for ((i=0;i<${#recon_ROI[@]};++i)); do
    printf "Processing data for region %s ...\n" "${recon_ROI[i]}"
    python.exe deep_recon.py  --path_file $path_files --out_dir $out_dir --embedding $embedding --recon_ROI ${recon_ROI[i]} --decoder_dir $decoder_dir --regressor_dir $regressor_dir --mz_range $mz_range  --if_process_raw $if_process_raw --if_simu $if_simu
done