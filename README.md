# MEISTER
Multiscale and integrative analysis of tissue and single cells using mass spectrometry with deep learning reconstruction.

This is the code repository for the paper "Multiscale biochemical mapping of the brain through deep-learning-enhanced high-throughput mass spectrometry", **Nature Methods, 2024** [link](https://www.nature.com/articles/s41592-024-02171-3)

:construction: **We are expanding our code documentations to make it friendly** :construction:

## Installation
### via Anaconda (recommended way)
Create a conda enviroment:
```
conda env create -f environment.yml
conda activate MEISTER
```

## Set up MEISTER
* The documentation for training MEISTER signal models for reconstruction can be found [here](https://github.com/richardxie1119/MEISTER/blob/main/document/MEISTER_doc.pdf). 
* The complete computational protocol for reconstruction and downstream multiscale data analysis can be found in the online Supplementary Information [here](https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-024-02171-3/MediaObjects/41592_2024_2171_MOESM1_ESM.pdf)
  
## Multiscale data analysis notebooks

The Notebooks for the multifaceted data analysis can be found [here](https://github.com/richardxie1119/multiscale_analysis). Detailed instructions for reproducing the analysis with these notebooks are provided in the [Supplementary Protocol](https://github.com/richardxie1119/MEISTER/blob/main/Supplementary_Protocol.pdf). 

| Script | Description |
| --- | --- |
| [coronal3D.ipynb](https://github.com/richardxie1119/multiscale_analysis/blob/main/coronal3D.ipynb) | Post processing of the reconstructed 3D coronal data. |
| [embed3D.ipynb](https://github.com/richardxie1119/multiscale_analysis/blob/main/embed_3D.ipynb) and [embed_3DCoronal](https://github.com/richardxie1119/multiscale_analysis/blob/main/embed_3DCoronal.ipynb) | Parametric UMAP to obtain coherent feature images for sagittal and coronal data sets. |
| [regmri_atlas_coronal.ipynb](https://github.com/richardxie1119/multiscale_analysis/blob/main/regmri_atlas_coronal.ipynb) | Brain regional analysis of registered 3D coronal data with MRI atlas. |
| [tissue_sc_mapping_deepmsi.ipynb](https://github.com/richardxie1119/multiscale_analysis/blob/main/tissue_sc_mapping_deepmsi.ipynb) | Integrative analysis of 3D MSI data with 13K single-cell MS data from five brain regions. |
| [tissue_sc_mapping_deepmsi_hip.ipynb](https://github.com/richardxie1119/multiscale_analysis/blob/main/tissue_sc_mapping_deepmsi_hip.ipynb) | Integrative analysis of hippocampus MSI data with single-cell MS on hippocampal cells. |
| [MEISTER_eval.ipynb](https://github.com/richardxie1119/multiscale_analysis/blob/main/MEISTER_eval.ipynb) | Evaluation of MEISTER models on experimental MSI data sets. |
| [MEISTER_simulation.ipynb](https://github.com/richardxie1119/multiscale_analysis/blob/main/MEISTER_simulation.ipynb) | Evaluation of MEISTER models on simulated MSI data sets. |
| [MEISTER_singlecell.ipynb](https://github.com/richardxie1119/multiscale_analysis/blob/main/tissue_sc_mapping_deepmsi_hip.ipynb) | Evaluation of MEISTER models on experimental single-cell MS data sets.|



 ## Example data
Data can be accessed [here](https://doi.org/10.13012/B2IDB-9740536_V1). Raw transient files need to be downloaded 
separately from [this link](https://uofi.box.com/s/oagdxhea1wi8tvfij4robj0z0w8wq7j4) and placed 
into the .d folder within the Bruker data folder.
- `saved_model.zip`: trained model weights (decoder and regressor) on high-resolution MSI data used in the manuscript. 
- `20210930_ShortTransient_S3_5.zip`: serial sections of rat coronal sections acquired with low mass resolution as input for reconstruction in the manuscript.
- `coronal3D_propagated_decoded.h5`: reconstructed and processed MSI data of serial sections of rat coronal sections.
- `peak_data_sc_regional_decode.imzML`: peak data of 13,566 single rat brain cells.
- `coronal3D_slide3_5_R00_propagated_decoded.imzML`: peak data of a representative rat brain coronal section.


