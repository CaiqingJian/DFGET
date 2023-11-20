# DFGET: Displacement-Field-Assisted Graph Energy Transmitter for Gland Instance Segmentation

## Introduction
This repository is the official implementation of the paper, "DFGET: Displacement-Field-Assisted Graph Energy  
Transmitter for Gland Instance Segmentation".

## Dataset
The GlaS dataset used in this paper comes from the Gland Challenge of MICCAI 2015 [<sup>1</sup>](#refer-anchor-1), including 165 images,  
of which 85 are training set and 80 are test set.

## Installation
### Step 1: 
Install packages according to requirements.txt.  
### Step 2: 
Download the 8 sub-files from DFGET_encryption_aa to DFGET_encryption_ah and put these 8 files in the same folder ./DFGET.  
Then use the following commands to merge the 8 sub-files into a whole and unzip it:  
```she
cat DFGET_encryption_* > DFGET_encryption.zip 
unzip -o DFGET_encryption.zip
```
Use cd to enter the "DFGET_encryption" folder, Execute the following commands in sequence to complete the training,  
testing, and scoring


## Training of DFNet
```she
cd DFNet_train
python dist/train_DFNets.py
```

## Visualization of displacement fields
```she
cd DFNet_train
python dist/visual_dfhat_arrow.py
```
![示例图片](https://example.com/image.jpg)

## Training of EFNet
```she
cd EFNet_train
python dist/train_EFNet.py
```

## Inference of EFNet
```she
cd EFNet_test
python dist/Test_InSeg_GET_softmax_with_pw.py
```

## Score of three metrics
```she
cd metrics
python dist/metrics522x768.py
```

## REFERENCES
<div id="refer-anchor-1"></div>
- [1] K. Sirinukunwattana et al., “Gland segmentation in colon histology images: The glas challenge contest,” Medical image analysis, vol. 35, pp. 489-502, Jan. 2017. 





