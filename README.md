# DFGET: Displacement-Field-Assisted Graph Energy Transmitter for Gland Instance Segmentation

## Introduction
This repository is the official implementation of the paper, "DFGET: Displacement-Field-Assisted Graph Energy  
Transmitter for Gland Instance Segmentation".

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







