# -*- coding: utf-8 -*-
"""
I:\DataSet\GLAS\论文1-实验\DF_DGT__GLAS\0413-1415__GET\0607-2042\0619-1925_GET\0628-1037
/home/jcq/python_code/GLAS/GNT_InstanceSeg/Instance_Seg/Diffu_G_Trans

2023.11.17 1114
代码加密上传Github
本地：I:\分割小论文项目-211103\数据集和代码\GLAS\DFGET_encryption\EFNet_train
服务器：/home/jcq/python_code/GLAS/DFGET_encryption/EFNet_train
"""
# =============================================================================
"""--------------------------------- 加载库 --------------------------------"""
import os
import sys
import time
import numpy as np
import cv2
import random 
from tqdm import tqdm
import torch as th
# import torch.nn.functional as F
# import math
import torch.nn as nn
# import dgl.ops as dglops
import dgl.function as fn
from dgl.data.utils import load_graphs
# from dgl.sampling import sample_neighbors #select_topk, 
# import skimage
from skimage import morphology #, measure, io
# from metrics import ObjectDice
import labelme
print('torch version: ',th.__version__)
#
def setup_seed(random_seed):
    # 实验重现设置
    th.manual_seed(random_seed)  # 为CPU设置随机种子
    th.cuda.manual_seed(random_seed)
    th.cuda.manual_seed_all(random_seed)  # 为所有GPU设置随机种子
    random.seed(random_seed)
    np.random.seed(random_seed)
# setup_seed(2021)
"""--------------------------------- 加载库 --------------------------------"""
# =============================================================================



# =============================================================================
"""-----------------------------文件路径-----------------------------"""
current_dir = os.getcwd()  # 获取当前工作目录路径
sys.path.append(current_dir)
parent_dir = os.path.dirname(current_dir)  # 获取当前目录的上一级目录路径
sys.path.append(current_dir)
data_dir = parent_dir + '/data' # jcq
model_dir = current_dir + '/params'
model_dir_DF = parent_dir +'/DFNet_train/params' # 存位移网络参数
device = th.device('cuda:0')
#
# GST+ graph
img_graphs, _ = load_graphs(data_dir + '/graph_GST+_6144nodes_square17x17.bin') 
bg8 = img_graphs[0] # 
src_rowidx = bg8.edata.pop('src_rowidx')
src_colidx = bg8.edata.pop('src_colidx')
dst_rowidx = bg8.edata.pop('dst_rowidx')
dst_colidx = bg8.edata.pop('dst_colidx')
InfoIdx = th.stack((src_rowidx, src_colidx, dst_rowidx, dst_colidx), dim=1) # -> (num_edges, 4)
bg8.edata['InfoIdx'] = InfoIdx
_ = bg8.edata.pop('distance')
_ = bg8.ndata.pop('coords')
#
# 计算位移场
patch_graphs, _ = load_graphs(data_dir + '/graph_GET_6144nodes_r4.bin')
bg4 = patch_graphs[0] # 4x4大小的patch级的节点
_ = bg4.edata.pop('dst_rowidx')
_ = bg4.edata.pop('dst_colidx')
src_rowidx = bg4.edata.pop('src_rowidx') # 使用了源节点的分发机制
src_colidx = bg4.edata.pop('src_colidx')
InfoIdx_src = th.stack((src_rowidx, src_colidx), dim=1) # -> (num_edges, 2)
bg4.edata['InfoIdx_src'] = InfoIdx_src
"""-----------------------------文件路径-----------------------------"""
# =============================================================================



#=======================================================================================================================
"""--------------------------------- 定义模型 --------------------------------"""
#=======================================================================================================================
from DFNets import DFNet
from GET_softmax_no_global import EFNet # 
# from GET_EFNet import EFNet # 
#
net_DVF = DFNet((model_dir, model_dir_DF), device).to(device) # 参数已冻结
net_DVF.eval()
net = EFNet(data_dir, device).to(device)
#
# 数据增强
import albumentations as A
aug_old = A.Compose([
    A.Resize(256, 384, interpolation=cv2.INTER_LINEAR, p=1),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=30, p=0.5), 
    A.OneOf([A.ElasticTransform(),
             A.GridDistortion(),
             A.OpticalDistortion()], p=1),
    A.OneOf([
             A.ColorJitter(),
             A.RandomGamma(),
             A.RandomBrightnessContrast()], p=1),
    A.GaussianBlur(p=0.5),
 ])
# =============================================================================
resize_384x384 = A.Resize(384, 384, interpolation=cv2.INTER_LINEAR, p=1) # 缩放到固定尺寸
# 输入原始图片img(3, H0=256, W0=384), 旋转角度angle°.
def resize_rotate_aug(img, label, step_time): 
    img = img.numpy().astype('uint8')
    label = label.numpy().astype('uint8') # 实例标签
    # num_class0 = len(np.unique(label)) # 变换前的class数
    img = img.transpose(1, 2, 0) # -> (H, W, C)
    # 按 p=0.5 决定是否执行新的数据增强
    act_newAug = np.random.normal(0, 1, 1)[0]
    if act_newAug >= 0: 
        augmented = resize_384x384(image=img, mask=label) # 
        img = augmented['image']
        label = augmented['mask']
        img_label = np.concatenate((img, np.expand_dims(label, 2)), axis=2) # 最后一个通道是标签
        #
        angle = np.clip(np.random.normal(0, 180, 1), -540, 540)[0] # 正态分布(均值0，标准差180)，生成随机旋转角度
        H, W = img_label.shape[:2] # (384, 384)
        # 填充图像边界，旋转后再截取中心 HxW区域
        # convet angle into rad 角度转弧度
        rangle = np.deg2rad(angle)  # angle in radians 
        # calculate new image width and height
        NW = (abs(np.sin(rangle)*H) + abs(np.cos(rangle)*W)) # 新图宽度
        NW = max(NW, W + 16)
        NH = (abs(np.cos(rangle)*H) + abs(np.sin(rangle)*W)) # 
        NH = max(NH, H + 16)
        pad1 = int((NH - H)//2) # 上下各填充 pad1
        pad2 = int((NW - W)//2) # 左右各填充 pad2
        imglabelPad = cv2.copyMakeBorder(img_label, pad1, pad1, pad2, pad2, cv2.BORDER_REFLECT) # 镜像
        #
        rows, cols = imglabelPad.shape[:2] # assert rows==NH, cols==NW
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1) # 旋转中心(cols / 2, rows / 2)，逆时针旋转angle°，缩放比例1
        imglabelR = cv2.warpAffine(imglabelPad, M, (cols, rows))
        imglabelR = imglabelR[pad1:-pad1, pad2:-pad2] # -> (384, 384, 4)
        imgR, labelR = imglabelR[:,:,:3], imglabelR[:,:,3] # -> (384, 384, 3)  (384, 384)
    else:
        imgR, labelR = img, label # (256, 384, 3) (256, 384)
    #
    augmented_old = aug_old(image=imgR, mask=labelR)
    X = augmented_old['image'] # (256, 384, 3)
    y = augmented_old['mask'] # (256, 384)
    # num_class = len(np.unique(y)) # 变换后的class数
    # assert num_class <= num_class0 # 
    # 把数据增强的图片和标签打印出来看看
    # if step_time==1:
    #     cv2.imwrite(model_dir +'/visual_DataAug/imgR.png', X)
    #     labelme.utils.lblsave(model_dir +'/visual_DataAug/imgR_anno.png', y)
    #
    X = X.transpose(2, 0, 1) # -> (C, H, W)
    X = th.from_numpy(X) #
    X = (X.float() / 255).to(device)
    X = X.unsqueeze(0) # -> (1, C, H, W)
    #
    y = th.from_numpy(y).float()
    y = y.to(device).unsqueeze(0) # -> (1, H, W) | y 实例标签
    return X, y # 返回旋转后的图片、标签
# =============================================================================
#=======================================================================================================================
"""--------------------------------- 定义模型 --------------------------------"""
#=======================================================================================================================



# =============================================================================
"""--------------------------------- 常用函数 --------------------------------"""
def findfile(datapath, prefix='fold0'): # 查找含特定前缀的文件名
    datanames = os.listdir(datapath)
    file_name0 = None
    for dataname in datanames:
        if dataname.split('_')[0] == prefix:
            file_name0 = dataname
            break
    return file_name0
# 将训练过程的指标存入TXT
def write_txt(content, filepath, mode="a"): # 以追加的模式写入TXT
    with open(filepath, mode, encoding='utf-8') as f:
        f.write(str(content[0])+'\n') # 第一行，图片数
        for i in range(1, len(content)):
            f.write(str(content[i])+'\n')
"""--------------------------------- 常用函数 --------------------------------"""
# =============================================================================



# =============================================================================
"""--------------------------------- 训练验证 --------------------------------"""
def visual_DF(DF_label_i, df_hat_i, y_binary, bound_mask, imgname, DFId): # df_hat_i (2, H, W) 
    if DF_label_i != None:
        np.save(model_dir + '/visual_DF/%s_dflabel_%s.npy'%(imgname, DFId), DF_label_i.cpu().numpy())
        DFlen = th.sqrt(th.sum(DF_label_i**2, 0))
        DFlen = DFlen.cpu().numpy() # -> (H, W)
        label_hue = np.uint8(np.round(179 * DFlen / DFlen.max())) # 坑：若 DF 仍然是uint8，则179*DF会导致数值越界
        blank_ch = 255 * np.ones_like(label_hue)
        color_DF = cv2.merge([label_hue, blank_ch, blank_ch])
        color_DF = cv2.cvtColor(color_DF, cv2.COLOR_HSV2BGR)
        color_DF[label_hue==0] = 0 #
        color_DF[(y_binary==1)*(label_hue==0)] = 255 #
        cv2.imwrite(model_dir + '/visual_DF/%s_DFlabel_%s.png'%(imgname, DFId), color_DF) # 
    # 位移长可视化
    df_hat_i = df_hat_i.cpu().numpy()
    np.save(model_dir + '/visual_DF/%s_dfhat_%s.npy'%(imgname, DFId), df_hat_i)
    DFlen = np.sqrt(np.sum(df_hat_i**2, axis=0)) # -> (H, W) 
    label_hue = np.uint8(np.round(179 * DFlen / DFlen.max())) # 坑：若 TF 仍然是uint8，则179*DF会导致数值越界
    blank_ch = 255 * np.ones_like(label_hue)
    color_DF = cv2.merge([label_hue, blank_ch, blank_ch])
    color_DF = cv2.cvtColor(color_DF, cv2.COLOR_HSV2BGR)
    color_DF[label_hue==0] = 0 #
    color_DF[(y_binary==1)*(label_hue==0)] = 255 #
    color_DF[bound_mask==1] = 255 # 边缘设为白色
    cv2.imwrite(model_dir + '/visual_DF/%s_DFhat_%s.png'%(imgname, DFId), color_DF)
###############################################################################
#   
###############################################################################
def evaluate_score(bg4, bg8, valid_iter, test_imgnames, net, epoch): 
    iou_sum, n = 0, 0 # 
    for imgname, Xy in zip(test_imgnames, valid_iter):
        X, y = Xy[:,:3,:,:], Xy[:,3,:,:]
        X = (X.float()/255).to(device) 
        y = y.float().to(device) # -> (B, 128, 192)
        #
        semantic_label = (y > 0.5).float().squeeze() # 
        ###########################################
        df_hat, bg8 = net_DVF(X, bg4, bg8) # df_hat(2, 256, 384) | 
        y_hat = net(X, bg8) # 
        semantic_seg = th.round(y_hat) # 原图尺寸的语义分割结果
        ###########################################
        n += len(y)
        iou_sum += IoU(semantic_seg.squeeze(), semantic_label.squeeze()).item()
        #
        # 可视化 ###########################################
        # 可视化 ###########################################
    valN_iou = iou_sum/n
    return valN_iou # 
###############################################################################


###############################################################################
def IoU(yhat, gy): # yhat(256, 384)  gy(256, 384)
    assert yhat.shape == (256, 384)
    assert gy.shape == yhat.shape 
    interArea  = th.sum(yhat*gy)
    Area1 = th.sum(yhat)
    Area2 = th.sum(gy)
    iou = interArea / (Area1 + Area2 - interArea)
    return iou
#
def L2IoULoss(yhat, gy): # yhat(256, 384)  gy(256, 384)
    l2 = (yhat-gy)**2 # -> (256, 384)
    l2loss = th.mean(l2)
    #
    interArea = th.sum(yhat*gy) # -> (1,)
    Area1 = th.sum(yhat) # -> (1,)
    Area2 = th.sum(gy) # -> (1,)
    iousoft = (interArea + 1e-19) / (Area1 + Area2 - interArea + 1e-19) # -> (1,)
    #
    pred_img = th.round(yhat)
    iou = IoU(pred_img, th.round(gy))
    #
    return 2*l2loss - iousoft, iou
###############################################################################

   
###############################################################################
def train(bg4, bg8, train_iter, valid_iter, test_imgnames, batch_size, net, fold, num_epochs, lr):
    optimizer = th.optim.Adam(net.parameters(), lr=lr) 
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=6) # max模式，指标停止增加时，lr将衰减
    #
    valN_iou0 = 0.5
    train_log = [] # 存训练验证过程各项指标
    bg4 = bg4.to(device)
    bg8 = bg8.to(device) # 
    for epoch in range(num_epochs):  
        traN_l_sum, traN_iou_sum, n, time0 = 0, 0, 0, time.time()
        # 常修改 ########################################################
        net.train() # 训练模式
        step_time = 0
        for X_y in tqdm(train_iter): # X_y(B, 4, 128, 192)
            #
            # 数据增强 ##################################
            X, y = X_y[0,:3,:,:], X_y[0,3,:,:]  # -> (C, H, W) (H, W)
            X, y = resize_rotate_aug(X, y, step_time) # 
            step_time += 1
            #
            semantic_label = (y > 0.5).float().squeeze()
            # 数据增强 ##################################
            #
            ################################################
            df_hat, bg8 = net_DVF(X, bg4, bg8) # df_hat(2, 256, 384) | 
            #
            y_hat = net(X, bg8) # 
            ################################################
            loss, iou = L2IoULoss(y_hat.squeeze(), semantic_label)
            ################################################
            optimizer.zero_grad() # 梯度归零
            loss.backward() # 
            optimizer.step() #
            ################################################
            traN_l_sum += loss.item()*len(y) # Node损失
            traN_iou_sum += iou.item()*len(y)
            n += len(y)
            #
            # break
        # 常修改 ########################################################
        #
        net.eval()
        with th.no_grad(): # 推理阶段不要梯度
            traN_loss = traN_l_sum/n
            traN_iou = traN_iou_sum/n
            #
            valN_iou = evaluate_score(bg4, bg8, valid_iter, test_imgnames, net, epoch)
            current_lr = optimizer.state_dict()['param_groups'][-1]['lr'] # 当前epoch的学习率
            scheduler.step(valN_iou) # 学习率衰减
            ###########################################################################################
            epochs = "Epoch {:05d} | lr {:.8f} | Time(s) {:.4f} | traN_loss {:.4f} | \n".format(
                        epoch, current_lr, time.time()-time0, traN_loss)
            trains1 = "| traN_iou {:.4f} | valN_iou {:.4f} \n".format(traN_iou, valN_iou)
            #
            print(epochs + trains1 + '# ============= \n#\n# =============')
            train_log.append(epochs + trains1 + '# ============= \n#\n# =============')
            write_txt(train_log[-1], model_dir +'/train_log_fold%s.txt'%fold, mode="a")
            #
            if valN_iou0 < valN_iou: #
                file_name = model_dir + '/EFnet.pth.tar'
                th.save({'state_dict': net.state_dict()}, file_name)
                valN_iou0 = valN_iou
        # 训练结束，写入日志
        write_txt(train_log, model_dir +'/train_log_fold%s.txt'%fold, mode="w")
"""--------------------------------- 训练验证 --------------------------------"""
# =============================================================================




# =============================================================================
"""----------------------------- 训练 ----------------------------"""
if True:
    # 超参数 ###################
    num_epochs = 300
    lr = 0.001
    batch_size = 1
    ###########################
    #
    # 模型导入参数
    param_path = model_dir + '/EFNet.pth.tar'
    # 如果字典中存在一些模型不需要的参数，或者模型需要的参数在字典中不存在，则可使用 strict=False 来避免报错。
    net.load_state_dict(th.load(param_path, map_location='cpu')['state_dict'], strict=False)
    # lr = 1e-5
    #
    #
    train_data = np.load(data_dir + '/train_Img_InLabel.npy') # 
    train_data = th.from_numpy(train_data)
    train_data = train_data.permute(0, 3, 1, 2)
    assert train_data.shape == (85, 4, 256, 384)
    print('train_data.shape: ', train_data.shape)
    #
    test_data = np.load(data_dir + '/test_Img_InLabel.npy') # 
    test_data = th.from_numpy(test_data)
    test_data = test_data.permute(0, 3, 1, 2)
    assert test_data.shape == (80, 4, 256, 384)
    print('test_data.shape: ', test_data.shape) 
    #
    test_imgnames = np.load(data_dir + '/imgnames_for_test_Img_InLabel_original.npy')
    #
    train_iter = th.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False) 
    test_iter = th.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False) 
    # #
    #
    for fold in range(3,4):
        print('train fold %d'%fold)
        print('random seed %d'%(2021*fold))
        setup_seed(2021*fold) 
        train(bg4, bg8, train_iter, test_iter, test_imgnames, batch_size, net, fold, num_epochs, lr)
"""----------------------------- 训练 ----------------------------"""
# =============================================================================
    
    

    


    