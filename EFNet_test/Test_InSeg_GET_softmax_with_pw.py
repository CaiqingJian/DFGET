# -*- coding: utf-8 -*-
"""
I:\DataSet\GLAS\论文1-实验\消融实验\GET_softmax_no_global\0715-1637_GET_softmax_with_pw\Test_Instance_Seg
/home/jcq/python_code/GLAS/GNT_InstanceSeg/Instance_Seg/Ablation/GET_softmax_no_global/with_pw

2023.11.17 1114
代码加密上传Github
本地：I:\分割小论文项目-211103\数据集和代码\GLAS\DFGET_encryption\EFNet_test
服务器：/home/jcq/python_code/GLAS/DFGET_encryption/EFNet_test
"""
# =============================================================================
"""--------------------------------- 加载库 --------------------------------"""
import os
import sys
import time
import numpy as np
import cv2
# import random 
# import collections
from tqdm import tqdm
import torch as th
# import torch.nn.functional as F
# import math
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.data.utils import load_graphs
# from dgl.sampling import sample_neighbors #select_topk, 
# import skimage
from skimage import measure, morphology #, io
from metrics import ObjectDice
import labelme
print('torch version: ',th.__version__)
#
"""--------------------------------- 加载库 --------------------------------"""
# =============================================================================



# =============================================================================
"""-----------------------------文件路径-----------------------------"""
current_dir = os.getcwd()  # 获取当前工作目录路径
sys.path.append(current_dir)
parent_dir = os.path.dirname(current_dir)  # 获取当前目录的上一级目录路径
data_dir = parent_dir + '/data' # jcq
# model_dir = current_dir + '/params'
model_dir = parent_dir + '/EFNet_train/params'
model_dir_DF = parent_dir +'/DFNet_train/params' # 存位移网络参数
save_dir = current_dir +'/save_dir'
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
#
"""-----------------------------文件路径-----------------------------"""
# =============================================================================



#=======================================================================================================================
"""--------------------------------- 定义模型 --------------------------------"""
#=======================================================================================================================
from DFNets import DFNet
from GET_softmax_no_global import EFNet # 
#
net_DVF = DFNet((model_dir, model_dir_DF), device).to(device) # 参数已冻结
net_DVF.eval()
net = EFNet(data_dir, device).to(device)
#=======================================================================================================================
"""--------------------------------- 定义模型 --------------------------------"""
#=======================================================================================================================


#=======================================================================================================================
"""--------------------------------- 位移场收缩实例，反传实例标签 --------------------------------"""
#=======================================================================================================================
def coordinates(sp=0, step=1, h=256, w=384): # 
    xs = np.arange(sp, h, step).reshape((-1, 1)) 
    ys = np.arange(sp, w, step).reshape((1, -1)) 
    xs = np.repeat(xs, w, axis=1)
    ys = np.repeat(ys, h, axis=0)
    cds = np.stack([xs, ys], axis=-1) # -> (h, w, 2)
    cds = th.from_numpy(cds).float().reshape((-1, 2))
    return cds
#
class Graph_Cluster_Module(nn.Module): # 后处理
    def __init__(self, semantic_thr=0.5, dflen_thr=3, disk_radius=1, Hight=256, Width=384):
        super(Graph_Cluster_Module, self).__init__()
        self.y_thr = semantic_thr # 前景阈值
        self.dflen_thr = dflen_thr # 位移长阈值
        self.disk_radius = disk_radius # 开运算的核半径
        self.Hight = Hight
        self.Width = Width
        self.coords = coordinates(0, 1, Hight, Width).to(device) # 节点坐标 -> (256*384, 2)
        self.maxpool = nn.MaxPool2d(3, stride=1, padding=1)
        self.avgpool = nn.AvgPool2d(5, stride=1, padding=2, count_include_pad=False)
    #
    def coords_fg(self, DFin, DFlen, src_cds):
        DF_unit = DFin / (DFlen.unsqueeze(1) + 1e-19) 
        DFlen = DFlen.clamp(self.dflen_thr, 6) # 限定位移长范围
        DFin = DF_unit * DFlen.unsqueeze(1)
        dst_cds = src_cds + DFin # 目标节点坐标
        dst_cds[:,0] = dst_cds[:,0].clamp(0, self.Hight - 1) # -> (256*384,) 
        dst_cds[:,1] = dst_cds[:,1].clamp(0, self.Width - 1) # -> (256*384,)
        src_coords = th.round(src_cds).long() # 坐标一定要先取整，再转成一维索引
        dst_coords = th.round(dst_cds).long()
        src_ids = src_coords[:,0]*self.Width + src_coords[:,1] # W是图片宽度，高度:H=256，宽度:W=384
        dst_ids = dst_coords[:,0]*self.Width + dst_coords[:,1] # 
        tg = dgl.graph((src_ids, dst_ids), device=device) # 由源节点向目标节点移动
        tg.ndata['nids'] = src_ids # 节点ID 
        return tg
    #
    def forward(self, y_hat, df_hat): # coords(256*384, 2) | y_hat(256, 384) | df_hat(2, 256, 384)
        with th.no_grad():
            semantic_mask = (y_hat > self.y_thr).float() # -> (256, 384)
            # (2) 方法2：位移场前传反传 #############################################
            # 根据位移场构建 dst 到 src 的graph
            DF = df_hat.reshape((2, -1)).permute(1,0) # -> (256*384, 2)
            DFlen = th.sqrt(th.sum(DF**2, 1)) 
            #
            fg = self.coords_fg(DF, DFlen, self.coords) # 向实例内部收缩
            # 位移场前传 #####################################################
            fore_mask = semantic_mask.reshape((-1,)) # -> (256*384,) | 完整的前景mask
            fg.ndata['y'] = fore_mask
            for i in range(2):
                fg.update_all(fn.copy_u('y', 'm'), fn.sum('m', 'y')) # 背景节点的传入全是零，相加结果y是零；而前景边缘的入度为零，故无传入，使用默认的零
            y = fg.ndata.pop('y').reshape((self.Hight, self.Width)) # -> (256, 384)
            y = self.avgpool(y.unsqueeze(0)) # -> (1, 256, 384) # 滤波去噪
            # 位移场前传 #####################################################
            #
            # 给分开的实例标号 ################################################
            eroded_mask_np = (y >= 0.5).squeeze().cpu().numpy().astype('uint8') # 密度大于0.5即为前景
            #
            openkernel = morphology.disk(self.disk_radius, np.uint8)
            eroded_mask_np = cv2.erode(eroded_mask_np, openkernel, iterations=1)
            #
            eroded_instance = measure.label(eroded_mask_np, connectivity=2) # connectivity=1表示4连通
            eroded_instance = th.from_numpy(eroded_instance).float().to(device)
            # eroded_instance = self.maxpool(eroded_instance.unsqueeze(0)).squeeze() # -> (256, 384)
            instance_A = eroded_instance.reshape((-1,)) # -> (256*384,)
            #
            # 实例反传，并用 fore_mask 限定在前景 ###############################
            rg = dgl.reverse(fg) # 取反向图，每条edge的方向逆转，其余不变
            rg.ndata['Inst'] = instance_A # 分开的实例编号矩阵
            for i in range(8): # 实例反传能有效移除一些实例边缘的孤立小patch，而孤立小patch是导致F1指标较低的主要原因
                rg.update_all(fn.copy_u('Inst', 'm'), fn.max('m', 'Inst')) # 反传时，多个节点反传到一个节点
            rg.ndata['Inst'] = rg.ndata['Inst'] * fore_mask # fore_mask 限定反传在前景
            #
            # 有向图消息传递移除小面积实例 #######################################
            InG = dgl.graph((rg.ndata['nids'], rg.ndata['Inst'].long()), device=device)
            InG.ndata['fgrd'] = (rg.ndata['Inst'] > 0.5).float() # 节点的前景值
            InG.update_all(fn.copy_u('fgrd', 'm'), fn.sum('m', 'fgrd_sum'))
            InG.ndata['Inst'] = rg.ndata['Inst']
            InG.update_all(fn.copy_u('Inst', 'm'), fn.max('m', 'Inst_m'))
            InG.ndata['Inst'] = (InG.ndata['fgrd_sum'] > 256).float() * InG.ndata['Inst_m']
            reverse_InG = dgl.reverse(InG)
            reverse_InG.update_all(fn.copy_u('Inst', 'm'), fn.max('m', 'Inst_r'))
            rg.ndata['Inst'] = reverse_InG.ndata['Inst_r']
            # 有向图消息传递移除小面积实例 #######################################
            #
            # 输出最终实例分割结果
            instance_seg = rg.ndata['Inst'].reshape(semantic_mask.shape) # -> (256, 384) 
            instance_seg = instance_seg.cpu().numpy()
            # (2) 方法2：位移场前传反传 #############################################
            return instance_seg
# =============================================================================
# =============================================================================
GCM = Graph_Cluster_Module(semantic_thr=0.5, dflen_thr=1, disk_radius=1).to(device)
#=======================================================================================================================
"""--------------------------------- 位移场收缩实例，反传实例标签 --------------------------------"""
#=======================================================================================================================



# =============================================================================
"""--------------------------------- 测试实例分割 --------------------------------"""
def IoU(yhat, gy): # yhat(256, 384)  gy(256, 384)
    assert yhat.shape == (256, 384)
    assert gy.shape == yhat.shape 
    interArea  = th.sum(yhat*gy)
    Area1 = th.sum(yhat)
    Area2 = th.sum(gy)
    iou = interArea / (Area1 + Area2 - interArea)
    return iou
# =============================================================================
# =============================================================================
def evaluate_score(bg4, bg8, valid_iter, test_imgnames, net, epoch): 
    dice_sum, iou_sum, n = 0, 0, 0 # 
    for imgname, Xy in tqdm(zip(test_imgnames, valid_iter)):
        X, y = Xy[:,:3,:,:], Xy[:,3,:,:]
        X = (X.float()/255).to(device) 
        y = y.float().to(device) # -> (B, 256, 384)
        #
        semantic_label = (y > 0.5).float().squeeze() # 
        ###########################################
        df_hat, bg8 = net_DVF(X, bg4, bg8) # df_hat(2, 256, 384) | 
        y_hat = net(X, bg8) # -> edge_hat(num_edges, numlayer)
        semantic_seg = th.round(y_hat) # 原图尺寸的语义分割结果
        ###########################################
        n += len(y)
        iou_sum += IoU(semantic_seg.squeeze(), semantic_label.squeeze()).item()
        #
        df_hat = df_hat.squeeze() #.to(device)
        instance_seg_dfm = GCM(y_hat, df_hat) # 位移场前传和反传
        dice_sum += ObjectDice(instance_seg_dfm, y.squeeze().cpu().numpy()) 
        #
        # 可视化 ###########################################
        instance_seg = instance_seg_dfm.astype('uint8')
        cv2.imwrite(save_dir +'/instance_seg/%s.png'%imgname, instance_seg) # 使用 instance_seg.png计算分割的三个指标
        # 可视化 ###########################################
    valN_iou = iou_sum/n
    valN_dice = dice_sum/n
    return valN_iou, valN_dice
###############################################################################
"""--------------------------------- 测试实例分割 --------------------------------"""
# =============================================================================




# =============================================================================
"""----------------------------- 训练 ----------------------------"""
if True:
    param_path = model_dir + '/EFNet.pth.tar'
    net.load_state_dict(th.load(param_path, map_location='cpu')['state_dict'], strict=False)
    print(param_path)
    #
    #
    test_data = np.load(data_dir + '/test_Img_InLabel.npy') # 
    test_data = th.from_numpy(test_data)
    test_data = test_data.permute(0, 3, 1, 2)
    assert test_data.shape == (80, 4, 256, 384)
    print('test_data.shape: ', test_data.shape) 
    #
    test_imgnames = np.load(data_dir + '/imgnames_for_test_Img_InLabel_original.npy')
    #
    test_iter = th.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, drop_last=False) 
    #
    #
    net.eval()
    with th.no_grad():
        bg4 = bg4.to(device)
        bg8 = bg8.to(device)
        valN_iou, valN_dice = evaluate_score(bg4, bg8, test_iter, test_imgnames, net, 0)
        print('valN_iou: %.4f, valN_dice: %.4f'%(valN_iou, valN_dice))
"""----------------------------- 训练 ----------------------------"""
# =============================================================================
    
    

    


    