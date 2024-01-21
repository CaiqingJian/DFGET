"""
"""

#=======================================================================================================================
"""--------------------------------- unet_parts --------------------------------"""
#=======================================================================================================================
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.data.utils import load_graphs
import dgl.function as fn
from dgl.sampling import sample_neighbors # select_topk #,

#=======================================================================================================================
"""--------------------------------- ConvEncoder_parts --------------------------------"""
#=======================================================================================================================
# =============================================================================
# =============================================================================
def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1, groups=1, slope=1e-2):
    return nn.Sequential(
        nn.LeakyReLU(slope),
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=groups),
        nn.BatchNorm2d(out_num, affine=True, track_running_stats=False) 
        )
def lkrelu_conv(in_num, out_num, kernel_size=3, padding=1, stride=1, groups=1, slope=1e-2):
    return nn.Sequential(
        nn.LeakyReLU(slope),
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=groups)
        )
def conv_lkrelu(in_num, out_num, kernel_size=3, padding=1, stride=1, groups=1, slope=1e-2):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=groups),
        nn.LeakyReLU(slope)
        )
# =============================================================================
# =============================================================================
# Residual block
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.layer0 = conv_batch(in_channels, in_channels*4, kernel_size=3, padding=1, groups=in_channels) # 
        self.layer1 = conv_batch(in_channels*4, out_channels, kernel_size=1, padding=0, groups=1) # 
    def forward(self, x):
        skip = x
        out = self.layer0(x)
        out = self.layer1(out)
        return out + skip
# =============================================================================
# =============================================================================
class Inc(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=2, kernel_size=3, padding=1, stride=1):
        super(Inc, self).__init__()
        self.conv1 = conv_lkrelu(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        # self.conv1 = conv_batch(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride) # conv_lkrelu
        conv2s = [] # 
        for i in range(num_blocks):
            conv2s.append( DoubleConv(out_channels, out_channels) )
        self.conv2 = nn.Sequential(*conv2s)
    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)
# =============================================================================
# =============================================================================
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=1, kernel_size=3, padding=0, stride=2):
        super(Down, self).__init__()
        self.pad1 = nn.ReplicationPad2d((1,0,1,0)) # 
        self.conv1 = lkrelu_conv(in_channels, in_channels, kernel_size=kernel_size, padding=0, stride=stride, groups=in_channels) # 
        # self.conv1 = conv_batch(in_channels, in_channels, kernel_size=kernel_size, padding=0, stride=stride) # 
        self.pad2 = nn.ReplicationPad2d((0,1,1,0)) # 
        self.conv2 = lkrelu_conv(in_channels, in_channels, kernel_size=kernel_size, padding=0, stride=stride, groups=in_channels) #
        # self.conv2 = conv_batch(in_channels, in_channels, kernel_size=kernel_size, padding=0, stride=stride) # 
        self.pad3 = nn.ReplicationPad2d((1,0,0,1)) # 
        self.conv3 = lkrelu_conv(in_channels, in_channels, kernel_size=kernel_size, padding=0, stride=stride, groups=in_channels) # 
        # self.conv3 = conv_batch(in_channels, in_channels, kernel_size=kernel_size, padding=0, stride=stride) # 
        self.pad4 = nn.ReplicationPad2d((0,1,0,1)) # 
        self.conv4 = lkrelu_conv(in_channels, in_channels, kernel_size=kernel_size, padding=0, stride=stride, groups=in_channels) # 
        # self.conv4 = conv_batch(in_channels, in_channels, kernel_size=kernel_size, padding=0, stride=stride) # 
        self.fuse = conv_batch(in_channels*4, out_channels, kernel_size=1, padding=0, stride=1) # 
        # 
        resconvs = [] 
        for i in range(num_blocks):
            resconvs.append( DoubleConv(out_channels, out_channels) )
        self.resconvs = nn.Sequential(*resconvs)
    def forward(self, x):
        x1 = self.conv1(self.pad1(x)) # -> (B, in_c, H/2, W/2)
        x2 = self.conv2(self.pad2(x)) # -> (B, in_c, H/2, W/2)
        x3 = self.conv3(self.pad3(x)) # -> (B, in_c, H/2, W/2)
        x4 = self.conv4(self.pad4(x)) # -> (B, in_c, H/2, W/2)
        xs = th.cat((x1, x2, x3, x4), dim=1) # -> (B, 4*in_c, H/2, W/2)
        xf = self.fuse(xs) # -> (B, out_channels, H/2, W/2)
        return self.resconvs(xf)
# =============================================================================
# =============================================================================
class Up(nn.Module): # 
    def __init__(self, in_channels=128, out_channels=64*4, num_blocks=2, k=3, pad=1): # 
        super(Up, self).__init__()
        self.conv_kxk = lkrelu_conv(in_channels, out_channels, kernel_size=k, padding=pad, stride=1, groups=out_channels//4, slope=0.1) # 
        # self.conv_kxk = conv_batch(in_channels, out_channels, kernel_size=3, padding=1, stride=1, groups=out_channels//4, slope=0.1) # 
        conv2s = [] # 
        for i in range(num_blocks):
            conv2s.append( DoubleConv(out_channels//4, out_channels//4) )
        self.conv2 = nn.Sequential(*conv2s)
    def forward(self, x): # (B, C, H, W)
        # (1) Upsample
        out = self.conv_kxk(x) # -> (B, 4C, H, W)
        B, C4, H, W = out.shape # C4==4*C
        out = out.reshape((B, C4//4, 2, 2, H, W))
        out = out.permute(0, 1, 4, 2, 5, 3) # -> (B, C4//4, H,2, W,2)
        out = out.reshape((B, C4//4, H*2, W*2))
        # (2) 
        return self.conv2(out) # -> (B, out_channels//4, H*2, W*2)
# =============================================================================
# =============================================================================
class DoubleConvUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvUp, self).__init__()
        self.layer0 = conv_batch(in_channels, out_channels, kernel_size=1, padding=0, groups=1)
        self.layer1 = lkrelu_conv(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels) # 
        # self.layer1 = conv_batch(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels) # 
    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        return out
# =============================================================================
# =============================================================================
class UpMerge(nn.Module): #
    def __init__(self, in_channels, out_channels, num_blocks=1):
        super(UpMerge, self).__init__()
        self.conv_3x3 = lkrelu_conv(in_channels, out_channels, kernel_size=3, padding=1, stride=1, groups=out_channels//4, slope=0.1) # 
        # self.conv_3x3 = conv_batch(in_channels, out_channels, kernel_size=3, padding=1, stride=1, groups=out_channels//4, slope=0.1) # 
        self.conv = DoubleConvUp(in_channels, in_channels//2)
        conv2s = [] # 
        for i in range(num_blocks):
            conv2s.append( DoubleConv(out_channels//4, out_channels//4) )
        self.conv2 = nn.Sequential(*conv2s)
        #
    def forward(self, x1, x2):
        # (1) Upsample
        x1 = self.conv_3x3(x1) # -> (B, 4C, H, W)
        B, C4, H, W = x1.shape # C4==4*C
        x1 = x1.reshape((B, C4//4, 2, 2, H, W))
        x1 = x1.permute(0, 1, 4, 2, 5, 3) # -> (B, C4//4, H,2, W,2)
        x1 = x1.reshape((B, C4//4, H*2, W*2))
        # input is BCHW
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffY // 2, diffY - diffY // 2,
                        diffX // 2, diffX - diffX // 2])
        x = th.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.conv2(x)
        return x
# =============================================================================
# =============================================================================
class ConvEncoder(nn.Module):
    def __init__(self, in_channels=3, dmodel=64):
        super(ConvEncoder, self).__init__()
        self.bn0 = nn.BatchNorm2d(in_channels, affine=True, track_running_stats=True, momentum=0.1)
        # self.bn1 = nn.BatchNorm2d(in_channels, affine=True, track_running_stats=False)
        self.inc = Inc(in_channels, dmodel, num_blocks=2, kernel_size=3, padding=1, stride=1) # 64
        self.down1 = Down(dmodel, dmodel*2, num_blocks=2, kernel_size=3, padding=0, stride=2) # 128
        self.down2 = Down(dmodel*2, dmodel*4, num_blocks=2, kernel_size=3, padding=0, stride=2) # 256
        self.down3 = Down(dmodel*4, dmodel*8, num_blocks=2, kernel_size=3, padding=0, stride=2) # 512
        self.up3 = UpMerge(dmodel*8, dmodel*16, num_blocks=2)
        #
    def forward(self, x): # x(B, 3, H, W)
        x = self.bn0(x) #+ self.bn1(x)
        x0 = self.inc(x) # -> (B, 64, H, W)
        x1 = self.down1(x0) # -> (B, 128, H/2, W/2)
        x2 = self.down2(x1) # -> (B, 256, H/4, W/4)
        x3 = self.down3(x2) # -> (B, 512, H/8, W/8)
        x2up = self.up3(x3, x2) # -> (B, 256, H/4, W/4)
        return x2up
#=======================================================================================================================
"""--------------------------------- ConvEncoder_parts --------------------------------"""
#=======================================================================================================================



#=======================================================================================================================
#=======================================================================================================================
class EncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_signal=1):
        super(EncoderLayer, self).__init__()
        self.d_signal = d_signal
        #
        self.PPM = nn.Sequential(
            conv_batch(d_model, d_model, kernel_size=17, padding=8, groups=d_model),
            conv_batch(d_model, d_model, kernel_size=1, padding=0, groups=1))
        self.FFN_1 = nn.Sequential(
            conv_batch(d_model, d_model, kernel_size=1, padding=0, groups=1),
            conv_batch(d_model, d_model, kernel_size=1, padding=0, groups=1))
        # 
        self.Dis = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 289*d_signal))
        self.act = nn.ReLU() # nn.Sequential(nn.LeakyReLU(5), nn.Sigmoid())
        self.bn = nn.BatchNorm1d(d_model, affine=True, track_running_stats=False) # 
        # 
        self.FFN_2 = DoubleConv(d_model, d_model) # 
        #
    def forward(self, bg, shape): # X(6144, 256)
        InfoIdx = bg.edata['InfoIdx']
        X = bg.ndata['X'] # (6144, 256)
        X = X.permute(1,0).reshape(shape) # -> (1, 256, 64, 96)
        # 
        X = X + self.PPM(X)
        X = X + self.FFN_1(X) # -> (1, 256, 64, 96)
        # 
        X = X.reshape((shape[1], -1)).permute(1,0) # -> (6144, 256)
        signal = self.Dis(X) # -> (6144, 289*d_signal)
        signal = signal.reshape((-1, 289, self.d_signal)) # -> (6144, 289, d_signal)
        src = signal[InfoIdx[:,0], InfoIdx[:,1]] # -> (num_edges, d_signal)
        dst = signal[InfoIdx[:,2], InfoIdx[:,3]] # -> (num_edges, d_signal)
        E = src + dst #
        #
        bg.edata['aij'] = th.exp(E.clamp(-5,5)) * bg.edata['edgmsk'] # 
        # bg.edata['aij'] = th.exp(E.clamp(-5,5)) # 
        bg.update_all(fn.copy_e('aij', 'm'), fn.sum('m', 'A_sum'))
        bg.ndata['X'] = X #
        X_skip = bg.ndata['X']
        bg.update_all(fn.u_mul_e('X', 'aij', 'm'), fn.sum('m', 'XA_sum')) # -> (6144, 256)
        X_trans = bg.ndata['XA_sum']  / (bg.ndata['A_sum'] + 1e-5) # -> (6144, 256)
        bg.ndata['X'] = X_skip + self.bn(X_trans) # -> (6144, 256)
        #
        # 
        X = bg.ndata['X'].permute(1,0).reshape(shape)  # -> (256, 6144) -> (1, 256, 64, 96)
        bg.ndata['X'] = self.FFN_2(X).reshape((shape[1], -1)).permute(1,0) # 
        #
        # return E
# =============================================================================
# 
# =============================================================================
class GSTransmitter(nn.Module):
    def __init__(self, d_model=256, d_signal=1, numlayer=6, data_dir=None, device=None):
        super(GSTransmitter, self).__init__()
        self.numlayer = numlayer
        self.encoderlist = nn.ModuleList()
        for li in range(numlayer):
            self.encoderlist.append( EncoderLayer(d_model=d_model, d_signal=d_signal) )
        self.edge_out = nn.Sigmoid() 
        #
    def forward(self, X, bg, shape): # X(6144, 256)
        # Es = [] 
        with bg.local_scope():
            bg.ndata['X'] = X
            for i in range(self.numlayer):
                self.encoderlist[i](bg, shape)
                # Es.append(E)
            nfeat = bg.ndata['X'] # 
        # bg.edata['E'] = E # 
        # Es = th.cat(Es, dim=1) # -> (num_edges, numlayer)
        # edge_hat = self.edge_out(Es)
        return nfeat
###################################################################
#
#
###################################################################
class GETLayer_inter1(nn.Module):
    def __init__(self, d_model=256):
        super(GETLayer_inter1, self).__init__()
        self.d_model = d_model #
        self.Dis = nn.Linear(d_model, d_model//16, bias=False)
        self.act = nn.ReLU() # 
    def forward(self, bg2): # 
        Xe = bg2.ndata['Xe']
        q = self.Dis(Xe) # -> (6208, 16)
        bg2.ndata['q'] = q / 4 # 
        bg2.apply_edges(fn.u_dot_v('q', 'q', 'E'))
        E = bg2.edata['E']
        bg2.edata['aij'] = self.act(E) # 
        bg2.edata['rij'] = self.act(-E) # 
        bg2.update_all(fn.copy_e('aij', 'm'), fn.sum('m', 'A_sum')) # 
        bg2.update_all(fn.copy_e('rij', 'm'), fn.sum('m', 'R_sum')) # 
        denominator = bg2.ndata['A_sum'] + bg2.ndata['R_sum'] + 1e-2 # 
        bg2.update_all(fn.u_mul_e('Xe', 'aij', 'm'), fn.sum('m', 'XeA_sum')) # -> (6208, 256)
        Xe_trans = (bg2.ndata['XeA_sum'] + bg2.ndata['Xe']*bg2.ndata['R_sum']) / denominator # -> (6208, 256) 
        #
        return Xe_trans # 
# =============================================================================
# =============================================================================
class GETLayer_inter2(nn.Module):
    def __init__(self, d_model=256):
        super(GETLayer_inter2, self).__init__()
        self.getlayer_inter1 = GETLayer_inter1()
        self.bn = nn.BatchNorm1d(d_model, affine=True, track_running_stats=False) #
        self.FFN = nn.Sequential(
            nn.Linear(d_model, d_model, bias=True), 
            nn.BatchNorm1d(d_model, affine=True, track_running_stats=False),
            nn.ReLU(), 
            nn.Linear(d_model, d_model, bias=True),
            nn.BatchNorm1d(d_model, affine=True, track_running_stats=False))
        #
    def forward(self, bg2, num_dst): # 
        Xe_skip = bg2.ndata['Xe'][:num_dst] # -> (6144, 256)
        Xe_trans = self.getlayer_inter1(bg2)
        Xe = Xe_skip + self.bn(Xe_trans[:num_dst]) # -> (6208, 256) 
        return Xe + self.FFN(Xe) # 
# =============================================================================
# =============================================================================
class GET_inter(nn.Module): #
    def __init__(self, data_dir=None, device=None, numlayer=1, d_model=256):
        super(GET_inter, self).__init__()
        self.device = device
        self.cpu = th.device('cpu')
        Trans_graphs, _ = load_graphs(data_dir + '/graph_GET_6144+384nodes.bin') 
        self.Tg = Trans_graphs[0].to(device) # 
        self.num_src = 384 # 
        self.numlayer = numlayer
        self.d_model = d_model # 
        # self.fuse_encoderlist = nn.ModuleList()
        # for li in range(numlayer):
            # self.fuse_encoderlist.append( GETLayer_inter1(d_model=d_model) )
        self.trans_encoder = GETLayer_inter2(d_model=d_model)     
    ####################################################################
    def forward(self, Xe, bg, shape): # 
        num_dst = Xe.shape[0] # 
        with th.no_grad():
            exclude_eid = th.nonzero((bg.edata['E'].squeeze() <= 0).int()).squeeze() # 
            sg = sample_neighbors(bg, bg.nodes(), -1, exclude_edges=exclude_eid, edge_dir='in') # 
            sg.update_all(fn.copy_e('E', 'm'), fn.mean('m', 'Div')) #
            sg.apply_edges(fn.u_sub_v('Div', 'Div', 'detDiv')) # 
            sg.edata['pr'] = th.relu(sg.edata['detDiv']).squeeze()
            sg.update_all(fn.copy_e('pr', 'm'), fn.mean('m', 'pr')) # 
            _, SRC_ids = th.topk(sg.ndata['pr'], self.num_src, dim=0, largest=False, sorted=True) # 
        SRC = Xe[SRC_ids] #
        with self.Tg.local_scope():
            self.Tg.ndata['Xe'] = th.cat((Xe, SRC), dim=0) # -> (6144+384, 256)
            nfeat = self.trans_encoder(self.Tg, num_dst) # -> (6144, 256)
        return nfeat
###################################################################
#
#
###################################################################
class EFNet(nn.Module):
    def __init__(self, data_dir=None, device=None, dmodel=256):
        super(EFNet, self).__init__()
        self.dmodel = dmodel
        self.Embedding = ConvEncoder()
        self.Transmitter = GSTransmitter(d_model=dmodel, d_signal=1, numlayer=6, data_dir=data_dir, device=device) # 
        # self.get_inter = GET_inter(data_dir, device) #
        #
        self.semantic = nn.Sequential(
            Up(dmodel, dmodel*2, num_blocks=2),
            Up(dmodel//2, dmodel, num_blocks=2),
            nn.Conv2d(dmodel//4, 1, kernel_size=1), nn.Sigmoid())
        #
    def forward(self, X, bg): # X(1, 3, 256, 384)
        Ft = self.Embedding(X) # -> (1, 256, 64, 96)
        shape = Ft.shape # (1, 256, 64, 96)
        # GST+
        Ft = Ft.reshape((self.dmodel, -1)).permute(1,0) # -> (6144, 256)
        nfeat = self.Transmitter(Ft, bg, shape) # 
        # GET_inter
        # nfeat = self.get_inter(nfeat, bg, shape) # -> (6144, 256)
        nfeat = nfeat.permute(1,0).reshape(shape) # -> (1, 256, 64, 96)
        # 
        semantic_out = self.semantic(nfeat) # -> (1, 1, 256, 384)
        #
        return semantic_out.squeeze() # 
#=======================================================================================================================
#=======================================================================================================================









