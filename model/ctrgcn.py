import math
import torch
import sys
sys.path.append("/home/data/qiao/FR-headc/TD-FCH/")

import numpy as np
from torch.autograd import Variable
from model.modules import *
from model.ML import ST_FeatureCalibration
from TDFL import TDFL


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

class Model(nn.Module):
    def build_basic_blocks(self):
        A = self.graph.A  # 3,25,25
        self.l1 = TCN_GCN_unit(self.in_channels, self.base_channel, A, residual=False, adaptive=self.adaptive)
        self.l2 = TCN_GCN_unit(self.base_channel, self.base_channel, A, adaptive=self.adaptive)
        self.l3 = TCN_GCN_unit(self.base_channel, self.base_channel, A, adaptive=self.adaptive)
        self.l4 = TCN_GCN_unit(self.base_channel, self.base_channel, A, adaptive=self.adaptive)
        self.l5 = TCN_GCN_unit(self.base_channel, self.base_channel * 2, A, stride=2, adaptive=self.adaptive)
        self.l6 = TCN_GCN_unit(self.base_channel * 2, self.base_channel * 2, A, adaptive=self.adaptive)
        self.l7 = TCN_GCN_unit(self.base_channel * 2, self.base_channel * 2, A, adaptive=self.adaptive)
        self.l8 = TCN_GCN_unit(self.base_channel * 2, self.base_channel * 4, A, stride=2, adaptive=self.adaptive)
        self.l9 = TCN_GCN_unit(self.base_channel * 4, self.base_channel * 4, A, adaptive=self.adaptive)
        self.l10 = TCN_GCN_unit(self.base_channel * 4, self.base_channel * 4, A, adaptive=self.adaptive)

    def ml_block(self):
        if self.fch == "Metric-loss":
            self.ren_fin = ST_FeatureCalibration(self.base_channel * 4, self.num_frame // 4, self.num_point, self.num_person, n_class=self.num_class, version=self.ml_version, pred_threshold=self.pred_threshold, use_p_map=self.use_p_map, fp_loss=self.fp_loss)
        else:
            raise KeyError(f"no such Feature Calibration Head {self.fch}")

    def __init__(self,
                 # Base Params
                 num_class=60, num_point=25, num_frame=64, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 base_channel=64, drop_out=0, adaptive=True,
                 # Module Params
                 fch=None, ml_version='V0', pred_threshold=0, use_p_map=True, fp_loss=0.5
                 ):
        super(Model, self).__init__()

        self.num_class = num_class
        self.num_point = num_point
        self.num_frame = num_frame
        self.num_person = num_person
        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
        self.in_channels = in_channels
        self.base_channel = base_channel
        self.drop_out = nn.Dropout(drop_out) if drop_out else lambda x: x
        self.adaptive = adaptive
        self.fch = fch
        self.ml_version = ml_version
        self.pred_threshold = pred_threshold
        self.use_p_map = use_p_map
        self.fp_loss = fp_loss
        self.TDFL_low = TDFL(self.base_channel, self.num_frame)
        self.TDFL_mid = TDFL(self.base_channel, self.num_frame)
        self.TDFL_high = TDFL(self.base_channel * 2, self.num_frame // 2)
        self.TDFL_fin = TDFL(self.base_channel * 4, self.num_frame // 4)

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.build_basic_blocks()

        if self.fch is not None:
            self.ml_block()

        self.fc = nn.Linear(self.base_channel * 4, self.num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def Metric_loss_output(self, x, fin, label):
        logits = self.fc(x)
        ml_loss = self.ren_fin(fin, label.detach(), logits.detach())
        return logits, ml_loss

    def forward(self, x, label=None, get_metric_loss=False, get_hidden_feat=False, **kwargs):

        if get_hidden_feat:
            return self.get_hidden_feat(x)

        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.TDFL_low(x)

        x = self.l2(x)
        x = self.TDFL_mid(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.TDFL_high(x)

        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.TDFL_fin(x)

        x = self.l9(x)
        x = self.l10(x)
        fin = x.clone()

        # N*M,C,T*V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)


        if get_metric_loss and self.fch == "Metric-loss":
            return self.Metric_loss_output(x, fin, label)

        return self.fc(x)
