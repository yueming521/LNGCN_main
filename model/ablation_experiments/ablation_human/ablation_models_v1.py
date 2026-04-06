import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import dgl
from dgl.nn import GraphConv, Set2Set
from ncps.torch import CfC, LTC
from ncps.wirings import AutoNCP, NCP, FullyConnected, Wiring
import numpy as np

import os
DEBUG_SHAPES = os.getenv("PPI_DEBUG_SHAPES", "0") == "1"

ode_unfolds = 2
nhid = 256
nhidh = 128
nhidhh = 64
dropout = 0.4
time_steps = 5

class CfCCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, is_first_layer=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.is_first_layer = is_first_layer
        input_concat_dim = input_dim if is_first_layer else (input_dim + hidden_dim)
        self.backbone = nn.Sequential(
            nn.Linear(input_concat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), 
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), 
            nn.SiLU()
        )
        self.f_head = nn.Linear(hidden_dim, hidden_dim)
        self.g_head = nn.Linear(hidden_dim, hidden_dim)
        self.h_head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x_prev, feat, t):
        if self.is_first_layer:
            B = self.backbone(feat)
        else:
            combined = torch.cat([x_prev, feat], dim=-1)
            B = self.backbone(combined)
        ft = torch.sigmoid(-self.f_head(B) * t)
        dynamic_feat = self.g_head(B)
        static_feat = self.h_head(B)
        return ft * dynamic_feat + (1 - ft) * static_feat

class PreprocessCfC(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.cells = nn.ModuleList([
            CfCCell((in_dim - 1) if i == 0 else hidden_dim, hidden_dim, is_first_layer=(i == 0))
            for i in range(n_layers)
        ])

    def forward(self, feat):
        if feat.shape[1] > self.hidden_dim:
            base_feat = feat[:, :-1] 
            timestep_feat = feat[:, -1:]  
        else:
            base_feat = feat
            timestep_feat = torch.zeros(feat.size(0), 1, device=feat.device, dtype=feat.dtype)
        h = torch.zeros(base_feat.size(0), self.hidden_dim, device=feat.device, dtype=feat.dtype)
        x_in = base_feat
        for i, cell in enumerate(self.cells):
            t = (i + 1) / float(self.n_layers)
            h = cell(h, x_in, t)
            x_in = h
        output = torch.cat([h, timestep_feat], dim=1)  # (N, hidden_dim + 1)
        return output

class offLTC(nn.Module):
    def __init__(self, nhid, ode_unfolds=ode_unfolds):
        super().__init__()
        self.nhid = nhid
        self.ode_unfolds = ode_unfolds
        self.tau = nn.Parameter(torch.empty(nhid))
        nn.init.uniform_(self.tau, 0.1, 10)
        self.w_gate = nn.Sequential(
            nn.Linear(nhid + 1, nhid),
            nn.LayerNorm(nhid),
            nn.Sigmoid()
        )
        self.w_hid = nn.Sequential(
            nn.Linear(nhid + 1, nhid),
            nn.LayerNorm(nhid),
            nn.Tanh()
        )
        self.transform = nn.Sequential(
            nn.Linear(nhid, nhid),
            nn.LayerNorm(nhid)
        )
        self.A = nn.Parameter(torch.ones(nhid))
        self.out = nn.Linear(nhid, nhid)
        nn.init.xavier_normal_(self.transform[0].weight)
        nn.init.xavier_normal_(self.w_gate[0].weight)
        nn.init.xavier_normal_(self.w_hid[0].weight)

    def off_compute_dynamics(self, x, t):
        t_broadcast = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype) * t
        x_t = torch.cat([x, t_broadcast], dim=1)
        gate = self.w_gate(x_t)
        hidden = self.w_hid(x_t)
        dynamic = self.transform(gate * hidden)
        return dynamic

    def off_ode_step(self, x, f, delta_t):
        tau = torch.relu(self.tau) + 1e-8
        tau_inv = 1.0 / tau
        numerator = x + delta_t * f * self.A
        denominator = 1 + delta_t * (tau_inv + f)
        return numerator / denominator

    def forward(self, t, x, dt=0.01):
        delta_t = dt / self.ode_unfolds
        for _ in range(self.ode_unfolds):
            f = self.off_compute_dynamics(x, t)
            x = self.off_ode_step(x, f, delta_t)
            t += delta_t
        return self.out(x)


class LTCDense(nn.Module):
    def __init__(self, in_features, out_features, time_steps):
        super().__init__()
        self.ltc = offLTC(nhid=in_features, ode_unfolds=ode_unfolds)
        self.time_steps = time_steps
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        dt = 1.0 / self.time_steps
        for step in range(self.time_steps):
            t = step * dt
            x = self.ltc(t, x, dt)
        return self.fc(x)


class EnhancedDistanceLTC(nn.Module):
    def __init__(self, nhid, ode_unfolds=5):
        super().__init__()
        self.nhid = nhid
        self.ode_unfolds = ode_unfolds
        self.tau = nn.Parameter(torch.empty(nhid))
        nn.init.uniform_(self.tau, 0.5, 5)
        self.agg_conv = dgl.nn.GraphConv(nhid, nhid, norm='both', weight=True)
        
        # 距离特征专用编码器（简化版）
        self.distance_encoder = nn.Sequential(
            nn.Linear(1, nhid),
            nn.LayerNorm(nhid),
            nn.ReLU()
        )
        
        # 简化的融合层
        self.fusion_layer = nn.Linear(nhid * 2, nhid)
        self.transform = nn.Linear(nhid, nhid)
        self.A = nn.Parameter(torch.ones(nhid))
        self.out = nn.Linear(nhid, nhid)

        # 初始化
        nn.init.xavier_normal_(self.transform.weight)
        nn.init.xavier_normal_(self.agg_conv.weight)
        nn.init.xavier_normal_(self.fusion_layer.weight)

    def _extract_distance_features(self, features):
        distance_times = features[:, -1:]  # [N, 1]
        other_features = features[:, :-1]   # [N, D-1]
        return other_features, distance_times

    def _compute_enhanced_gating(self, graph, x, distance_times):
        # 邻域聚合
        x_agg = self.agg_conv(graph, x)
        
        # 距离编码（简化处理）
        distance_encoded = self.distance_encoder(distance_times)
        
        # 简单融合空间信息和邻域信息
        combined = torch.cat([x_agg, distance_encoded], dim=1)
        fused = self.fusion_layer(combined)
        
        gate = torch.sigmoid(fused)
        return gate

    def _ode_step(self, x, f, delta_t):
        tau = torch.relu(self.tau) + 1e-8
        tau_inv = 1.0 / tau
        numerator = x + delta_t * f * self.A
        denominator = 1 + delta_t * (tau_inv + f)
        denominator = denominator + 1e-12
        return numerator / denominator

    def forward(self, graph, features, t=0.0, dt=0.01):
        if features.shape[1] == self.nhid + 1:  # 包含距离特征 (hidden_dim + 1)
            # 提取距离时间特征
            x = features[:, :-1]  # [N, nhid]
            distance_times = features[:, -1:]  # [N, 1]
        else:
            x = features if features.shape[1] == self.nhid else features[:, :self.nhid]
            distance_times = torch.zeros(features.size(0), 1, device=features.device, dtype=features.dtype)
        if DEBUG_SHAPES:
            if not hasattr(self, "_shape_logged"):
                print(f"[EnhancedDistanceLTC] in={tuple(features.shape)}, has_time={features.shape[1]==self.nhid+1}, x={tuple(x.shape)}, tcol={tuple(distance_times.shape)}")
                self._shape_logged = True
        
        delta_t = dt / self.ode_unfolds
        for _ in range(self.ode_unfolds):
            gate = self._compute_enhanced_gating(graph, x, distance_times)
            f = self.transform(gate)
            x = self._ode_step(x, f, delta_t)
        return self.out(x)


class EnhancedLTCLayer(nn.Module):
    def __init__(self, nhid, time_steps=5, residual=True):
        super().__init__()
        self.enhanced_ltc = EnhancedDistanceLTC(nhid=nhid, ode_unfolds=ode_unfolds)
        self.graph_conv = dgl.nn.GraphConv(nhid, nhid, norm='both', weight=True)
        self.multi_scale_conv = nn.ModuleList([
            dgl.nn.GraphConv(nhid, nhid, norm='both', weight=True), 
            dgl.nn.GraphConv(nhid, nhid, norm='both', weight=True),
        ])
        self.scale_fusion = nn.Linear(nhid * 3, nhid) 
        self.bn = nn.LayerNorm(nhid)
        self.relu = nn.ReLU()
        self.residual = residual
        nn.init.xavier_normal_(self.graph_conv.weight)
        for conv in self.multi_scale_conv:
            nn.init.xavier_normal_(conv.weight)
        nn.init.xavier_normal_(self.scale_fusion.weight)
        
    def forward(self, graph, feat):
        ltc_feat = self.enhanced_ltc(graph, feat)
        residual = ltc_feat
        graph_feat = self.graph_conv(graph, ltc_feat)
        scale1_feat = self.multi_scale_conv[0](graph, ltc_feat) 
        scale2_feat = self.multi_scale_conv[1](graph, scale1_feat) 
        multi_scale = torch.cat([ltc_feat, scale1_feat, scale2_feat], dim=-1)
        fused_feat = self.scale_fusion(multi_scale)
        x = self.bn(graph_feat + fused_feat)
        x = self.relu(x)

        if self.residual:
            x = x + residual
            x = self.relu(x)
        return x


class LTC(nn.Module):
    def __init__(self, nhid, ode_unfolds=ode_unfolds):
        super().__init__()
        self.nhid = nhid
        self.ode_unfolds = ode_unfolds
        self.tau = nn.Parameter(torch.empty(nhid))
        nn.init.uniform_(self.tau, 0.5, 5)
        self.agg_conv = dgl.nn.GraphConv(nhid, nhid, norm='both', weight=True)
        self.time_linear = nn.Linear(nhid + 1, nhid)
        self.transform = nn.Linear(nhid, nhid)
        self.A = nn.Parameter(torch.ones(nhid))
        self.out = nn.Linear(nhid, nhid)
        nn.init.xavier_normal_(self.transform.weight)
        nn.init.xavier_normal_(self.agg_conv.weight)
        nn.init.xavier_normal_(self.time_linear.weight)

    def _compute_hybrid_gating(self, graph, x, t):
        x_agg = self.agg_conv(graph, x)
        t_broadcast = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype) * t
        x_time = torch.cat([x, t_broadcast], dim=1)
        x_time = torch.tanh(self.time_linear(x_time))
        combined = x_agg + x_time
        gate = torch.sigmoid(combined)
        return gate

    def _compute_dynamics(self, graph, x, t):
        gate = self._compute_hybrid_gating(graph, x, t)
        dynamic = self.transform(gate)
        return dynamic

    def _ode_step(self, x, f, delta_t):
        tau = torch.relu(self.tau) + 1e-8
        tau_inv = 1.0 / tau
        numerator = x + delta_t * f * self.A
        denominator = 1 + delta_t * (tau_inv + f)
        denominator = denominator + 1e-12  # 防止分母为0
        return numerator / denominator

    def forward(self, graph, x, t=0.0, dt=0.01):
        delta_t = dt / self.ode_unfolds
        for _ in range(self.ode_unfolds):
            f = self._compute_dynamics(graph, x, t)
            x = self._ode_step(x, f, delta_t)
            t += delta_t
        return self.out(x)


# ====== 消融实验替代组件 ======

class SimpleMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers=3, activation='relu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 支持不同激活函数
        if activation.lower() == 'relu':
            act_fn = nn.ReLU()
        elif activation.lower() == 'tanh':
            act_fn = nn.Tanh()
        elif activation.lower() == 'sigmoid':
            act_fn = nn.Sigmoid()
        elif activation.lower() == 'gelu':
            act_fn = nn.GELU()
        else:
            act_fn = nn.ReLU()
        
        layers = []
        current_dim = in_dim - 1  # 去除时间步特征
        
        for i in range(n_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                act_fn,
                nn.Dropout(0.1)
            ])
            current_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, feat):
        # 分离时间步特征并用MLP处理基础特征
        if feat.shape[1] > self.hidden_dim:
            base_feat = feat[:, :-1]
            timestep_feat = feat[:, -1:]
        else:
            base_feat = feat
            timestep_feat = torch.zeros(feat.size(0), 1, device=feat.device, dtype=feat.dtype)
        
        h = self.mlp(base_feat)
        # 重新附加时间步特征
        output = torch.cat([h, timestep_feat], dim=1)
        return output


class EnhancedCfC(nn.Module):
    """增强CfC预处理模块"""
    def __init__(self, in_dim, hidden_dim, n_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # 增强版：添加注意力机制和残差连接
        self.cells = nn.ModuleList([
            CfCCell((in_dim - 1) if i == 0 else hidden_dim, hidden_dim, is_first_layer=(i == 0))
            for i in range(n_layers)
        ])
        
        # 增强特性
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, feat):
        # 分离时间步特征
        if feat.shape[1] > self.hidden_dim:
            base_feat = feat[:, :-1]
            timestep_feat = feat[:, -1:]
        else:
            base_feat = feat
            timestep_feat = torch.zeros(feat.size(0), 1, device=feat.device, dtype=feat.dtype)
        
        # CfC处理
        h = torch.zeros(base_feat.size(0), self.hidden_dim, device=feat.device, dtype=feat.dtype)
        x_in = base_feat
        for i, cell in enumerate(self.cells):
            t = (i + 1) / float(self.n_layers)
            h_new = cell(h, x_in, t)
            
            # 添加注意力机制（增强特性）
            h_att, _ = self.attention(h_new.unsqueeze(1), h_new.unsqueeze(1), h_new.unsqueeze(1))
            h = self.layer_norm(h_new + h_att.squeeze(1))
            x_in = h
        
        # 重新附加时间步特征
        output = torch.cat([h, timestep_feat], dim=1)
        return output


class LinearProjection(nn.Module):
    """线性投影替代CfC"""
    def __init__(self, in_dim, hidden_dim, n_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Linear(in_dim - 1, hidden_dim)
        
    def forward(self, feat):
        # 分离时间步特征并用线性投影处理基础特征
        if feat.shape[1] > self.hidden_dim:
            base_feat = feat[:, :-1]
            timestep_feat = feat[:, -1:]
        else:
            base_feat = feat
            timestep_feat = torch.zeros(feat.size(0), 1, device=feat.device, dtype=feat.dtype)
        
        h = self.projection(base_feat)
        # 重新附加时间步特征
        output = torch.cat([h, timestep_feat], dim=1)
        return output


class SimpleLTCLayer(nn.Module):
    """简化的LTC层"""
    def __init__(self, nhid, time_steps=5, residual=True):
        super().__init__()
        self.ltc = LTC(nhid=nhid, ode_unfolds=ode_unfolds)
        self.residual = residual
        
    def forward(self, graph, feat):
        # 提取nhid维特征（去除时间步）
        if feat.shape[1] == nhid + 1:
            x = feat[:, :-1]
        else:
            x = feat if feat.shape[1] == nhid else feat[:, :nhid]
        
        residual = x
        x = self.ltc(graph, x)
        
        if self.residual:
            x = x + residual
            
        return x


class StandardLTCLayer(nn.Module):
    """标准LTC层（无增强特性）"""
    def __init__(self, nhid, time_steps=5, residual=True):
        super().__init__()
        self.ltc = LTC(nhid=nhid, ode_unfolds=ode_unfolds)
        self.residual = residual
        
    def forward(self, graph, feat):
        # 提取nhid维特征（去除时间步）
        if feat.shape[1] == nhid + 1:
            x = feat[:, :-1]
        else:
            x = feat if feat.shape[1] == nhid else feat[:, :nhid]
        
        residual = x
        x = self.ltc(graph, x)
        
        if self.residual:
            x = x + residual
            
        return x


class DenseConnectionLTCLayer(nn.Module):
    """LTC+密集连接层"""
    def __init__(self, nhid, time_steps=5, residual=True):
        super().__init__()
        self.enhanced_ltc = EnhancedDistanceLTC(nhid=nhid, ode_unfolds=ode_unfolds)
        self.graph_conv = dgl.nn.GraphConv(nhid, nhid, norm='both', weight=True)
        
        # 密集连接
        self.dense_layers = nn.ModuleList([
            nn.Linear(nhid, nhid) for _ in range(3)
        ])
        self.dense_fusion = nn.Linear(nhid * 4, nhid)  # LTC + GraphConv + 3个Dense
        
        self.bn = nn.LayerNorm(nhid)
        self.relu = nn.ReLU()
        self.residual = residual
        
    def forward(self, graph, feat):
        # 提取nhid维特征（去除时间步）
        if feat.shape[1] == nhid + 1:
            x = feat[:, :-1]
            distance_times = feat[:, -1:]
            feat_with_time = feat
        else:
            x = feat if feat.shape[1] == nhid else feat[:, :nhid]
            distance_times = torch.zeros(feat.size(0), 1, device=feat.device, dtype=feat.dtype)
            feat_with_time = torch.cat([x, distance_times], dim=1)
        
        residual = x
        
        # LTC处理
        ltc_out = self.enhanced_ltc(graph, feat_with_time)
        
        # 图卷积处理
        graph_out = self.graph_conv(graph, x)
        
        # 密集层处理
        dense_outs = []
        dense_input = x
        for dense_layer in self.dense_layers:
            dense_out = dense_layer(dense_input)
            dense_outs.append(dense_out)
            dense_input = dense_out
        
        # 密集连接融合
        all_features = torch.cat([ltc_out, graph_out] + dense_outs, dim=-1)
        fused = self.dense_fusion(all_features)
        
        x = self.bn(fused)
        x = self.relu(x)
        
        if self.residual:
            x = x + residual
            x = self.relu(x)
            
        return x


class NoLTCLayer(nn.Module):
    """移除LTC，仅使用图卷积 - 根据配置决定是否包含增强组件"""
    def __init__(self, nhid, time_steps=5, residual=True, use_multi_scale=False, use_structure_enhance=False):
        super().__init__()
        self.use_multi_scale = use_multi_scale
        self.use_structure_enhance = use_structure_enhance
        
        # 基础图卷积
        self.graph_conv = dgl.nn.GraphConv(nhid, nhid, norm='both', weight=True)
        
        # 多尺度卷积（可选）
        if use_multi_scale:
            self.multi_scale_conv = nn.ModuleList([
                dgl.nn.GraphConv(nhid, nhid, norm='both', weight=True),
                dgl.nn.GraphConv(nhid, nhid, norm='both', weight=True),
            ])
            self.scale_fusion = nn.Linear(nhid * 3, nhid)
        else:
            self.multi_scale_conv = None
            self.scale_fusion = None
        
        # 结构增强（可选）
        if use_structure_enhance:
            self.structure_enhance = dgl.nn.GraphConv(nhid, nhid, norm='both', weight=True)
        else:
            self.structure_enhance = None
            
        self.bn = nn.LayerNorm(nhid)
        self.relu = nn.ReLU()
        self.residual = residual
        
    def forward(self, graph, feat):
        # 提取nhid维特征（去除时间步）
        if feat.shape[1] == nhid + 1:
            x = feat[:, :-1]
        else:
            x = feat if feat.shape[1] == nhid else feat[:, :nhid]
        
        residual = x
        
        # 基础图卷积
        x = self.graph_conv(graph, x)
        
        # 多尺度融合（可选）
        if self.use_multi_scale and self.multi_scale_conv is not None:
            scale1_feat = self.multi_scale_conv[0](graph, residual)
            scale2_feat = self.multi_scale_conv[1](graph, scale1_feat)
            multi_scale = torch.cat([x, scale1_feat, scale2_feat], dim=-1)
            x = self.scale_fusion(multi_scale)
        
        # 结构增强（可选）
        if self.use_structure_enhance and self.structure_enhance is not None:
            x = x + self.structure_enhance(graph, x)
        
        x = self.bn(x)
        x = self.relu(x)
        
        if self.residual:
            x = x + residual
            x = self.relu(x)
            
        return x


class SimpleDense(nn.Module):
    """简化的Dense层"""
    def __init__(self, in_features, out_features, time_steps, activation='relu'):
        super().__init__()
        if activation.lower() == 'relu':
            act_fn = nn.ReLU()
        elif activation.lower() == 'tanh':
            act_fn = nn.Tanh()
        elif activation.lower() == 'sigmoid':
            act_fn = nn.Sigmoid()
        elif activation.lower() == 'gelu':
            act_fn = nn.GELU()
        else:
            act_fn = nn.ReLU()
            
        self.fc = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            act_fn
        )

    def forward(self, x):
        return self.fc(x)


# ====== 主要消融模型 ======

class AblationGCN(nn.Module):
    """基于9.26版本MyGCN的消融模型"""
    def __init__(self, in_dim, nhid=nhid, dropout=dropout, time_steps=time_steps, experiment_config=None):
        super(AblationGCN, self).__init__()
        self.nhid = nhid
        self.experiment_config = experiment_config or {}
        
        # ====== CfC预处理模块选择 ======
        if self.experiment_config.get('use_cfc', True):
            cfc_type = self.experiment_config.get('cfc_type', 'original')
            cfc_layers = self.experiment_config.get('cfc_layers', 3)
            
            if cfc_type == 'simple':
                cfc_activation = self.experiment_config.get('cfc_activation', 'relu')
                self.cfc_preprocess = SimpleMLP(in_dim=in_dim, hidden_dim=nhid, n_layers=cfc_layers, activation=cfc_activation)
            elif cfc_type == 'enhanced':
                self.cfc_preprocess = EnhancedCfC(in_dim=in_dim, hidden_dim=nhid, n_layers=cfc_layers)
            else:  # original
                self.cfc_preprocess = PreprocessCfC(in_dim=in_dim, hidden_dim=nhid, n_layers=cfc_layers)
        else:
            # 完全移除CfC预处理，直接使用原始特征
            self.cfc_preprocess = None
        
        # ====== LTC层选择 ======
        ltc_params = self.experiment_config.get('ltc_params', {})
        
        if not self.experiment_config.get('use_ltc', True):
            # 完全移除LTC，使用简单图卷积（根据配置决定增强组件）
            ltc_residual = self.experiment_config.get('ltc_residual', False)
            use_multi_scale = self.experiment_config.get('use_multi_scale', True)
            use_structure_enhance = self.experiment_config.get('use_structure_enhance', True)
            self.ltc_conv1 = NoLTCLayer(nhid=nhid, time_steps=time_steps, residual=ltc_residual, 
                                       use_multi_scale=use_multi_scale, use_structure_enhance=use_structure_enhance)
            self.ltc_conv2 = None  # E12等极端消融实验不需要第二层
        else:
            ltc_type = self.experiment_config.get('ltc_type', 'enhanced')
            ltc_layers = self.experiment_config.get('ltc_layers', 2)
            ltc_residual = self.experiment_config.get('ltc_residual', True)
            
            if ltc_type == 'simple':
                self.ltc_conv1 = SimpleLTCLayer(nhid=nhid, time_steps=time_steps, residual=ltc_residual)
                if ltc_layers > 1:
                    self.ltc_conv2 = SimpleLTCLayer(nhid=nhid, time_steps=time_steps, residual=ltc_residual)
                else:
                    self.ltc_conv2 = None
            elif ltc_type == 'standard':
                self.ltc_conv1 = StandardLTCLayer(nhid=nhid, time_steps=time_steps, residual=ltc_residual)
                if ltc_layers > 1:
                    self.ltc_conv2 = StandardLTCLayer(nhid=nhid, time_steps=time_steps, residual=ltc_residual)
                else:
                    self.ltc_conv2 = None
            elif ltc_type == 'dense_connection':
                self.ltc_conv1 = DenseConnectionLTCLayer(nhid=nhid, time_steps=time_steps, residual=ltc_residual)
                if ltc_layers > 1:
                    self.ltc_conv2 = DenseConnectionLTCLayer(nhid=nhid, time_steps=time_steps, residual=ltc_residual)
                else:
                    self.ltc_conv2 = None
            else:  # enhanced
                self.ltc_conv1 = EnhancedLTCLayer(nhid=nhid, time_steps=time_steps, residual=ltc_residual)
                if ltc_layers > 1:
                    self.ltc_conv2 = EnhancedLTCLayer(nhid=nhid, time_steps=time_steps, residual=ltc_residual)
                else:
                    self.ltc_conv2 = None
        
        # ====== 图结构增强模块 ======
        # 注意：当use_ltc=False时，结构增强已集成到NoLTCLayer中，避免重复
        if self.experiment_config.get('use_ltc', True) and self.experiment_config.get('use_structure_enhance', True):
            self.structure_enhance = nn.Sequential(
                dgl.nn.GraphConv(nhid, nhid, norm='both', weight=True),
                nn.LayerNorm(nhid),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)
            )
        else:
            self.structure_enhance = None
        
        # ====== 池化策略选择 ======
        pool_type = self.experiment_config.get('pool_type', 'set2set')
        if pool_type == 'set2set':
            self.pool = dgl.nn.Set2Set(nhid, n_iters=3, n_layers=1)
            pool_output_dim = 2 * nhid
        elif pool_type == 'mean':
            self.pool = dgl.nn.AvgPooling()
            pool_output_dim = nhid
        elif pool_type == 'max':
            self.pool = dgl.nn.MaxPooling()
            pool_output_dim = nhid
        elif pool_type == 'global':
            # 全局池化：同时使用mean和max
            self.pool_mean = dgl.nn.AvgPooling()
            self.pool_max = dgl.nn.MaxPooling()
            pool_output_dim = nhid * 2
        else:  # none 或其他
            self.pool = dgl.nn.AvgPooling()  # 默认使用平均池化
            pool_output_dim = nhid
        
        self.projection = nn.Linear(pool_output_dim, nhid)
        
        # ====== 融合策略选择 ======
        fusion_type = self.experiment_config.get('fusion_type', 'concat_diff')
        use_fusion = self.experiment_config.get('use_fusion', True)
        
        if not use_fusion:
            fusion_dim = nhid  # 仅使用单一表示
        elif fusion_type == 'symmetric':
            # 对称融合：sum + prod + abs-diff（保证 f(A,B)=f(B,A)）
            fusion_dim = 3 * nhid
        elif fusion_type == 'concat_only':
            fusion_dim = 2 * nhid  # 仅concat
        elif fusion_type == 'diff_only':
            fusion_dim = nhid  # 仅diff
        elif fusion_type == 'hadamard':
            fusion_dim = 2 * nhid  # concat + hadamard product
        elif fusion_type == 'cosine':
            fusion_dim = 2 * nhid + 1  # concat + cosine similarity
        else:  # concat_diff
            fusion_dim = 3 * nhid  # concat + diff
        
        # ====== Dense层配置 ======
        use_ltc_dense = self.experiment_config.get('use_ltc_dense', True)
        fc_layers = self.experiment_config.get('fc_layers', 2)
        fc_activation = self.experiment_config.get('fc_activation', 'relu')
        
        if use_ltc_dense:
            self.fc1 = LTCDense(fusion_dim, nhidh, time_steps)
            if fc_layers > 1:
                if fc_layers == 2:
                    self.fc2 = LTCDense(nhidh, nhidhh, time_steps)
                else:  # fc_layers > 2，支持更多层
                    self.fc_layers = nn.ModuleList([
                        LTCDense(nhidh if i == 0 else nhidhh, nhidhh, time_steps)
                        for i in range(fc_layers - 1)
                    ])
                    self.fc2 = None
            else:
                self.fc2 = None
                self.fc_layers = None
        else:
            self.fc1 = SimpleDense(fusion_dim, nhidh, time_steps, activation=fc_activation)
            if fc_layers > 1:
                if fc_layers == 2:
                    self.fc2 = SimpleDense(nhidh, nhidhh, time_steps, activation=fc_activation)
                else:  # fc_layers > 2
                    self.fc_layers = nn.ModuleList([
                        SimpleDense(nhidh if i == 0 else nhidhh, nhidhh, time_steps, activation=fc_activation)
                        for i in range(fc_layers - 1)
                    ])
                    self.fc2 = None
            else:
                self.fc2 = None
                self.fc_layers = None
        
        # 最终分类层
        final_input_dim = nhidhh if fc_layers > 1 else nhidh
        self.fc3 = nn.Linear(final_input_dim, 2)
        self.dropout = nn.Dropout(dropout)

        # 权重初始化
        if self.structure_enhance is not None:
            nn.init.xavier_normal_(self.structure_enhance[0].weight)
        nn.init.xavier_normal_(self.projection.weight)
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, g1, g2, fea1=None, fea2=None):
        # 9.26版本兼容性：支持新旧调用方式
        if fea1 is None and fea2 is None:
            # 从图节点获取特征
            fea1 = g1.ndata['fea']
            fea2 = g2.ndata['fea']
        
        # 🧬 步骤1: CfC预处理 - 保持残基距离时间步特征
        if self.cfc_preprocess is not None:
            fea1 = checkpoint(self.cfc_preprocess, fea1, use_reentrant=False)
            fea2 = checkpoint(self.cfc_preprocess, fea2, use_reentrant=False)
        
        # 🚀 步骤2: 第一层LTC处理
        fea1 = self.ltc_conv1(g1, fea1)
        fea1 = self.dropout(fea1)

        fea2 = self.ltc_conv1(g2, fea2)
        fea2 = self.dropout(fea2)

        # 🚀 步骤3: 第二层LTC处理（如果存在）
        if self.ltc_conv2 is not None:
            fea1 = self.ltc_conv2(g1, fea1)
            fea2 = self.ltc_conv2(g2, fea2)
        
        # 🆕 步骤4: 额外的图结构增强（如果启用）
        if self.structure_enhance is not None:
            enhanced_fea1 = self.structure_enhance[0](g1, fea1)  # GraphConv
            enhanced_fea1 = self.structure_enhance[1](enhanced_fea1)  # LayerNorm
            enhanced_fea1 = self.structure_enhance[2](enhanced_fea1)  # ReLU
            enhanced_fea1 = self.structure_enhance[3](enhanced_fea1)  # Dropout
            
            enhanced_fea2 = self.structure_enhance[0](g2, fea2)  # GraphConv
            enhanced_fea2 = self.structure_enhance[1](enhanced_fea2)  # LayerNorm
            enhanced_fea2 = self.structure_enhance[2](enhanced_fea2)  # ReLU
            enhanced_fea2 = self.structure_enhance[3](enhanced_fea2)  # Dropout
            
            # 残差连接
            fea1 = fea1 + enhanced_fea1
            fea2 = fea2 + enhanced_fea2
        
        fea1 = self.dropout(fea1)
        fea2 = self.dropout(fea2)

        # 🔄 步骤5: 图池化处理
        g1.ndata['h'] = fea1
        g2.ndata['h'] = fea2

        pool_type = self.experiment_config.get('pool_type', 'set2set')
        
        if pool_type == 'global':
            # 全局池化：组合mean和max
            hg1_mean = self.pool_mean(g1, g1.ndata['h'])
            hg1_max = self.pool_max(g1, g1.ndata['h'])
            hg1 = torch.cat([hg1_mean, hg1_max], dim=-1)
            
            hg2_mean = self.pool_mean(g2, g2.ndata['h'])
            hg2_max = self.pool_max(g2, g2.ndata['h'])
            hg2 = torch.cat([hg2_mean, hg2_max], dim=-1)
        elif pool_type == 'attention':
            hg1 = self.pool(g1, g1.ndata['h'])
            hg2 = self.pool(g2, g2.ndata['h'])
        elif isinstance(self.pool, dgl.nn.Set2Set):
            hg1 = self.pool(g1, g1.ndata['h'])
            hg2 = self.pool(g2, g2.ndata['h'])
        else:
            hg1 = self.pool(g1, g1.ndata['h'])
            hg2 = self.pool(g2, g2.ndata['h'])

        # 清理图节点数据以释放内存
        del g1.ndata['h']
        del g2.ndata['h']

        # 🎯 步骤6: 图表示投影和融合
        hg1 = self.projection(hg1)
        hg2 = self.projection(hg2)

        # 根据融合策略组合特征
        fusion_type = self.experiment_config.get('fusion_type', 'concat_diff')
        use_fusion = self.experiment_config.get('use_fusion', True)
        
        if not use_fusion:
            # 不使用融合，仅使用第一个图的表示
            hg = hg1
        elif fusion_type == 'symmetric':
            # 对称融合：sum + prod + abs-diff（与主模型对齐）
            hg_sum = hg1 + hg2
            hg_prod = hg1 * hg2
            hg_diff = torch.abs(hg1 - hg2)
            hg = torch.cat([hg_sum, hg_prod, hg_diff], dim=-1)
        elif fusion_type == 'concat_only':
            hg = torch.cat([hg1, hg2], dim=-1)
        elif fusion_type == 'diff_only':
            hg = torch.abs(hg1 - hg2)
        elif fusion_type == 'hadamard':
            hg_concat = torch.cat([hg1, hg2], dim=-1)
            hg_hadamard = hg1 * hg2
            hg = torch.cat([hg_concat, hg_hadamard], dim=-1)
        elif fusion_type == 'cosine':
            hg_concat = torch.cat([hg1, hg2], dim=-1)
            # 计算余弦相似度
            cosine_sim = F.cosine_similarity(hg1, hg2, dim=-1, eps=1e-8).unsqueeze(-1)
            hg = torch.cat([hg_concat, cosine_sim], dim=-1)
        else:  # concat_diff
            hg_concat = torch.cat([hg1, hg2], dim=-1)
            hg_diff = torch.abs(hg1 - hg2)
            hg = torch.cat([hg_concat, hg_diff], dim=-1)

        # 清理中间变量
        del hg1, hg2

        # 🔚 步骤7: 全连接层预测
        h = F.relu(self.fc1(hg))
        h = self.dropout(h)
        
        fc_layers = self.experiment_config.get('fc_layers', 2)
        
        if hasattr(self, 'fc_layers') and self.fc_layers is not None:
            # 多层Dense（超过2层）
            for fc_layer in self.fc_layers:
                h = F.relu(fc_layer(h))
                h = self.dropout(h)
        elif self.fc2 is not None:
            # 标准2层Dense
            h = F.relu(self.fc2(h))
        
        return self.fc3(h)


# 别名定义，保持兼容性
AblationPPIModel = AblationGCN

if __name__ == "__main__":
    # 测试消融模型
    print("测试消融模型...")
    
    # 测试配置
    config = {
        'use_cfc': True,
        'cfc_type': 'original',
        'use_ltc': True,
        'ltc_type': 'enhanced',
        'ltc_layers': 2,
        'use_ltc_dense': True,
        'fc_layers': 2,
        'pool_type': 'set2set',
        'fusion_type': 'concat_diff'
    }
    
    model = AblationGCN(in_dim=1804, experiment_config=config)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("✅ 消融模型测试通过")