import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

class DenseLayer(nn.Module):

    def __init__(self, 
                 c_in, 
                 c_out, 
                 zero_init=False, 
                 ):

        super().__init__()

        self.linear = nn.Linear(c_in, c_out)

        # Initialization
        if zero_init:
            nn.init.zeros_(self.linear.weight.data)
        else:
            nn.init.uniform_(self.linear.weight.data, -np.sqrt(
                6 / (c_in + c_out)), np.sqrt(6 / (c_in + c_out)))
        nn.init.zeros_(self.linear.bias.data)

    def forward(self, 
    			node_feats, # input node features
    			):

        node_feats = self.linear(node_feats)

        return node_feats

class SineLayer(nn.Module):
    def __init__(self, c_in, c_out, bias=True, zero_init=False,
                 omega_0=1):
        super().__init__()
        self.omega_0 = omega_0
        self.zero_init = zero_init
        self.in_features = c_in
        self.linear = nn.Linear(c_in, c_out, bias=bias)
        
        if self.zero_init:
            nn.init.zeros_(self.linear.weight.data)
        else:
            nn.init.uniform_(self.linear.weight.data, -np.sqrt(
                6 / (c_in + c_out)), np.sqrt(6 / (c_in + c_out)))
            nn.init.zeros_(self.linear.bias.data)
            
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class DeconvNet(nn.Module):

    def __init__(self, 
                 hidden_dims,
                 n_celltypes, 
                 n_slices,
                 n_heads, 
                 slice_emb_dim, 
                 adj_dim,
                 coef_fe,
                 ):
        import torch
        import random
        seed=1
        torch.manual_seed(seed) 
        torch.cuda.manual_seed(seed) 
        torch.cuda.manual_seed_all(seed)  
        np.random.seed(seed)  
        random.seed(seed)  
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        super().__init__()
        self.training_steps = 14001 # 这里修改了，原来为 14001。
        mid_channel = 200
        
        # self.encoder_layer0 = nn.Sequential(SineLayer(3, mid_channel),
        # ========== 修改开始：调整encoder_layer0输入维度以适应拼接后的坐标 ==========
        # 原始坐标维度为3，位置编码维度也为3，拼接后维度为6
        # 因此encoder_layer0的第一层输入维度从3改为6
        self.encoder_layer0 = nn.Sequential(SineLayer(6, mid_channel),
                                            SineLayer(mid_channel, mid_channel),
                                            SineLayer(mid_channel, 30),
                                            DenseLayer(30, hidden_dims[0]))
        # ========== 修改结束 ==========
                    
        self.encoder_layer1 = DenseLayer(hidden_dims[0],hidden_dims[2])
        
        self.decoder = nn.Sequential(SineLayer(hidden_dims[2], mid_channel),
                                            DenseLayer(mid_channel, hidden_dims[0]))
        
        self.deconv_alpha_layer = DenseLayer(hidden_dims[2] + slice_emb_dim, 
                                             1, zero_init=True)
        
        self.deconv_beta_layer = nn.Sequential(DenseLayer(hidden_dims[2], n_celltypes, 
                                                          zero_init=True))

        self.gamma = nn.Parameter(torch.Tensor(n_slices, 
                                               hidden_dims[0]).zero_())

        self.slice_emb = nn.Embedding(n_slices, slice_emb_dim) # n_slice, 16

        self.coef_fe = coef_fe

        # ========== 修改开始：添加位置编码相关参数 ==========
        self.use_positional_encoding = True  # 是否使用位置编码（已启用）
        self.positional_encoding_scale = 10000.0  # 位置编码缩放因子
        # ========== 修改结束 ==========

    def forward(self, 
                coord, # n1*3 coordinate matrix (拼接后变为 n1*6)
                adj_matrix,
                node_feats, # input node features n1*n2
                count_matrix, # gene expression counts
                library_size, # library size (based on Y)
                slice_label, # slice label
                basis, # basis matrix
                step
                ):
        # ========== 修改说明：坐标维度变化 ==========
        # 原始输入coord维度: (n_spots, 3)
        # 启用位置编码后，拼接维度: (n_spots, 6) = 原始坐标(3) + 位置编码(3)
        # encoder_layer0的第一层输入维度已从3调整为6
        # ========== 修改说明结束 ==========
        
        # encoder
        
        self.node_feats = node_feats
        
        # self.coord = coord/100 # Scale the coordinates

        # ========== 修改开始：坐标缩放与位置编码（拼接方式） ==========
        self.coord = coord / 100  # Scale the coordinates

        # 添加位置编码
        if self.use_positional_encoding:
            # 生成位置编码矩阵
            pe_matrix = self.generate_positional_encoding(self.coord)
            
            # 验证维度匹配
            if pe_matrix.shape != self.coord.shape:
                raise ValueError(
                    f"位置编码矩阵维度 {pe_matrix.shape} 与坐标矩阵维度 "
                    f"{self.coord.shape} 不匹配！\n"
                    f"坐标矩阵形状: (n_spots={self.coord.shape[0]}, d_model={self.coord.shape[1]})\n"
                    f"位置编码矩阵形状: (n_spots={pe_matrix.shape[0]}, d_model={pe_matrix.shape[1]})"
                )
            
            # self.coord = self.coord + pe_matrix

            # ========== 修改：从叠加改为拼接 ==========
            # 原方式：元素级加法，维度不变 [1,2] + [2,3] = [3,5]
            # 新方式：拼接操作，维度翻倍 [1,2] + [2,3] = [1,2,2,3]
            # 拼接后维度从 (n_spots, 3) 变为 (n_spots, 6)
            self.coord = torch.cat([self.coord, pe_matrix], dim=1)
            # ========== 修改结束 ==========
            
            # 调试信息（可选，测试通过后可移除）
            # print(f"[DEBUG] 原始坐标矩阵形状: {coord.shape}")
            # print(f"[DEBUG] 缩放后坐标矩阵形状: {(coord/100).shape}")
            # print(f"[DEBUG] 位置编码矩阵形状: {pe_matrix.shape}")
            # print(f"[DEBUG] 拼接后坐标矩阵形状: {self.coord.shape}")
            # print(f"[DEBUG] 坐标矩阵范围: [{self.coord.min():.4f}, {self.coord.max():.4f}]")
            # print(f"[DEBUG] 位置编码范围: [{pe_matrix.min():.4f}, {pe_matrix.max():.4f}]")
        # ========== 修改结束 ==========
        
        Z, mid_fea = self.encoder(node_feats)

        # deconvolutioner
        slice_label_emb = self.slice_emb(slice_label)
        
        beta, alpha = self.deconvolutioner(Z, slice_label_emb)
        
        self.node_feats_recon = self.decoder(Z)

        # deconvolution loss
        log_lam = torch.log(torch.matmul(beta, basis) + 1e-6) + alpha + self.gamma[slice_label]
        lam = torch.exp(log_lam)
        self.decon_loss = - 5*torch.mean(torch.sum(count_matrix * 
                                        (torch.log(library_size + 1e-6) + log_lam
                                         ) - library_size * lam, axis=1))
        
        self.fea_loss = 1*torch.norm(node_feats-mid_fea, 2)+2*torch.norm(node_feats-self.node_feats_recon, 2)
        
        # Total loss
        loss = 1*(self.decon_loss + self.fea_loss)  
        
        denoise = torch.matmul(beta, basis)
        return loss, mid_fea, denoise, Z, 0, 0

    def evaluate(self, adj_matrix, coord, node_feats, slice_label):
        slice_label_emb = self.slice_emb(slice_label)
        Z, _ = self.encoder(node_feats)
        beta, alpha = self.deconvolutioner(Z, slice_label_emb)
        
        return Z, beta, alpha, self.gamma
            
    def encoder(self, H):
        self.mid_fea = self.encoder_layer0(self.coord)
        Z = self.encoder_layer1(self.mid_fea)
        return Z, self.mid_fea
    
    def deconvolutioner(self, Z, slice_label_emb):
        beta = self.deconv_beta_layer(torch.sin(Z))
        beta = F.softmax(beta, dim=1)
        H = torch.sin(torch.cat((Z, slice_label_emb), axis=1))
        alpha = self.deconv_alpha_layer(H)
        return beta, alpha

    # ===添加===
    def generate_positional_encoding(self, coord_matrix):
        """
        生成位置编码矩阵
        
        参数:
            coord_matrix: 坐标矩阵，形状 (n_spots, d_model)
        
        返回:
            pe_matrix: 位置编码矩阵，形状 (n_spots, d_model)
        
        位置编码公式（Transformer风格）:
            PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
            PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
        其中:
            pos: 位置索引（对应坐标矩阵的行索引）
            d_model: 模型维度（对应坐标矩阵的列数）
            i: 维度索引
        """
        n_spots, d_model = coord_matrix.shape
        
        # 生成位置索引 (0, 1, 2, ..., n_spots-1)
        pos_indices = torch.arange(n_spots, dtype=torch.float32, device=coord_matrix.device)
        pos_indices = pos_indices.unsqueeze(1)  # (n_spots, 1)
        
        # 生成维度索引 (0, 1, 2, ..., d_model-1)
        dim_indices = torch.arange(d_model, dtype=torch.float32, device=coord_matrix.device)
        dim_indices = dim_indices.unsqueeze(0)  # (1, d_model)
        
        # 计算分母项: 10000^(2i/d_model)
        div_term = torch.pow(self.positional_encoding_scale, 
                        (2.0 * dim_indices) / d_model)  # (1, d_model)
        
        # 计算位置编码
        # pos / div_term: (n_spots, d_model)
        pos_div = pos_indices / div_term
        
        # 分离奇偶维度
        pe_matrix = torch.zeros_like(coord_matrix)
        
        # 偶数维度使用sin: 2i
        even_indices = (dim_indices % 2 == 0).squeeze(0)  # (d_model,)
        pe_matrix[:, even_indices] = torch.sin(pos_div[:, even_indices])
        
        # 奇数维度使用cos: 2i+1
        odd_indices = (dim_indices % 2 == 1).squeeze(0)  # (d_model,)
        pe_matrix[:, odd_indices] = torch.cos(pos_div[:, odd_indices])
        
        return pe_matrix
    # ===添加结束===