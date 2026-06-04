import copy
import torch.nn.functional as F
import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import reset, uniform
from torch_scatter import scatter_add
import einops
import math

from torch_geometric.nn import EdgeConv as PyGEdgeConv

class EdgeConv_Local(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()
        self.in_proj = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, bias=bias)
        self.w_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=(1, kernel_size),
                                stride=1, padding=(0, kernel_size // 2), groups=mid_channels)
        self.h_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=(kernel_size, 1),
                                stride=1, padding=(kernel_size // 2, 0), groups=mid_channels)
        self.out_proj = nn.Conv2d(in_channels=mid_channels * 2, out_channels=out_channels,
                                  kernel_size=1, bias=True)

    def forward(self, x):
        x = self.in_proj(x)
        x_w = self.w_conv(x)
        x_h = self.h_conv(x)
        x = torch.cat([x_w, x_h], dim=1)
        x = self.out_proj(x)
        return x



class HoGEdgeGateConv(nn.Module):
    def __init__(self, in_dim, nbins, cell_size=(8, 8)):
        super().__init__()
        self.nbins = nbins
        self.cell_size = cell_size

        self.hog_feat = nn.Sequential(
            nn.Conv2d(nbins, in_dim, kernel_size=1),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim, bias=False),
            nn.GroupNorm(in_dim // 8, in_dim),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.weight = nn.Sequential(
            EdgeConv_Local(in_channels=in_dim, mid_channels=in_dim // 2, out_channels=in_dim),
            nn.GroupNorm(in_dim // 8, in_dim)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, stride=1),
            nn.GroupNorm(in_dim // 8, in_dim)
        )

        self.fuse_block = nn.Sequential(
            EdgeConv_Local(in_channels=in_dim, mid_channels=in_dim // 2, out_channels=in_dim, kernel_size=3),
            nn.GroupNorm(in_dim // 8, in_dim)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        x = image2patches(x)
        x_hog = self.get_hog_feature(x)
        x_hog = self.hog_feat(x_hog)

        x1 = self.sigmoid(self.weight(x + x_hog))
        x2 = self.conv(x)
        x = x1 * x2

        x = patches2image(x)
        x = x + residual
        x = self.fuse_block(x)
        return x

    def get_hog_feature(self, x):
        x_mean = x.mean(dim=1, keepdim=True)
        B, _, H, W = x_mean.shape
        device = x_mean.device

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)

        dx = F.conv2d(x_mean.float(), sobel_x, padding=1)
        dy = F.conv2d(x_mean.float(), sobel_y, padding=1)

        gradient_dir = torch.atan2(dy, dx)
        gradient_dir = torch.abs(gradient_dir)

        cell_h, cell_w = self.cell_size
        H_cells = int(H / cell_h)
        W_cells = int(W / cell_w)

        dirs_crop = gradient_dir[:, :, :H_cells * cell_h, :W_cells * cell_w]
        dirs = einops.rearrange(dirs_crop, 'b c (h c_h) (w c_w) -> b h w (c c_h c_w)', c_h=cell_h, c_w=cell_w)

        bin_with = torch.pi / self.nbins
        bin_indices = (dirs / bin_with).floor().long()
        bin_indices = torch.clamp(bin_indices, 0, self.nbins - 1)

        weight = F.one_hot(bin_indices, num_classes=self.nbins).sum(dim=-2).float() / (cell_h * cell_w)

        start = torch.pi / (2 * self.nbins)
        hog_feature = torch.linspace(
            start, torch.pi - start, self.nbins
        ).to(device).view(1, 1, 1, self.nbins) * weight

        return hog_feature.permute(0, 3, 1, 2)


def image2patches(x):
    x = einops.rearrange(x, 'b c (hg h) (wg w) -> (hg wg b) c h w', hg=2, wg=2)
    return x


def patches2image(x):
    x = einops.rearrange(x, '(hg wg b) c h w -> b c (hg h) (wg w)', hg=2, wg=2)
    return x


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def full_block(in_features, out_features, p_drop, act=nn.ELU()):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
        act,
        nn.Dropout(p=p_drop),
    )


import math
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2, act=F.relu, bn=True):
        super(GraphConv, self).__init__()
        bn = nn.BatchNorm1d if bn else nn.Identity
        self.in_features = in_features
        self.out_features = out_features
        self.bn = bn(out_features)
        self.act = act
        self.dropout = dropout
        self.conv = GCNConv(in_channels=self.in_features, out_channels=self.out_features, cached=True)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = self.act(x)
        x = F.dropout(x, self.dropout, self.training)
        return x


class Direction_Aware_Gated_Encoder(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.input_dim = input_dim
        self.feat_hidden1 = config['feat_hidden1']
        self.feat_hidden2 = config['feat_hidden2']
        self.gcn_hidden = config['gcn_hidden']
        self.latent_dim = config['latent_dim']
        self.p_drop = config['p_drop']


        self.encoder_L1 = full_block(self.input_dim, self.feat_hidden1, self.p_drop)
        self.hog_L1 = HoGEdgeGateConv(in_dim=self.feat_hidden1, nbins=9, cell_size=(8, 8))

        self.encoder_L2 = full_block(self.feat_hidden1, self.feat_hidden2, self.p_drop)
        self.hog_L2 = HoGEdgeGateConv(in_dim=self.feat_hidden2, nbins=9, cell_size=(8, 8))

        self.gc1 = GraphConv(self.feat_hidden2, self.gcn_hidden, dropout=self.p_drop, act=F.relu)
        self.gc2 = GraphConv(self.gcn_hidden, self.latent_dim, dropout=self.p_drop, act=lambda x: x)

    def n_c_to_image(self, x):
        N, C = x.shape
        side = math.ceil(math.sqrt(N))
        side = ((side + 7) // 8) * 8
        pad_size = side * side - N

        if pad_size > 0:
            x_padded = F.pad(x, (0, 0, 0, pad_size), mode='constant', value=0)
        else:
            x_padded = x

        x_image = x_padded.view(1, side, side, C).permute(0, 3, 1, 2)
        return x_image, N

    def image_to_n_c(self, x_image, original_N):
        x_flat = x_image.permute(0, 2, 3, 1).reshape(-1, x_image.shape[1])
        return x_flat[:original_N, :]

    def forward(self, x, edge_index):
        h1 = self.encoder_L1(x)
        h1_image, original_N1 = self.n_c_to_image(h1)
        h1_image_out = self.hog_L1(h1_image)
        h1 = h1 + self.image_to_n_c(h1_image_out, original_N1)

        h2 = self.encoder_L2(h1)
        h2_image, original_N2 = self.n_c_to_image(h2)
        h2_image_out = self.hog_L2(h2_image)
        h2 = h2 + self.image_to_n_c(h2_image_out, original_N2)

        x_gcn = self.gc1(h2, edge_index.clone())
        x_gcn = self.gc2(x_gcn, edge_index.clone())

        return x_gcn
class Decoder(nn.Module):
    def __init__(self, output_dim, config, imputation=True):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = config['latent_dim']
        self.p_drop = config['p_drop']
        self.imputation = imputation
        if self.imputation:
            self.layer1 = nn.Linear(self.input_dim, self.output_dim)
        else:
            self.layer1 = GraphConv(self.input_dim, self.output_dim, dropout=self.p_drop, act=nn.Identity())

    def forward(self, x, edge_index):
        if self.imputation:
            return self.layer1(x)
        return self.layer1(x, edge_index.clone())

class SpaDAR_model(nn.Module):
    def __init__(self, input_dim, config, imputation=True, efficient_dgi=False):
        super().__init__()
        self.efficient_dgi = efficient_dgi 
        self.imputation = imputation
        self.dec_in_dim = config['latent_dim']
        self.online_encoder = Direction_Aware_Gated_Encoder(input_dim, config)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self._init_target()

        self.encoder_to_decoder = nn.Linear(self.dec_in_dim, config['project_dim'], bias=False)
        nn.init.xavier_uniform_(self.encoder_to_decoder.weight)
        self.projector = GraphConv(config['project_dim'], self.dec_in_dim, dropout=config['p_drop'], act=lambda x: x)

        self.decoder = Decoder(input_dim, config, self.imputation)
        self.enc_mask_token = nn.Parameter(torch.zeros(1, input_dim))
        self.rep_mask = nn.Parameter(torch.zeros(1, self.dec_in_dim))
        self.mask_rate = config['mask_rate']
        self.t = config['t']
        self.momentum_rate = config['momentum_rate']
        self.replace_rate = 0.05
        self.mask_token_rate = 1 - self.replace_rate
        self.anchor_pair = None

        self.weight = nn.Parameter(torch.empty(self.dec_in_dim, self.dec_in_dim))
        uniform(self.dec_in_dim, self.weight)

    def _init_target(self):
        for param_teacher in self.target_encoder.parameters():
            param_teacher.detach()
            param_teacher.requires_grad = False

    def momentum_update(self):
        base_momentum = self.momentum_rate
        for param_encoder, param_teacher in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_teacher.data = param_teacher.data * base_momentum + param_encoder.data * (1. - base_momentum)

    def encoding_mask_noise(self, x, edge_index, mask_rate=0.3):
        num_nodes = x.shape[0]
        self.num_nodes = num_nodes
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        out_x = x.clone()

        if self.replace_rate > 0:
            num_noise_nodes = int(self.replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self.mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self.replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
            out_x[token_nodes] = out_x[token_nodes] + self.enc_mask_token
        else:
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0
            out_x[token_nodes] = out_x[token_nodes] + self.enc_mask_token

        use_edge_index = edge_index.clone()
        return out_x, use_edge_index, (mask_nodes, keep_nodes)

    def generate_neg_nodes(self, mask_nodes):
        num_mask_nodes = mask_nodes.size(0)
        neg_nodes_x = torch.randint(0, self.num_nodes, (num_mask_nodes,), device=mask_nodes.device)
        neg_nodes_y = torch.randint(0, self.num_nodes, (num_mask_nodes,), device=mask_nodes.device)
        return neg_nodes_x, neg_nodes_y

    def mask_attr_prediction(self, x, edge_index, anchor_pair):
        use_x, use_adj, (mask_nodes, keep_nodes) = self.encoding_mask_noise(x, edge_index, self.mask_rate)
        enc_rep = self.online_encoder(use_x, use_adj)

        with torch.no_grad():
            x_t = x.clone()
            x_t[keep_nodes] = 0.0
            x_t[keep_nodes] += self.enc_mask_token
            rep_t = self.target_encoder(x_t, use_adj)

        if anchor_pair is not None:
            anchor, positive, negative = anchor_pair
            summary = self.avg_readout(enc_rep, [anchor, positive])
            num_mask_nodes = mask_nodes.size(0)
            neg_nodes = torch.randint(0, self.num_nodes, (num_mask_nodes,), device=mask_nodes.device)
            cl_loss = self.dgi_loss(enc_rep[mask_nodes], enc_rep[neg_nodes], summary[mask_nodes])

            # cl_loss = self.triplet_loss(enc_rep, anchor, positive, negative)
        else:
            cl_loss = 0

        rep = enc_rep
        rep = self.encoder_to_decoder(rep)
        rep[mask_nodes] = 0.0
        # rep[mask_nodes] += self.rep_mask
        rep = self.projector(rep, use_adj)
        #
        match_loss = self.match_loss(rep, rep_t, mask_nodes)
        # pos_match = rep[mask_nodes] * rep_t[mask_nodes]
        # neg_match = rep[neg_nodes_x] * rep_t[neg_nodes_y]
        # pos_match_out = self.MccrProjector(pos_match)
        # neg_match_out = self.MccrProjector(neg_match)
        # match_loss = (self.bce_loss(pos_match_out, torch.ones_like(pos_match_out))
        #               + self.bce_loss(neg_match_out, torch.zeros_like(neg_match_out)))

        # rep[mask_nodes] = 0.0
        recon = self.decoder(rep, use_adj)
        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        # online = rep[mask_nodes]
        # target = rep_t[mask_nodes]
        # match_loss = F.mse_loss(online, target)
        rec_loss = self.sce_loss(x_rec, x_init, t=self.t)

        return match_loss, rec_loss, cl_loss



    def sce_loss(self, x, y, t=2):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        cos_m = (1 + (x * y).sum(dim=-1)) * 0.5
        loss = -torch.log(cos_m.pow_(t))
        return loss.mean()

    def triplet_loss(self, emb, anchor, positive, negative, margin=1.0):
        anchor_arr = emb[anchor]
        positive_arr = emb[positive]
        negative_arr = emb[negative]
        triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
        tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)
        return tri_output

    def forward(self, x, edge_index, anchor_pair):
        safe_edge_index = edge_index.clone() if edge_index is not None else None
        if anchor_pair is not None:
            safe_anchor_pair = (anchor_pair[0].clone(), anchor_pair[1].clone(), anchor_pair[2].clone())
        else:
            safe_anchor_pair = None

        return self.mask_attr_prediction(x, safe_edge_index, safe_anchor_pair)

    @torch.no_grad()
    def evaluate(self, x, edge_index):
        enc_rep = self.online_encoder(x, edge_index.clone())
        rep = self.encoder_to_decoder(enc_rep)
        rep = self.projector(rep, edge_index.clone())
        recon = self.decoder(rep, edge_index.clone())
        return enc_rep, recon

    @torch.no_grad()
    def std_tgt_embedding(self, x, edge_index):
        s_rep = self.online_encoder(x, edge_index.clone())
        t_rep = self.target_encoder(x, edge_index.clone())
        return s_rep, t_rep

    def avg_readout(self, rep_pos_x, edge_index):
        src, dst = edge_index[0], edge_index[1]
        neighbor_sum = scatter_add(rep_pos_x[src], dst, dim=0, dim_size=rep_pos_x.size(0))
        neighbor_count = scatter_add(torch.ones_like(src, dtype=torch.float), dst, dim=0, dim_size=rep_pos_x.size(0))
        neighbor_count = neighbor_count.clamp(min=1)
        summary = neighbor_sum / neighbor_count.unsqueeze(-1)
        return torch.sigmoid(summary)

    def discriminate(self, z, summary, sigmoid=True):
        assert isinstance(summary, torch.Tensor), "Summary should be a torch.Tensor"
        value = torch.matmul(z, torch.matmul(self.weight, summary.t()))
        return torch.sigmoid(value) if sigmoid else value

    def match_loss(self, rep, rep_t, mask_nodes, t=2):
        """
        带平滑动态权重的对比损失函数
        """
        pox_x_index, pox_y_index = mask_nodes, mask_nodes
        neg_x_index, neg_y_index = self.generate_neg_nodes(mask_nodes)

        std_emb = F.normalize(rep.clone(), p=2, dim=-1)
        tgt_emb = F.normalize(rep_t.clone(), p=2, dim=-1)

        pox_x = std_emb[pox_x_index]
        pox_y = tgt_emb[pox_y_index]
        neg_x = std_emb[neg_x_index]
        neg_y = tgt_emb[neg_y_index]

        pos_sim = (pox_x * pox_y).sum(dim=-1)
        pos_cos = (0.5 * (1 + pos_sim)).pow(t)
        pos_loss = -torch.log(pos_cos)

        neg_sim = (neg_x * neg_y).sum(dim=-1)

        neg_weight = 1.0 - torch.clamp(neg_sim, min=0.0)

        neg_cos = (0.5 * (1 + neg_sim)).pow(t)

        neg_loss = -torch.log(1 - neg_cos) * neg_weight

        loss = 0.5 * (pos_loss.mean() + neg_loss.mean())
        return loss
    def dgi_loss(self, pos_z, neg_z, summary):
        pos_loss = -torch.log(self.discriminate(pos_z, summary, sigmoid=True) + 1e-15).mean()
        neg_loss = -torch.log(1 - self.discriminate(neg_z, summary, sigmoid=True) + 1e-15).mean()
        return pos_loss + neg_loss

    def CL_Loss(self, pos_z, neg_z, summary):
        pos_loss = -torch.log(self.discriminate(pos_z, summary, sigmoid=True) + 1e-15).mean()
        neg_loss = -torch.log(1 - self.discriminate(neg_z, summary, sigmoid=True) + 1e-15).mean()
        Cos_loss = -torch.log(1 - F.cosine_similarity(pos_z, neg_z) + 1e-15).mean()
        loss = Cos_loss + pos_loss + neg_loss
        return loss