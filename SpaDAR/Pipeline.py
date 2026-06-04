class SC_pipeline:
    def __init__(self, adata, edge_index, num_clusters, device, config, roundseed=0, imputation=False):
        seed = config['seed'] + roundseed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.enabled = False
        torch.use_deterministic_algorithms(True)

        self.device = device
        self.adata = adata
        self.edge_index = edge_index
        self.train_config = config['train']
        self.model_config = config['model']
        self.num_clusters = num_clusters
        self.imputation = imputation

        if self.imputation:
            self.X = torch.FloatTensor(self.adata.X.copy()).to(self.device)
        else:
            self.X = torch.FloatTensor(self.adata.obsm['X_pca'].copy()).to(self.device)
        self.edge_index = self.edge_index.to(self.device)

        self.input_dim = self.X.shape[-1]
        self.model = SpaDAR_model(self.input_dim, self.model_config, imputation=self.imputation).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=0.001,
            weight_decay=3e-4,
        )

        self.sampler = GLNSampler(self.num_clusters, self.device)
        self.anchor_pair = None

    def trian(self):
        neighbors = self.train_config['topk_neighs']
        pbar = tqdm(range(self.train_config['epochs']))
        for epoch in pbar:
            if epoch % self.train_config['t_step'] == 0 and epoch > 1:
                self.model.eval()
                s_rep, t_rep = self.model.std_tgt_embedding(self.X, self.edge_index)
                self.anchor_pair = self.sampler(self.edge_index, F.normalize(s_rep, dim=-1, p=2),
                                                F.normalize(t_rep, dim=-1, p=2), neighbors, cluster_method="kmeans")

            self.model.train()
            self.optimizer.zero_grad()
            mean_loss, rec_loss, tri_loss = self.model(self.X, self.edge_index, self.anchor_pair)
            loss = self.train_config['w_recon'] * rec_loss + self.train_config['w_mean'] * mean_loss + \
                   self.train_config['w_tri'] * tri_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
            self.optimizer.step()
            with torch.no_grad():
                self.model.momentum_update()
            pbar.set_description(
                "Epoch {0} total loss={1:.3f} recon loss={2:.3f} mean loss={3:.3f} tri loss={4:.3f}".format(
                    epoch, loss, rec_loss, mean_loss, tri_loss),
                refresh=True)

    def process(self):
        self.model.eval()
        enc_rep, recon = self.model.evaluate(self.X, self.edge_index)
        enc_rep = enc_rep.to('cpu').detach().numpy()
        recon = recon.to('cpu').detach().numpy()
        recon[recon < 0] = 0

        self.adata.obsm['latent'] = enc_rep
        self.adata.obsm['ReX'] = recon
        return enc_rep, recon

class SC_BC_pipeline:
    def __init__(self, adata, edge_index, num_clusters, device, config, roundseed=0, imputation=False):
        seed = config['seed'] + roundseed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.enabled = False
        torch.use_deterministic_algorithms(True)

        self.device = device
        self.adata = adata
        self.edge_index = edge_index
        self.train_config = config['train']
        self.model_config = config['model']
        self.num_clusters = num_clusters
        self.imputation = imputation
        self.batch_id = torch.tensor(adata.obs['slice_id'].to_numpy(), dtype=torch.float32)

        if self.imputation:
            self.X = torch.FloatTensor(self.adata.X.copy()).to(self.device)
        else:
            self.X = torch.FloatTensor(self.adata.obsm['X_pca'].copy()).to(self.device)
        self.edge_index = self.edge_index.to(self.device)

        self.input_dim = self.X.shape[-1]
        self.model = SpaDAR_model(self.input_dim, self.model_config, imputation=self.imputation).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=0.001,
            weight_decay=3e-4,
        )

        self.sampler = GLNSampler_BC(self.num_clusters, self.device)
        self.anchor_pair = None

    def trian(self):
        neighbors = self.train_config['topk_neighs']
        neighbors_inter = self.train_config['topk_neighs_inter']
        pbar = tqdm(range(self.train_config['epochs']))
        for epoch in pbar:
            if epoch % self.train_config['t_step'] == 0 and epoch > 1:
                self.model.eval()
                s_rep, t_rep = self.model.std_tgt_embedding(self.X, self.edge_index)
                # (self, adj, enc_rep, batch_id, top_k, top_k_inter, cluster_method="kmeans")
                self.anchor_pair = self.sampler(self.edge_index, F.normalize(s_rep, dim=-1, p=2), self.batch_id, neighbors, neighbors_inter, cluster_method="kmeans")

            self.model.train()
            self.optimizer.zero_grad()
            mean_loss, rec_loss, tri_loss = self.model(self.X, self.edge_index, self.anchor_pair)
            loss = self.train_config['w_recon'] * rec_loss + self.train_config['w_mean'] * mean_loss + \
                   self.train_config['w_tri'] * tri_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
            self.optimizer.step()
            with torch.no_grad():
                self.model.momentum_update()
            pbar.set_description(
                "Epoch {0} total loss={1:.3f} recon loss={2:.3f} mean loss={3:.3f} tri loss={4:.3f}".format(
                    epoch, loss, rec_loss, mean_loss, tri_loss),
                refresh=True)
    def process(self):
        self.model.eval()
        enc_rep, recon = self.model.evaluate(self.X, self.edge_index)
        enc_rep = enc_rep.to('cpu').detach().numpy()
        recon = recon.to('cpu').detach().numpy()
        recon[recon < 0] = 0

        self.adata.obsm['latent'] = enc_rep
        self.adata.obsm['ReX'] = recon
        return enc_rep, recon


import os
import random
import numpy as np
import scipy.sparse as sp
import networkx as nx
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.backends import cudnn
from torch.cuda.amp import autocast, GradScaler
import hnswlib
from sklearn.neighbors import NearestNeighbors
import anndata as ad

from .Models import SpaDAR_model
from .GLNS import GLNSampler, GLNSampler_BC

def nn_approx(ds1, ds2, names1, names2, knn=50):
    dim = ds2.shape[1]
    num_elements = ds2.shape[0]
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=100, M=16)
    p.set_ef(10)
    p.add_items(ds2)
    ind, distances = p.knn_query(ds1, k=knn)
    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((names1[a], names2[b_i]))
    return match

def mnn(ds1, ds2, names1, names2, knn=20, approx=True):
    if approx:
        match1 = nn_approx(ds1, ds2, names1, names2, knn=knn)
        match2 = nn_approx(ds2, ds1, names2, names1, knn=knn)
    else:
        nn1 = NearestNeighbors(n_neighbors=knn, p=2).fit(ds2)
        ind1 = nn1.kneighbors(ds1, return_distance=False)
        match1 = set([(names1[a], names2[b_i]) for a, b in zip(range(ds1.shape[0]), ind1) for b_i in b])

        nn2 = NearestNeighbors(n_neighbors=knn, p=2).fit(ds1)
        ind2 = nn2.kneighbors(ds2, return_distance=False)
        match2 = set([(names2[a], names1[b_i]) for a, b in zip(range(ds2.shape[0]), ind2) for b_i in b])

    mutual = match1 & set([(b, a) for a, b in match2])
    return mutual

def create_dictionary_mnn(adata_pair, use_rep, batch_name, k=50, approx=True):
    cell_names = adata_pair.obs_names
    batch_list = adata_pair.obs[batch_name]

    batches = batch_list.unique()
    assert len(batches) == 2, "This function is designed for pairwise alignment."

    i, j = batches[0], batches[1]

    cells_i = cell_names[batch_list == i]
    cells_j = cell_names[batch_list == j]

    ds1 = adata_pair[cells_i].obsm[use_rep]
    ds2 = adata_pair[cells_j].obsm[use_rep]

    match = mnn(ds1, ds2, cells_i, cells_j, knn=k, approx=approx)

    G = nx.Graph()
    G.add_edges_from(match)
    node_names = np.array(G.nodes)
    anchors = list(node_names)
    adj = nx.adjacency_matrix(G)
    tmp = np.split(adj.indices, adj.indptr[1:-1])

    mnns = {}
    for idx in range(len(anchors)):
        key = anchors[idx]
        names = list(node_names[tmp[idx]])
        mnns[key] = names
    return mnns

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def combine_subgraph_edges(adj1, adj2):
    N1 = adj1.shape[0]
    N2 = adj2.shape[0]

    adj1_coo = adj1.tocoo()
    adj2_coo = adj2.tocoo()

    row = np.concatenate([adj1_coo.row, adj2_coo.row + N1])
    col = np.concatenate([adj1_coo.col, adj2_coo.col + N1])
    data = np.concatenate([adj1_coo.data, adj2_coo.data])

    joint_adj = sp.coo_matrix((data, (row, col)), shape=(N1 + N2, N1 + N2))
    return sparse_mx_to_torch_sparse_tensor(joint_adj)


class SC_2D_pipeline:
    def __init__(self, Batch_list, config, device, iter_comb=None, imputation=False, efficient_dgi=False, roundseed=0):
        seed = config['seed'] + roundseed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)

        self.device = device
        self.Batch_list = Batch_list
        self.config = config
        self.train_config = config['train']
        self.model_config = config['model']
        self.imputation = imputation
        self.efficient_dgi = efficient_dgi

        if iter_comb is None:
            self.iter_comb = [(i, i + 1) for i in range(len(self.Batch_list) - 1)]
        else:
            self.iter_comb = iter_comb

        if self.imputation:
            self.input_dim = self.Batch_list[0].X.shape[1]
        else:
            self.input_dim = self.Batch_list[0].obsm['X_pca'].shape[1]

        self.model = SpaDAR_model(self.input_dim, self.model_config, imputation=self.imputation,
                                    efficient_dgi=self.efficient_dgi).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=0.001,
            weight_decay=3e-4,
        )

    def _get_node_features(self, adata):
        if self.imputation:
            return torch.FloatTensor(adata.X.toarray() if sp.issparse(adata.X) else adata.X)
        else:
            return torch.FloatTensor(adata.obsm['X_pca'])

    def _clear_gcn_cache(self):
        for m in self.model.modules():
            if m.__class__.__name__ == 'GCNConv':
                m._cached_edge_index = None
                m._cached_adj_t = None

    def train(self):
        n_epochs = self.train_config['epochs']
        knn_neigh = self.train_config['topk_neighs']
        update_freq = self.train_config['t_step']

        print(f"Starting 2D Integration with {len(self.Batch_list)} slices. Pairwise subgraph training active.")

        pbar = tqdm(range(n_epochs))
        for epoch in pbar:
            if epoch % update_freq == 0 and epoch > 1:
                self.model.eval()
                with torch.no_grad():
                    for i, adata in enumerate(self.Batch_list):
                        X_i = self._get_node_features(adata).to(self.device)
                        edge_idx_i = sparse_mx_to_torch_sparse_tensor(adata.uns['adj']).to(self.device)
                        self._clear_gcn_cache()
                        z_i, _ = self.model.evaluate(X_i, edge_idx_i)
                        adata.obsm['current_z'] = z_i.cpu().numpy()

                self.subgraph_data = []
                for (i, j) in self.iter_comb:
                    adata_i = self.Batch_list[i]
                    adata_j = self.Batch_list[j]

                    batch_pair = ad.concat([adata_i, adata_j], label="batch_name", keys=[f"slice_{i}", f"slice_{j}"])
                    mnn_dict = create_dictionary_mnn(batch_pair, use_rep='current_z', batch_name='batch_name',
                                                     k=knn_neigh, approx=True)

                    cellname_by_batch_dict = {
                        f"slice_{i}": batch_pair.obs_names[batch_pair.obs['batch_name'] == f"slice_{i}"].values,
                        f"slice_{j}": batch_pair.obs_names[batch_pair.obs['batch_name'] == f"slice_{j}"].values
                    }

                    anchor_list, positive_list, negative_list = [], [], []
                    for anchor in mnn_dict.keys():
                        anchor_list.append(anchor)
                        positive_spot = mnn_dict[anchor][0]
                        positive_list.append(positive_spot)

                        anchor_batch = batch_pair.obs.loc[anchor, 'batch_name']
                        opp_batch = f"slice_{j}" if anchor_batch == f"slice_{i}" else f"slice_{i}"
                        section_size = len(cellname_by_batch_dict[opp_batch])
                        negative_list.append(cellname_by_batch_dict[opp_batch][np.random.randint(section_size)])

                    batch_as_dict = dict(zip(list(batch_pair.obs_names), range(0, batch_pair.shape[0])))
                    anchor_ind = list(map(lambda _: batch_as_dict[_], anchor_list))
                    positive_ind = list(map(lambda _: batch_as_dict[_], positive_list))
                    negative_ind = list(map(lambda _: batch_as_dict[_], negative_list))

                    X_pair = torch.cat([self._get_node_features(adata_i), self._get_node_features(adata_j)], dim=0).to(
                        self.device)
                    edge_index_pair = combine_subgraph_edges(adata_i.uns['adj'], adata_j.uns['adj']).to(self.device)

                    anchor_pair = (
                        torch.LongTensor(np.array(anchor_ind)).to(self.device),
                        torch.LongTensor(np.array(positive_ind)).to(self.device),
                        torch.LongTensor(np.array(negative_ind)).to(self.device)
                    )

                    self.subgraph_data.append({
                        'X': X_pair,
                        'edge_index': edge_index_pair,
                        'anchor_pair': anchor_pair
                    })

                torch.cuda.empty_cache()

            self.model.train()
            total_rec_loss = 0
            total_mean_loss = 0
            total_tri_loss = 0

            if epoch <= update_freq:
                self.optimizer.zero_grad()
                for i, adata in enumerate(self.Batch_list):
                    X_i = self._get_node_features(adata).to(self.device)
                    edge_idx_i = sparse_mx_to_torch_sparse_tensor(adata.uns['adj']).to(self.device)
                    fake_anchor = (torch.tensor([0]).to(self.device), torch.tensor([0]).to(self.device),
                                   torch.tensor([0]).to(self.device))

                    self._clear_gcn_cache()
                    mean_loss, rec_loss, tri_loss = self.model(X_i, edge_idx_i, fake_anchor)

                    loss = self.train_config['w_recon'] * rec_loss + self.train_config['w_mean'] * mean_loss
                    loss.backward()
                    total_rec_loss += rec_loss.item()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optimizer.step()
                with torch.no_grad():
                    self.model.momentum_update()

            else:
                self.optimizer.zero_grad()
                for sub_data in self.subgraph_data:
                    X_sub = sub_data['X']
                    edge_sub = sub_data['edge_index']
                    anchors_sub = sub_data['anchor_pair']

                    self._clear_gcn_cache()
                    mean_loss, rec_loss, tri_loss = self.model(X_sub, edge_sub, anchors_sub)

                    loss = self.train_config['w_recon'] * rec_loss + \
                           self.train_config['w_mean'] * mean_loss + \
                           self.train_config['w_tri'] * tri_loss
                    loss.backward()

                    total_rec_loss += rec_loss.item()
                    total_mean_loss += mean_loss.item()
                    total_tri_loss += tri_loss.item()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optimizer.step()
                with torch.no_grad():
                    self.model.momentum_update()

                epoch_total_loss = (self.train_config['w_recon'] * total_rec_loss +
                                    self.train_config['w_mean'] * total_mean_loss +
                                    self.train_config['w_tri'] * total_tri_loss)

                pbar.set_description(
                    "Epoch {0} total loss={1:.3f} recon loss={2:.3f} mean loss={3:.3f} tri loss={4:.3f}".format(
                        epoch, epoch_total_loss, total_rec_loss, total_mean_loss, total_tri_loss),
                    refresh=True)

    def process(self):
        self.model.eval()
        latent_list = []
        recon_list = []

        with torch.no_grad():
            for adata in self.Batch_list:
                X_i = self._get_node_features(adata).to(self.device)
                edge_idx_i = sparse_mx_to_torch_sparse_tensor(adata.uns['adj']).to(self.device)

                self._clear_gcn_cache()
                enc_rep, recon = self.model.evaluate(X_i, edge_idx_i)

                enc_rep = enc_rep.to('cpu').detach().numpy()
                recon = recon.to('cpu').detach().numpy()
                recon[recon < 0] = 0

                adata.obsm['latent'] = enc_rep
                adata.obsm['ReX'] = recon

                latent_list.append(enc_rep)
                recon_list.append(recon)

        return np.concatenate(latent_list, axis=0), np.concatenate(recon_list, axis=0)