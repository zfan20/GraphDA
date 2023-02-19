import torch
from dataloader import BasicDataset
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import SGConv

from gtn_propagation import GeneralPropagation
import warnings
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from torch_geometric.nn.conv import MessagePassing, GCNConv
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
import torch_geometric
from torch_sparse import SparseTensor


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 args,
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.args = args
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = self.args.recdim #config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)


class SGCN(nn.Module):
    def __init__(self,
                 args,
                 dataset:BasicDataset):
        super(SGCN, self).__init__()
        self.args = args
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        train_ui_mat = dataset.UserItemNet
        userind_arr, itemind_arr = train_ui_mat.nonzero()
        itemind_arr_added = [itemind+self.num_users for itemind in itemind_arr]
        edge_index = torch.tensor([np.concatenate((userind_arr, itemind_arr_added)),
                                np.concatenate((itemind_arr_added, userind_arr))], dtype=torch.long)

        self.latent_dim = self.args.recdim #config['latent_dim_rec']
        self.__init_weight()
        all_emb = torch.cat([self.embedding_user.weight, self.embedding_item.weight])
        self.graph_data = Data(x=all_emb, edge_index=edge_index).to(self.args.device)
        self.sgcn_conv = SGConv(in_channels=self.latent_dim, out_channels=self.latent_dim, K=args.layer, cached=True, add_self_loops=False).to(self.args.device)

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim).to(self.args.device)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim).to(self.args.device)

    def computer(self):
        sgcn_out = self.sgcn_conv(self.graph_data.x, self.graph_data.edge_index)
        users, items = torch.split(sgcn_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = torch.matmul(users_emb, items_emb.t())
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) +
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

        

class LightGCN(BasicModel):
    def __init__(self, 
                 args,
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.args = args
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.args.recdim #self.config['latent_dim_rec']
        self.n_layers = self.args.layer #self.config['lightGCN_n_layers']
        self.keep_prob = self.args.keepprob #self.config['keep_prob']
        self.A_split = False #self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph(self.dataset.UserItemNet)

    def reset_all(self):
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.Graph = self.dataset.getSparseGraph(self.dataset.UserItemNet)

    def reset_all_uuii(self):
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.Graph = self.dataset.getSparseGraph(self.dataset.UserItemNet, include_uuii=True)


    def reset_graph(self):
        self.Graph = self.dataset.getSparseGraph(self.dataset.UserItemNet)

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.args.dropout:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getUsersUsers(self, users):
        all_users, _ = self.computer()
        users_emb = all_users[users.long()]

        return self.f(torch.matmul(users_emb, all_users.t()))

    def getItemsItems(self, items):
        _, all_items = self.computer()
        items_emb = all_items[items.long()]
        
        return self.f(torch.matmul(items_emb, all_items.t()))

    def getItemsRating(self, items):
        all_users, all_items = self.computer()
        items_emb = all_items[items.long()]
        users_emb = all_users
        item_rating = self.f(torch.matmul(items_emb, users_emb.t()))
        return item_rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
       
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma



class GTN(BasicModel):
    def __init__(self, args, dataset: BasicDataset):
        super(GTN, self).__init__()
        self.dataset: dataloader.BasicDataset = dataset
        self.args = args
        self.__init_weight()

        self.gp = GeneralPropagation(self.args.layer, self.args.alpha, cached=True, args=args)

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        #self.latent_dim = self.config['latent_dim_rec']
        #self.n_layers = self.args.K
        #self.keep_prob = self.config['keep_prob']
        #self.A_split = self.config['A_split']
        self.latent_dim = self.args.recdim #self.config['latent_dim_rec']
        self.n_layers = self.args.layer #self.config['lightGCN_n_layers']
        self.keep_prob = self.args.keepprob #self.config['keep_prob']
        self.A_split = False #self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph(self.dataset.UserItemNet)

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self, corrupted_graph=None):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        if self.args.dropout:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                if corrupted_graph == None:
                    g_droped = self.Graph
                else:
                    g_droped = corrupted_graph
        else:
            if corrupted_graph == None:
                g_droped = self.Graph
            else:
                g_droped = corrupted_graph

        # our GCNs
        x = all_emb
        rc = g_droped.indices()
        r = rc[0]
        c = rc[1]
        num_nodes = g_droped.shape[0]
        edge_index = SparseTensor(row=r, col=c, value=g_droped.values(), sparse_sizes=(num_nodes, num_nodes))
        emb, embs = self.gp.forward(x, edge_index, mode=self.args.gcn_model)
        light_out = emb

        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        #loss = torch.mean(-torch.log(torch.sigmoid(pos_scores-neg_scores)+1e-24))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma


class UltraGCN(nn.Module):
    def __init__(self, args, dataset, constraint_mat, ii_constraint_mat, ii_neighbor_mat):
        super(UltraGCN, self).__init__()
        self.args = args
        self.user_num  = dataset.n_users
        self.item_num  = dataset.m_items
        self.embedding_dim = self.args.recdim
        self.w1 = self.args.w1
        self.w2 = self.args.w2
        self.w3 = self.args.w3
        self.w4 = self.args.w4

        self.negative_weight = self.args.negative_weight
        self.gamma = self.args.decay
        self.lambda_ = self.args.lambda_Iweight

        self.user_embeds = nn.Embedding(self.user_num, self.embedding_dim)
        self.item_embeds = nn.Embedding(self.item_num, self.embedding_dim)

        self.constraint_mat = constraint_mat
        self.ii_constraint_mat = ii_constraint_mat
        self.ii_neighbor_mat = ii_neighbor_mat


    def get_omegas(self, users, pos_items, neg_items):
        device = self.args.device
        #self.constraint_mat['beta_uD'] = self.constraint_mat['beta_uD'].to(device)
        #self.constraint_mat['beta_iD'] = self.constraint_mat['beta_iD'].to(device)
        if self.w2 > 0:
            pos_weight = torch.mul(self.constraint_mat['beta_uD'][users], self.constraint_mat['beta_iD'][pos_items]).to(device)
            pow_weight = self.w1 + self.w2 * pos_weight
        else:
            pos_weight = self.w1 * torch.ones(len(pos_items)).to(device)

        # users = (users * self.item_num).unsqueeze(0)
        if self.w4 > 0:
            neg_weight = torch.mul(self.constraint_mat['beta_uD'][users], self.constraint_mat['beta_iD'][neg_items]).to(device)
            neg_weight = self.w3 + self.w4 * neg_weight
        else:
            neg_weight = self.w3 * torch.ones(neg_items.size(0) * neg_items.size(1)).to(device)


        weight = torch.cat((pow_weight, neg_weight))
        return weight

    def cal_loss_L(self, users, pos_items, neg_items, omega_weight):
        device = self.args.device
        user_embeds = self.user_embeds(users.to(device))
        pos_embeds = self.item_embeds(pos_items.to(device))
        neg_embeds = self.item_embeds(neg_items.to(device))

        pos_scores = (user_embeds * pos_embeds).sum(dim=-1) # batch_size
        user_embeds = user_embeds.unsqueeze(1)
        neg_scores = (user_embeds * neg_embeds).sum(dim=-1) # batch_size * negative_num

        neg_labels = torch.zeros(neg_scores.size()).to(device)
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_labels, weight = omega_weight[len(pos_scores):], reduction='none').mean(dim = -1)

        pos_labels = torch.ones(pos_scores.size()).to(device)
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels, weight = omega_weight[:len(pos_scores)], reduction='none')

        loss = pos_loss + neg_loss * self.negative_weight

        return loss.sum()

    def cal_loss_I(self, users, pos_items):
        device = self.args.device
        #self.ii_neighbor_mat = self.ii_neighbor_mat.to(device)
        neighbor_embeds = self.item_embeds(self.ii_neighbor_mat[pos_items].to(device))    # len(pos_items) * num_neighbors * dim
        #self.ii_constraint_mat = self.ii_constraint_mat.to(device)
        sim_scores = self.ii_constraint_mat[pos_items].to(device)     # len(pos_items) * num_neighbors
        user_embeds = self.user_embeds(users.to(device)).unsqueeze(1)

        loss = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()

        # loss = loss.sum(-1)
        return loss.sum()

    def norm_loss(self):
        loss = 0.0
        for parameter in self.parameters():
            loss += torch.sum(parameter ** 2)
        return loss / 2

    def forward(self, users, pos_items, neg_items):
        omega_weight = self.get_omegas(users, pos_items, neg_items)

        loss = self.cal_loss_L(users, pos_items, neg_items, omega_weight)
        loss += self.gamma * self.norm_loss()
        loss += self.lambda_ * self.cal_loss_I(users, pos_items)
        return loss

    def getUsersRating(self, users):
        users_emb = self.user_embeds(users.long())
        items_emb = self.item_embeds.weight
        return users_emb.mm(items_emb.t())
