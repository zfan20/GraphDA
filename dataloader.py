import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from time import time
import copy
import pickle

def pload(path):
    with open(path, 'rb') as f:
        res = pickle.load(f)
    print('load path = {} object'.format(path))
    return res

def pstore(x, path):
    with open(path, 'wb') as f:
        pickle.dump(x, f)
    print('store object in path = {} ok'.format(path))

class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    def getSparseGraph(self, ui_mat):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError


class Loader(BasicDataset):

    def __init__(self, args):
        # train or test
        datapath = './data/'
        data_file = datapath+'/'+args.data_name+'.txt'
        self.args = args
        self.split = False
        self.folds = self.args.a_fold
        self.n_user = 0
        self.m_item = 0
        self.path = './data/'
        trainUniqueUsers, trainItem, trainUser = [], [], []
        validUniqueUsers, validItem, validUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.trainSize = 0
        self.validSize = 0
        self.testSize = 0

        lines = open(data_file).readlines()
        for line in lines:
            user, items = line.strip().split(' ', 1)
            items = items.split(' ')
            uid = int(user)-1
            itemids = [int(itemstr)-1 for itemstr in items]
            trainUniqueUsers.append(uid)
            trainUser.extend([uid] * (len(items)-2))
            trainItem.extend(itemids[:-2])
            self.n_user = max(self.n_user, uid)
            self.m_item = max(self.m_item, max(itemids))
            self.trainSize += len(items)-2

            validUniqueUsers.append(uid)
            validUser.extend([uid])
            validItem.extend([itemids[-2]])
            self.validSize += 1

            testUniqueUsers.append(uid)
            testUser.extend([uid])
            testItem.extend([itemids[-1]])
            self.testSize += 1

        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        self.validUniqueUsers = np.array(validUniqueUsers)
        self.validUser = np.array(validUser)
        self.validItem = np.array(validItem)
        
        self.Graph = None
        print(f"#of users: {self.n_user} and #of items: {self.m_items}")
        print(f"{self.trainSize} interactions for training")
        print(f"{self.validSize} interactions for validation")
        print(f"{self.testSize} interactions for testing")
        print(f"{self.args.data_name} Sparsity : {(self.trainSize + self.validSize + self.testSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.

        self.UserUserNet = None
        self.ItemItemNet = None
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict, self.test_pred_mask_mat = self.__build_test()
        self.__validDict = self.__build_valid()
        self.valid_pred_mask_mat = self.UserItemNet

    def random_sample_edges(self, adj, n, exclude):
        itr = self.sample_forever(adj, exclude=exclude)
        return [next(itr) for _ in range(n)]

    def sample_forever(self, adj, exclude):
        """Randomly random sample edges from adjacency matrix, `exclude` is a set
        which contains the edges we do not want to sample and the ones already sampled
        """
        while True:
            # t = tuple(np.random.randint(0, adj.shape[0], 2))
            # t = tuple(random.sample(range(0, adj.shape[0]), 2))
            t = tuple(np.random.choice(adj.shape[0], 2, replace=False))
            if t not in exclude:
                yield t
                exclude.add(t)
                exclude.add((t[1], t[0]))

    def reset_graph(self, newdata):
        new_row, new_col, new_val = newdata
        self.UserItemNet = csr_matrix((new_val, (new_row, new_col)), shape=(self.n_user, self.m_item))
        print('reset graph nnz: ', self.UserItemNet.nnz, ', density: ', self.UserItemNet.nnz/(self.n_user*self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1.
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.

    def reset_graph_ease(self, newdata):
        oldUserItemNet = self.UserItemNet

        new_row, new_col, new_val = newdata
        self.UserItemNet = csr_matrix((new_val, (new_row, new_col)), shape=(self.n_user, self.m_item))
        print('reset graph nnz: ', self.UserItemNet.nnz, ', density: ', self.UserItemNet.nnz/(self.n_user*self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1.
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.

        uuG = oldUserItemNet.dot(oldUserItemNet.T).toarray()
        uu_diagIndices = np.diag_indices(uuG.shape[0])
        uuG[uu_diagIndices] += self.args.uu_lambda
        uuP = np.linalg.inv(uuG)
        self.UserUserNet = uuP / (-np.diag(uuP))
        self.UserUserNet[uu_diagIndices] = 0
        self.UserUserNet[self.UserUserNet<0] = 0
        self.UserUserNet = csr_matrix(self.UserUserNet)
        print('uu net nnz: ', self.UserUserNet.nnz, ', density: ', self.UserUserNet.nnz/(self.n_user*self.n_user))


        iiG = oldUserItemNet.T.dot(oldUserItemNet).toarray()
        ii_diagIndices = np.diag_indices(iiG.shape[0])
        iiG[ii_diagIndices] += self.args.ii_lambda
        iiP = np.linalg.inv(iiG)
        self.ItemItemNet = iiP / (-np.diag(iiP))
        self.ItemItemNet[ii_diagIndices] = 0
        self.ItemItemNet[self.ItemItemNet<0] = 0
        self.ItemItemNet = csr_matrix(self.ItemItemNet)
        print('ii net nnz: ', self.ItemItemNet.nnz, ', density: ', self.ItemItemNet.nnz/(self.m_item*self.m_item))


    def reset_graph_uuii(self, newdata):
        [newuidata, newuudata, newiidata] = newdata

        new_row, new_col, new_val = newuidata
        self.UserItemNet = csr_matrix((new_val, (new_row, new_col)), shape=(self.n_user, self.m_item))
        print('reset graph nnz: ', self.UserItemNet.nnz, ', density: ', self.UserItemNet.nnz/(self.n_user*self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1.
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.

        new_uu_row, new_uu_col, new_uu_val = newuudata
        #uu_Net = csr_matrix((new_val, (new_row, new_col)), shape=(self.n_user, self.n_user))
        #self.UserUserNet = uu_Net.dot(uu_Net)
        #self.UserUserNet[self.UserUserNet>0] = 1.0
        self.UserUserNet = csr_matrix((new_uu_val, (new_uu_row, new_uu_col)), shape=(self.n_user, self.n_user))
        self.UserUserNet.setdiag(0)

        new_ii_row, new_ii_col, new_ii_val = newiidata
        #ii_Net = csr_matrix((new_val, (new_row, new_col)), shape=(self.m_item, self.m_item))
        #self.ItemItemNet = ii_Net.dot(ii_Net)
        #self.ItemItemNet[self.ItemItemNet>0] = 1.0
        self.ItemItemNet = csr_matrix((new_ii_val, (new_ii_row, new_ii_col)), shape=(self.m_item, self.m_item))
        self.ItemItemNet.setdiag(0)


    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.trainSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def validDict(self):
        return self.__validDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self, ui_spmat, include_uuii=False):
        print("generating adjacency matrix")
        s = time()
        adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = ui_spmat.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        print('before adj mat nnz', adj_mat.nnz)
        if include_uuii and (self.UserUserNet is not None and self.ItemItemNet is not None):
            print('Including UU and II')
            uu_mat = self.UserUserNet.tolil()
            ii_mat = self.ItemItemNet.tolil()
            adj_mat[:self.n_users, :self.n_users] = uu_mat
            adj_mat[self.n_users:, self.n_users:] = ii_mat
        adj_mat = adj_mat.todok()
        print('adj mat nnz', adj_mat.nnz)
        #adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        end = time()
        print(f"costing {end-s}s, saved norm_mat...")
        graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        return graph.coalesce().to(self.args.device)
        
    #def getSparseGraph(self, ui_spmat):
    #    print("loading adjacency matrix")
    #    if self.Graph is None:
    #        try:
    #            pre_adj_mat = sp.load_npz(self.path + self.args.data_name+'_s_pre_adj_mat.npz')
    #            print("successfully loaded...")
    #            norm_adj = pre_adj_mat
    #        except :
    #            print("generating adjacency matrix")
    #            s = time()
    #            adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
    #            adj_mat = adj_mat.tolil()
    #            R = self.UserItemNet.tolil()
    #            adj_mat[:self.n_users, self.n_users:] = R
    #            adj_mat[self.n_users:, :self.n_users] = R.T
    #            adj_mat = adj_mat.todok()
    #            # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
    #            
    #            rowsum = np.array(adj_mat.sum(axis=1))
    #            d_inv = np.power(rowsum, -0.5).flatten()
    #            d_inv[np.isinf(d_inv)] = 0.
    #            d_mat = sp.diags(d_inv)
    #            
    #            norm_adj = d_mat.dot(adj_mat)
    #            norm_adj = norm_adj.dot(d_mat)
    #            norm_adj = norm_adj.tocsr()
    #            end = time()
    #            print(f"costing {end-s}s, saved norm_mat...")
    #            sp.save_npz(self.path + self.args.data_name+'_s_pre_adj_mat.npz', norm_adj)

    #        if self.split == True:
    #            self.Graph = self._split_A_hat(norm_adj)
    #            print("done split matrix")
    #        else:
    #            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
    #            self.Graph = self.Graph.coalesce().to(self.args.device)
    #            print("don't split the matrix")
    #    return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_pred_mask_mat = copy.deepcopy(self.UserItemNet)
        for i, item in enumerate(self.validItem):
            user = self.validUser[i]
            test_pred_mask_mat[user, item] = 1.0
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data, test_pred_mask_mat

    def __build_valid(self):
        """
        return:
            dict: {user: [items]}
        """
        valid_data = {}
        for i, item in enumerate(self.validItem):
            user = self.validUser[i]
            if valid_data.get(user):
                valid_data[user].append(item)
            else:
                valid_data[user] = [item]
        return valid_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def getConstraintMat(self):
        items_D = np.sum(self.UserItemNet, axis = 0).reshape(-1)
        users_D = np.sum(self.UserItemNet, axis = 1).reshape(-1)

        beta_uD = (np.sqrt(users_D + 1) / (users_D+1)).reshape(-1, 1)
        beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)

        constraint_mat = {"beta_uD": torch.from_numpy(beta_uD).reshape(-1),
                          "beta_iD": torch.from_numpy(beta_iD).reshape(-1)}

        return constraint_mat

    def get_ii_constraint_mat(self, ii_diagonal_zero = False):
        ii_cons_mat_path = self.path + self.args.data_name + '_ii_constraint_mat'
        ii_neigh_mat_path = self.path + self.args.data_name + '_ii_neighbor_mat'

        if os.path.exists(ii_cons_mat_path):
            ii_constraint_mat = pload(ii_cons_mat_path)
            ii_neighbor_mat = pload(ii_neigh_mat_path)
        else:
            print('Computing \\Omega for the item-item graph... ')
            A = self.UserItemNet.T.dot(self.UserItemNet)      # I * I
            n_items = A.shape[0]
            res_mat = torch.zeros((n_items, self.args.ii_neighbor_num))
            res_sim_mat = torch.zeros((n_items, self.args.ii_neighbor_num))
            if ii_diagonal_zero:
                A[range(n_items), range(n_items)] = 0
            items_D = np.sum(A, axis = 0).reshape(-1)
            users_D = np.sum(A, axis = 1).reshape(-1)

            beta_uD = (np.sqrt(users_D + 1) / (users_D+1)).reshape(-1, 1)
            beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
            all_ii_constraint_mat = torch.from_numpy(beta_uD.dot(beta_iD))
            for i in range(n_items):
                row = all_ii_constraint_mat[i] * torch.from_numpy(A.getrow(i).toarray()[0])
                row_sims, row_idxs = torch.topk(row, self.args.ii_neighbor_num)
                res_mat[i] = row_idxs
                res_sim_mat[i] = row_sims
                if i % 15000 == 0:
                    print('i-i constraint matrix {} ok'.format(i))

            print('Computation \\Omega OK!')
            ii_neighbor_mat = res_mat
            ii_constraint_mat = res_sim_mat
            pstore(ii_neighbor_mat, ii_neigh_mat_path)
            pstore(ii_constraint_mat, ii_cons_mat_path)
        #return res_mat.long(), res_sim_mat.float()
        return ii_neighbor_mat.long(), ii_constraint_mat.float()

    # def getUserNegItems(self, users):
    #     negItems = []
    #     for user in users:
    #         negItems.append(self.allNeg[user])
    #     return negItems
