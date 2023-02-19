import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import copy
import model
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from torch.optim import Adam
from utils import recall_at_k, ndcg_k, cal_mrr, get_user_performance_perpopularity, get_item_performance_perpopularity


class Trainer:
    def __init__(self, model, dataset, args):

        self.args = args

        self.model = model
        
        self.dataset = dataset

        # self.data_name = self.args.data_name
        #betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr) #, betas=betas, weight_decay=self.args.decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]), flush=True)
        self.pred_mask_mat = self.dataset.valid_pred_mask_mat

    def train(self, epoch):
        self.iteration(epoch)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, full_sort, mode='valid')

    def test(self, epoch, full_sort=False):
        self.pred_mask_mat = self.dataset.test_pred_mask_mat
        return self.iteration(epoch, full_sort, mode='test')

    def iteration(self, epoch, full_sort=False, mode='train'):
        raise NotImplementedError

    def complicated_eval(self):
        self.pred_mask_mat = self.dataset.test_pred_mask_mat
        return self.eval_analysis()

    def eval_analysis(self):
        raise NotImplementedError

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg, mrr = [], [], 0
        recall_dict_list = []
        ndcg_dict_list = []
        for k in [1, 5, 10, 15, 20, 40]:
            recall_result, recall_dict_k = recall_at_k(answers, pred_list, k)
            recall.append(recall_result)
            recall_dict_list.append(recall_dict_k)
            ndcg_result, ndcg_dict_k = ndcg_k(answers, pred_list, k)
            ndcg.append(ndcg_result)
            ndcg_dict_list.append(ndcg_dict_k)
        mrr, mrr_dict = cal_mrr(answers, pred_list)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.8f}'.format(recall[0]), "NDCG@1": '{:.8f}'.format(ndcg[0]),
            "HIT@5": '{:.8f}'.format(recall[1]), "NDCG@5": '{:.8f}'.format(ndcg[1]),
            "HIT@10": '{:.8f}'.format(recall[2]), "NDCG@10": '{:.8f}'.format(ndcg[2]),
            "HIT@15": '{:.8f}'.format(recall[3]), "NDCG@15": '{:.8f}'.format(ndcg[3]),
            "HIT@20": '{:.8f}'.format(recall[4]), "NDCG@20": '{:.8f}'.format(ndcg[4]),
            "HIT@40": '{:.8f}'.format(recall[5]), "NDCG@40": '{:.8f}'.format(ndcg[5]),
            "MRR": '{:.8f}'.format(mrr)
        }
        print(post_fix, flush=True)
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[2], ndcg[2], recall[3], ndcg[3], recall[4], ndcg[4], recall[5], ndcg[5], mrr], str(post_fix), [recall_dict_list, ndcg_dict_list, mrr_dict]

    def get_pos_items_ranks(self, batch_pred_lists, answers):
        num_users = len(batch_pred_lists)
        batch_pos_ranks = defaultdict(list)
        for i in range(num_users):
            pred_list = batch_pred_lists[i]
            true_set = set(answers[i])
            for ind, pred_item in enumerate(pred_list):
                if pred_item in true_set:
                    batch_pos_ranks[pred_item].append(ind+1)
        return batch_pos_ranks

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.args.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name, map_location='cuda:0'))



class GraphRecTrainer(Trainer):

    def __init__(self, model, dataset, args):
        super(GraphRecTrainer, self).__init__(
            model, dataset, args
        )

    def iteration(self, epoch, full_sort=False, mode='train'):

        if mode=='train':
            self.model.train()
            rec_avg_loss = 0.0
            avg_norm_loss = 0.0

            with timer(name="Sample"):
                S = utils.UniformSample_original(self.dataset)
            users = torch.Tensor(S[:, 0]).long()
            posItems = torch.Tensor(S[:, 1]).long()
            negItems = torch.Tensor(S[:, 2]).long()

            users = users.to(self.args.device)
            posItems = posItems.to(self.args.device)
            negItems = negItems.to(self.args.device)
            users, posItems, negItems = utils.shuffle(users, posItems, negItems)
            total_batch = len(users) // self.args.bpr_batch + 1

            for (batch_i, (batch_users, batch_pos, batch_neg)) in enumerate(utils.minibatch(users, posItems, negItems, batch_size=self.args.bpr_batch)):
                if self.args.model_name == 'UltraGCN':
                    loss = self.model(batch_users.cpu(), batch_pos.cpu(), batch_neg.cpu())
                else:
                    loss, reg_loss = self.model.bpr_loss(batch_users, batch_pos, batch_neg)
                    reg_loss = reg_loss*self.args.decay
                    loss = loss + reg_loss

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                rec_avg_loss += loss.cpu().item()
                if self.args.model_name != 'UltraGCN':
                    avg_norm_loss += reg_loss.cpu().item()
            rec_avg_loss = rec_avg_loss / total_batch
            if self.args.model_name != 'UltraGCN':
                avg_norm_loss = avg_norm_loss / total_batch
            time_info = timer.dict()
            timer.zero()
            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss),
                "avg_norm_loss": '{:.4f}'.format(avg_norm_loss),
                "time_info": time_info
            }
            print(str(post_fix), flush=True)
        else:
            u_batch_size = self.args.testbatch
            if mode == 'valid':
                evalDict = self.dataset.validDict
            else:
                evalDict = self.dataset.testDict
                self.model.graph = self.dataset.getSparseGraph(self.dataset.test_pred_mask_mat)
            self.model.eval()

            pred_list = None

            with torch.no_grad():
                users = list(evalDict.keys())
                try:
                    assert u_batch_size <= len(users) / 10
                except AssertionError:
                    print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")

                users_list = []
                total_batch = len(users) // u_batch_size + 1
                answer_list = None
                i = 0
                for batch_users in utils.minibatch(users, batch_size=u_batch_size):
                    groundTrue = [evalDict[u] for u in batch_users]
                    batch_users_gpu = torch.Tensor(batch_users).long()
                    batch_users_gpu = batch_users_gpu.to(self.args.device)

                    rating_pred = self.model.getUsersRating(batch_users_gpu)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = batch_users_gpu.cpu().numpy()
                    rating_pred[self.pred_mask_mat[batch_user_index].toarray() > 0] = -(1<<10)
                    ind = np.argpartition(rating_pred, -40)[:, -40:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = np.array(groundTrue)
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, np.array(groundTrue), axis=0)
                    i += 1
                return self.get_full_sort_score(epoch, answer_list, pred_list)

    def eval_analysis(self):
        Ks = [1, 5, 10, 15, 20, 40]
        item_freq = defaultdict(int)
        train = defaultdict(list)
        for user_id, item_id in zip(self.dataset.trainUser, self.dataset.trainItem):
            train[user_id].append(item_id)
            item_freq[item_id] += 1
        freq_quantiles = np.array([3, 7, 20, 50])
        items_in_freqintervals = [[] for _ in range(len(freq_quantiles)+1)]
        for item, freq_i in item_freq.items():
            interval_ind = -1
            for quant_ind, quant_freq in enumerate(freq_quantiles):
                if freq_i <= quant_freq:
                    interval_ind = quant_ind
                    break
            if interval_ind == -1:
                interval_ind = len(items_in_freqintervals) - 1
            items_in_freqintervals[interval_ind].append(item)

        u_batch_size = self.args.testbatch
        all_pos_items_ranks = defaultdict(list)
        testDict = self.dataset.testDict
        self.model.eval()

        pred_list = None

        with torch.no_grad():
            users = list(testDict.keys())
            try:
                assert u_batch_size <= len(users) / 10
            except AssertionError:
                print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")

            users_list = []
            total_batch = len(users) // u_batch_size + 1
            answer_list = None
            i = 0
            for batch_users in utils.minibatch(users, batch_size=u_batch_size):
                groundTrue = [testDict[u] for u in batch_users]
                batch_users_gpu = torch.Tensor(batch_users).long()
                batch_users_gpu = batch_users_gpu.to(self.args.device)

                rating_pred = self.model.getUsersRating(batch_users_gpu)

                rating_pred = rating_pred.cpu().data.numpy().copy()
                batch_user_index = batch_users_gpu.cpu().numpy()
                rating_pred[self.pred_mask_mat[batch_user_index].toarray() > 0] = 0
                batch_pred_list = np.argsort(-rating_pred, axis=1)
                pos_items = np.array(groundTrue)

                pos_ranks = np.where(batch_pred_list==pos_items)[1]+1
                for each_pos_item, each_rank in zip(pos_items, pos_ranks):
                    all_pos_items_ranks[each_pos_item[0]].append(each_rank)

                partial_batch_pred_list = batch_pred_list[:, :40]

                if i == 0:
                    pred_list = partial_batch_pred_list
                    answer_list = np.array(groundTrue)
                else:
                    pred_list = np.append(pred_list, partial_batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, np.array(groundTrue), axis=0)
                i += 1
            scores, result_info, [recall_dict_list, ndcg_dict_list, mrr_dict] = self.get_full_sort_score('best', answer_list, pred_list)
            get_user_performance_perpopularity(train, [recall_dict_list, ndcg_dict_list, mrr_dict], Ks)
            get_item_performance_perpopularity(items_in_freqintervals, all_pos_items_ranks, Ks, freq_quantiles, self.dataset.m_items)
            return scores, result_info, None

    def generateKorderGraph(self, userK, itemK, threshold=0.5):
        u_batch_size = self.args.testbatch
        self.model.eval()

        pred_list = None

        assert userK<=50
        assert itemK<=50

        distill_user_row = []
        distill_item_col = []
        distill_value = []

        if userK > 0:
            with torch.no_grad():
                users = list(set(self.dataset.trainUser))
                try:
                    assert u_batch_size <= len(users) / 10
                except AssertionError:
                    print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")

                users_list = []
                total_batch = len(users) // u_batch_size + 1
                for batch_users in utils.minibatch(users, batch_size=u_batch_size):
                    batch_users_gpu = torch.Tensor(batch_users).long()
                    batch_users_gpu = batch_users_gpu.to(self.args.device)

                    rating_pred = self.model.getUsersRating(batch_users_gpu)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    ind = np.argpartition(rating_pred, -50)[:, -50:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    partial_batch_pred_list = batch_pred_list[:, :userK]

                    for batch_i in range(partial_batch_pred_list.shape[0]):
                        uid = batch_users[batch_i]
                        user_pred = partial_batch_pred_list[batch_i]
                        for eachpred in user_pred:
                            distill_user_row.append(uid)
                            distill_item_col.append(eachpred)
                            pred_val = rating_pred[batch_i, eachpred]
                            if threshold > 0:
                                if pred_val > threshold:
                                    distill_value.append(1)
                                else:
                                    distill_value.append(0)
                            else:
                                distill_value.append(pred_val)

        if itemK > 0:
            with torch.no_grad():
                items = [i for i in range(self.dataset.m_item)]
                total_batch = len(items) // u_batch_size + 1
                for batch_items in utils.minibatch(items, batch_size=u_batch_size):
                    batch_items_gpu = torch.Tensor(batch_items).long()
                    batch_items_gpu = batch_items_gpu.to(self.args.device)

                    rating_pred = self.model.getItemsRating(batch_items_gpu)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    ind = np.argpartition(rating_pred, -50)[:, -50:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    partial_batch_pred_list = batch_pred_list[:, :itemK]
                    for batch_i in range(partial_batch_pred_list.shape[0]):
                        iid = batch_items[batch_i]
                        item_pred = partial_batch_pred_list[batch_i]
                        for eachpred in item_pred:
                            distill_user_row.append(eachpred)
                            distill_item_col.append(iid)
                            pred_val = rating_pred[batch_i, eachpred]
                            if threshold > 0:
                                if pred_val > threshold:
                                    distill_value.append(1)
                                else:
                                    distill_value.append(0)
                            else:
                                distill_value.append(pred_val)
        return [distill_user_row, distill_item_col, distill_value]


    def generateKorderGraph_UUII(self, userK, itemK, uuK, iiK, threshold=0.5, uuii_thres=0.5):
        u_batch_size = self.args.testbatch
        self.model.eval()

        pred_list = None

        assert userK<=50
        assert itemK<=50

        distill_user_row = []
        distill_item_col = []
        distill_value = []

        distill_uu_row = []
        distill_uu_col = []
        distill_uu_value = []

        distill_ii_row = []
        distill_ii_col = []
        distill_ii_value = []

        if userK > 0:
            with torch.no_grad():
                users = list(set(self.dataset.trainUser))
                try:
                    assert u_batch_size <= len(users) / 10
                except AssertionError:
                    print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")

                users_list = []
                total_batch = len(users) // u_batch_size + 1
                for batch_users in utils.minibatch(users, batch_size=u_batch_size):
                    batch_users_gpu = torch.Tensor(batch_users).long()
                    batch_users_gpu = batch_users_gpu.to(self.args.device)

                    rating_pred = self.model.getUsersRating(batch_users_gpu)
                    uu_pred = self.model.getUsersUsers(batch_users_gpu)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    ind = np.argpartition(rating_pred, -50)[:, -50:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    uu_pred = uu_pred.cpu().data.numpy().copy()
                    uu_ind = np.argpartition(uu_pred, -50)[:, -50:]
                    uu_arr_ind = uu_pred[np.arange(len(uu_pred))[:, None], uu_ind]
                    uu_arr_ind_argsort = np.argsort(uu_arr_ind)[np.arange(len(uu_pred)), ::-1]
                    uu_batch_pred_list = uu_ind[np.arange(len(uu_pred))[:, None], uu_arr_ind_argsort]

                    partial_batch_pred_list = batch_pred_list[:, :userK]
                    uu_partial_batch_pred_list = uu_batch_pred_list[:, :uuK]

                    for batch_i in range(partial_batch_pred_list.shape[0]):
                        uid = batch_users[batch_i]
                        user_pred = partial_batch_pred_list[batch_i]
                        uu_user_pred = uu_partial_batch_pred_list[batch_i]
                        for eachpred in user_pred:
                            distill_user_row.append(uid)
                            distill_item_col.append(eachpred)
                            pred_val = rating_pred[batch_i, eachpred]
                            if threshold > 0:
                                if pred_val > threshold:
                                    distill_value.append(1)
                                else:
                                    distill_value.append(0)
                            else:
                                distill_value.append(pred_val)

                        for eachpred in uu_user_pred:
                            distill_uu_row.append(uid)
                            distill_uu_col.append(eachpred)
                            distill_uu_row.append(eachpred)
                            distill_uu_col.append(uid)
                            pred_val = uu_pred[batch_i, eachpred]
                            if uuii_thres > 0:
                                if pred_val > uuii_thres:
                                    distill_uu_value.append(1)
                                    distill_uu_value.append(1)
                                else:
                                    distill_uu_value.append(0)
                                    distill_uu_value.append(0)
                            else:
                                distill_uu_value.append(pred_val)
                                distill_uu_value.append(pred_val)

        if itemK > 0:
            with torch.no_grad():
                items = [i for i in range(self.dataset.m_item)]
                total_batch = len(items) // u_batch_size + 1
                for batch_items in utils.minibatch(items, batch_size=u_batch_size):
                    batch_items_gpu = torch.Tensor(batch_items).long()
                    batch_items_gpu = batch_items_gpu.to(self.args.device)

                    rating_pred = self.model.getItemsRating(batch_items_gpu)
                    ii_pred = self.model.getItemsItems(batch_items_gpu)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    ind = np.argpartition(rating_pred, -50)[:, -50:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    ii_pred = ii_pred.cpu().data.numpy().copy()
                    ii_ind = np.argpartition(ii_pred, -50)[:, -50:]
                    ii_arr_ind = ii_pred[np.arange(len(ii_pred))[:, None], ii_ind]
                    ii_arr_ind_argsort = np.argsort(ii_arr_ind)[np.arange(len(ii_pred)), ::-1]
                    ii_batch_pred_list = ii_ind[np.arange(len(ii_pred))[:, None], ii_arr_ind_argsort]

                    partial_batch_pred_list = batch_pred_list[:, :itemK]
                    ii_partial_batch_pred_list = ii_batch_pred_list[:, :iiK]
                    for batch_i in range(partial_batch_pred_list.shape[0]):
                        iid = batch_items[batch_i]
                        item_pred = partial_batch_pred_list[batch_i]
                        ii_item_pred = ii_partial_batch_pred_list[batch_i]
                        for eachpred in item_pred:
                            distill_user_row.append(eachpred)
                            distill_item_col.append(iid)
                            pred_val = rating_pred[batch_i, eachpred]
                            if threshold > 0:
                                if pred_val > threshold:
                                    distill_value.append(1)
                                else:
                                    distill_value.append(0)
                            else:
                                distill_value.append(pred_val)

                        for eachpred in ii_item_pred:
                            distill_ii_row.append(eachpred)
                            distill_ii_col.append(iid)
                            distill_ii_row.append(eachpred)
                            distill_ii_col.append(iid)
                            pred_val = ii_pred[batch_i, eachpred]
                            if uuii_thres > 0:
                                if pred_val > uuii_thres:
                                    distill_ii_value.append(1)
                                    distill_ii_value.append(1)
                                else:
                                    distill_ii_value.append(0)
                                    distill_ii_value.append(0)
                            else:
                                distill_ii_value.append(pred_val)
                                distill_ii_value.append(pred_val)
        return [[distill_user_row, distill_item_col, distill_value], [distill_uu_row, distill_uu_col, distill_uu_value], [distill_ii_row, distill_ii_col, distill_ii_value]]
