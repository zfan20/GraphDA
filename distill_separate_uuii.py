import utils
import torch
import numpy as np
import dataloader
import time
from parse import parse_args
import multiprocessing
import os
from os.path import join
from model import LightGCN, PureMF, UltraGCN, SGCN
from trainers import GraphRecTrainer
from utils import EarlyStopping


def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

ROOT_PATH = "./"
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')
import sys
sys.path.append(join(CODE_PATH, 'sources'))


if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)

args.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
args.cores = multiprocessing.cpu_count() // 2

utils.set_seed(args.seed)

# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)



#Recmodel = register.MODELS[world.model_name](world.config, dataset)
dataset = dataloader.Loader(args)
if args.model_name == 'LightGCN':
    model = LightGCN(args, dataset)
elif args.model_name == 'UltraGCN':
    constraint_mat = dataset.getConstraintMat()
    ii_neighbor_mat, ii_constraint_mat = dataset.get_ii_constraint_mat()
    model = UltraGCN(args, dataset, constraint_mat, ii_constraint_mat, ii_neighbor_mat)
elif args.model_name == 'SGCN':
    model = SGCN(args, dataset)
else:
    model = PureMF(args, dataset)
model = model.to(args.device)
trainer = GraphRecTrainer(model, dataset, args)

checkpoint_path = utils.getFileName("./checkpoints/", args)
print(f"load and save to {checkpoint_path}")

if args.do_eval:
    if args.data_name == 'Office_Products':
        checkpoint_path = './checkpoints/Office_Products-LightGCN-Office_Products-4-0.001-0.0001-1000-1-0.7.pth.tar'
    print(f'Load model from {checkpoint_path} for test!')
    trainer.load(checkpoint_path)

    [[distill_user_row, distill_item_col, distill_value], [distill_uu_row, distill_uu_col, distill_uu_value], [distill_ii_row, distill_ii_col, distill_ii_value]] = trainer.generateKorderGraph_UUII(userK=args.distill_userK, itemK=args.distill_itemK, uuK=args.distill_uuK, iiK=args.distill_iiK, threshold=args.distill_thres, uuii_thres=args.uuii_thres)
    dataset.reset_graph_uuii([[distill_user_row, distill_item_col, distill_value], [distill_uu_row, distill_uu_col, distill_uu_value], [distill_ii_row, distill_ii_col, distill_ii_value]])
    model.dataset = dataset
    trainer.dataset = dataset
    checkpoint_path = utils.getDistillUUIIFileName("./checkpoints_distill/", args)
    trainer.model.load_state_dict(torch.load(checkpoint_path))
    model.n_layers = args.distill_layers
    model.dataset = dataset
    model.reset_graph()
    print(f'Load model from {checkpoint_path} for test!')
    #scores, result_info, _ = trainer.test(0, full_sort=True)
    scores, result_info, _ = trainer.complicated_eval()

else:

    if args.data_name == 'Office_Products':
        checkpoint_path = './checkpoints/Office_Products-LightGCN-Office_Products-4-0.001-0.0001-1000-1-0.7.pth.tar'
    elif args.data_name == 'Beauty':
        checkpoint_path = './checkpoints/Beauty-LightGCN-Beauty-2-0.001-0.0001-1000-1-0.7.pth.tar'
    elif args.data_name == 'Tools_and_Home_Improvement':
        checkpoint_path = './checkpoints/Tools_and_Home_Improvement-LightGCN-Tools_and_Home_Improvement-4-0.001-0.0001-1000-1-0.3.pth.tar'
    elif args.data_name == 'Toys_and_Games':
        checkpoint_path = './checkpoints/Toys_and_Games-LightGCN-Toys_and_Games-4-0.001-0.0001-1000-1-0.7.pth.tar'
    print(f'Load model from {checkpoint_path} for test!')
    trainer.load(checkpoint_path)

    [[distill_user_row, distill_item_col, distill_value], [distill_uu_row, distill_uu_col, distill_uu_value], [distill_ii_row, distill_ii_col, distill_ii_value]] = trainer.generateKorderGraph_UUII(userK=args.distill_userK, itemK=args.distill_itemK, uuK=args.distill_uuK, iiK=args.distill_iiK, threshold=args.distill_thres, uuii_thres=args.uuii_thres)
    dataset.reset_graph_uuii([[distill_user_row, distill_item_col, distill_value], [distill_uu_row, distill_uu_col, distill_uu_value], [distill_ii_row, distill_ii_col, distill_ii_value]])
    model.dataset = dataset
    model.reset_all_uuii()
    model.n_layers = args.distill_layers
    trainer.optim = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer.dataset = dataset
    checkpoint_path = utils.getDistillUUIIFileName("./checkpoints_distill/", args)
    early_stopping = EarlyStopping(checkpoint_path, patience=10, verbose=True)
    for epoch in range(args.epochs):
        trainer.train(epoch)
        if (epoch+1) %10==0:
            scores, _, _ = trainer.valid(epoch, full_sort=True)
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    print('---------------Change to Final testing!-------------------')
    # load the best model
    trainer.model.load_state_dict(torch.load(checkpoint_path))
    valid_scores, _, _ = trainer.valid('best', full_sort=True)
    #trainer.args.train_matrix = test_rating_matrix
    scores, result_info, _ = trainer.test('best', full_sort=True)
    scores, result_info, _ = trainer.complicated_eval()
