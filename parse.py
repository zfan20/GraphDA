import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="GraphRec")
    parser.add_argument('--bpr_batch', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--model_name', type=str, default='LightGCN',
                        help="model names")
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int,default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int,default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float,default=1.0,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--data_name', type=str,default='Office_Products',
                        help="available datasets: [lastfm, gowalla, yelp2018, amazon-book]")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[20]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int,default=1,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=1000)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')

    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")

    #ultragcn specific
    parser.add_argument("--w1", type=float, default=1e-7, help="w1")
    parser.add_argument("--w2", type=float, default=1, help="w2")
    parser.add_argument("--w3", type=float, default=1e-7, help="w3")
    parser.add_argument("--w4", type=float, default=1, help="w4")
    parser.add_argument("--lambda_Iweight", type=float, default=1.0, help="lambdaiweight")
    parser.add_argument("--negative_weight", type=int, default=100, help="negweight")
    parser.add_argument("--negative_num", type=int, default=100, help="negnum")
    parser.add_argument("--ii_neighbor_num", type=int, default=10, help="ii_neighbor_num")


    #distill K and threshold
    parser.add_argument("--distill_userK", type=int, default=5, help="distill K")
    parser.add_argument("--distill_uuK", type=int, default=5, help="distill K")
    parser.add_argument("--distill_itemK", type=int, default=5, help="distill K")
    parser.add_argument("--distill_iiK", type=int, default=5, help="distill K")
    parser.add_argument("--distill_layers", type=int, default=5, help="distill number of layers")
    parser.add_argument("--distill_thres", type=float, default=-1, help="distill threshold")
    parser.add_argument("--uuii_thres", type=float, default=-1, help="distill threshold")
    parser.add_argument("--uu_lambda", type=float, default=100, help="lambda for ease")
    parser.add_argument("--ii_lambda", type=float, default=200, help="lambda for ease")


    #GTN model
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--alpha1', type=float, default=0.25)
    parser.add_argument('--alpha2', type=float, default=0.25)
    parser.add_argument('--lambda2', type=float, default=4.0) #2, 3, 4,...
    parser.add_argument('--gcn_model', type=str,
                        default='GTN', help='GTN')
    parser.add_argument('--prop_dropout', type=float, default=0.1)
    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--ogb', type=bool, default=True)
    parser.add_argument('--incnorm_para', type=bool, default=True)


    return parser.parse_args()
