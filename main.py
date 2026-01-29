from SELFRec import SELFRec
from util.conf import ModelConf
import time
import argparse
import os
import random

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='yelp2018')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--output_path', type=str, default='results')
    parser.add_argument('--model_name', type=str, default='SimGCL')
    parser.add_argument('--model_type', type=str, default='graph')
    parser.add_argument('--item_ranking', type=str, default='10,20,40')
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--reg_lambda', type=float, default=0.0001)

    parser.add_argument('--seed', type=int, default=2026)
    
    # Common
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--lmbda', type=float, default=0.5)
    parser.add_argument('--tau', type=float, default=0.2)
    parser.add_argument('--tau2', type=float, default=0.2)
    
    # SGL
    parser.add_argument('--aug_type', type=int, default=1)
    parser.add_argument('--drop_rate', type=float, default=0.5)
    
    # SimGCL
    parser.add_argument('--eps', type=float, default=0.1)
    
    # XSimGCL
    parser.add_argument('--target_layer', type=int, default=1)
    
    # NCL
    parser.add_argument('--ssl_reg', type=float, default=1e-6)
    parser.add_argument('--proto_reg', type=float, default=1e-7)
    parser.add_argument('--hyper_layers', type=int, default=1)
    parser.add_argument('--num_clusters', type=int, default=2000)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    
    parser.add_argument('--lmbda_user_item', type=float, default=0.5)
    parser.add_argument('--lmbda_item', type=float, default=0.5)
    parser.add_argument('--lmbda_user', type=float, default=0.5)
    
    parser.add_argument('--lmbda_ui', type=float, default=0.5)
    parser.add_argument('--lmbda_iu', type=float, default=0.5)
    parser.add_argument('--lmbda_ii', type=float, default=0.5)
    parser.add_argument('--lmbda_uu', type=float, default=0.5)
    parser.add_argument('--ego', type=int, default=1)
    parser.add_argument('--cos', type=int, default=1)
    parser.add_argument('--bpr', type=int, default=1)
    
    parser.add_argument('--lmbda_1', type=float, default=0.5)
    parser.add_argument('--lmbda_2', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--num_neg', type=int, default=1)
    
    parser.add_argument('--left_norm', type=float, default=0.5)
    parser.add_argument('--right_norm', type=float, default=0.5)

    parser.add_argument('--loss_type', type=str, default='bpr')

    parser.add_argument('--r_ui', type=float, default=0.01)
    parser.add_argument('--r_iu', type=float, default=0.01)
    parser.add_argument('--r_ii', type=float, default=0.01)
    parser.add_argument('--r_uu', type=float, default=0.01)

    parser.add_argument('--pt_layer', type=int, default=1)

    parser.add_argument('--pt_path', type=str, default='')

    parser.add_argument('--alpha_ui', type=float, default=0.01)
    parser.add_argument('--beta_ui', type=float, default=0.01)
    parser.add_argument('--alpha_iu', type=float, default=0.01)
    parser.add_argument('--beta_iu', type=float, default=0.01)
    parser.add_argument('--alpha_uu', type=float, default=0.01)
    parser.add_argument('--beta_uu', type=float, default=0.01)
    parser.add_argument('--alpha_ii', type=float, default=0.01)
    parser.add_argument('--beta_ii', type=float, default=0.01)
    
    args = parser.parse_args()

    random.seed(args.seed)

    # GPU 
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('embs', exist_ok=True)
    
    # Data path
    args.training_set = f'./dataset/{args.dataset}/train.txt'
    args.valid_set = f'./dataset/{args.dataset}/valid.txt'
    args.test_set = f'./dataset/{args.dataset}/test.txt'
    
    s = time.time()
    rec = SELFRec(args)
    rec.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
