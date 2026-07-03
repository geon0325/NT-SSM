import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise, next_batch_pairwise_bidir
from base.torch_interface import TorchGraphInterface
from util.loss_torch import l2_reg_loss
import pickle as pkl
import os
# paper: LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR'20


class LightGCN_NT(GraphRecommender):
    def __init__(self, conf, training_set, valid_set, test_set):
        super(LightGCN_NT, self).__init__(conf, training_set, valid_set, test_set)
        self.n_layers = conf.n_layer
        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers)

        self.loss_type = conf.loss_type
        
        self.alpha_uu = conf.alpha_uu
        self.alpha_ii = conf.alpha_ii
        self.alpha_ui = conf.alpha_ui
        self.alpha_iu = conf.alpha_iu

        self.tau = conf.tau
        
        self.config_name = f'{conf.dataset}_{conf.model_name}_lr{conf.learning_rate}_reg{conf.reg_lambda}_dim{self.emb_size}_nl{self.n_layers}_uu{self.alpha_uu:.2f}_ii{self.alpha_ii:.2f}_ui{self.alpha_ui:.2f}_iu{self.alpha_iu:.2f}'

        if self.loss_type == 'bpr':
            self.config_name += f'_loss-bpr'
        elif self.loss_type == 'ssm':
            self.config_name += f'_loss-ssm_tau{self.tau}'

        self.config_name += f'_seed{conf.seed}'
            
        print()
        print(self.config_name)
    
        if os.path.exists(f'embs/{self.config_name}.pkl'):
            print('Exists!')
            exit(0)
            
    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        
        best_valid, patience, wait_cnt = -1e10, 10, 0
        
        for epoch in range(self.maxEpoch):
            if self.loss_type == 'bpr':
                batches = next_batch_pairwise_bidir(self.data, self.batch_size)
            else:
                batches = next_batch_pairwise(self.data, self.batch_size)
                
            for n, batch in enumerate(batches):
                if self.loss_type == 'bpr':
                    user_idx, neg_user_idx, pos_idx, neg_idx = batch
                else:
                    user_idx, pos_idx, neg_idx = batch
                    neg_user_idx = None
                user_even_rec, user_odd_rec, item_even_rec, item_odd_rec = model()

                pos_user_even = user_even_rec[user_idx]
                pos_user_odd = user_odd_rec[user_idx]
                if neg_user_idx is not None:
                    neg_user_even = user_even_rec[neg_user_idx]
                    neg_user_odd = user_odd_rec[neg_user_idx]
                item_even_pos = item_even_rec[pos_idx]
                item_even_neg = item_even_rec[neg_idx]
                item_odd_pos = item_odd_rec[pos_idx]
                item_odd_neg = item_odd_rec[neg_idx]

                if self.loss_type == 'bpr':
                    loss_1 = self.bpr_loss(pos_user_even, pos_user_odd, item_even_pos, item_odd_pos, item_even_neg, item_odd_neg,
                                           self.alpha_iu, self.alpha_ii)
                    loss_2 = self.bpr_loss(item_even_pos, item_odd_pos, pos_user_even, pos_user_odd, neg_user_even, neg_user_odd,
                                           self.alpha_uu, self.alpha_ui)
                    loss = (loss_1 + loss_2) / 2
                elif self.loss_type == 'ssm':
                    loss_1 = self.ssm_loss(pos_user_even, pos_user_odd, item_even_pos, item_odd_pos,
                                           self.alpha_iu, self.alpha_ii, self.tau)
                    loss_2 = self.ssm_loss(item_even_pos, item_odd_pos, pos_user_even, pos_user_odd, 
                                           self.alpha_uu, self.alpha_ui, self.tau)
                    loss = (loss_1 + loss_2) / 2

                batch_loss = loss + l2_reg_loss(self.reg, model.embedding_dict['user_emb'][user_idx],model.embedding_dict['item_emb'][pos_idx],model.embedding_dict['item_emb'][neg_idx])/self.batch_size
            
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            
            with torch.no_grad():
                user_even_rec, user_odd_rec, item_even_rec, item_odd_rec = model()
                self.user_emb, self.item_emb = user_even_rec + user_odd_rec, item_even_rec + item_odd_rec
            self.evaluate(self.test('valid'), 'valid')
            result_valid = [r[:-1] for r in self.result]
            self.evaluate(self.test('test'), 'test')
            result_test = [r[:-1] for r in self.result]
            
            for _i in range(3):
                print('Valid\t', result_valid[_i*5], result_valid[_i*5+3], result_valid[_i*5+4])
            for _i in range(3):
                print('Test\t', result_test[_i*5], result_test[_i*5+3], result_test[_i*5+4])
                
            with open(f'logs/{self.config_name}.txt', 'a') as f:
                valid_log, test_log = '', ''
                for _i in range(3):
                    recall = result_valid[_i*5+3].split(':')[1]
                    ndcg = result_valid[_i*5+4].split(':')[1]
                    valid_log += f',{recall},{ndcg}'
                    
                    recall = result_test[_i*5+3].split(':')[1]
                    ndcg = result_test[_i*5+4].split(':')[1]
                    test_log += f',{recall},{ndcg}'
                f.write(f'{epoch+1},valid,{valid_log}\n')
                f.write(f'{epoch+1},test,{test_log}\n')
                
            ndcg_valid = float(result_valid[9].split(':')[1])
            if ndcg_valid > best_valid:
                best_valid = ndcg_valid
                self.best_user_emb = self.model.embedding_dict['user_emb'].detach().cpu()
                self.best_item_emb = self.model.embedding_dict['item_emb'].detach().cpu()
                wait_cnt = 0
            else:
                wait_cnt += 1
                print(f'Patience... {wait_cnt}/{patience}')
                
            if wait_cnt == patience:
                print('Early Stopping!')
                break
            print()
                
            
        self.user_emb = self.best_user_emb.detach().cpu()
        self.item_emb = self.best_item_emb.detach().cpu()
        
        with open(f'embs/{self.config_name}.pkl', 'wb') as f:
            pkl.dump([self.user_emb, self.item_emb], f)

    def bpr_loss(self, user_even, user_odd, item_even_pos, item_odd_pos, item_even_neg, item_odd_neg,
                 alpha_u, alpha_i):
        pos_uu = torch.mul(user_even, item_odd_pos).sum(dim=1)
        pos_ii = torch.mul(user_odd, item_even_pos).sum(dim=1)
        pos_ui = torch.mul(user_even, item_even_pos).sum(dim=1)
        pos_iu = torch.mul(user_odd, item_odd_pos).sum(dim=1)

        neg_uu = torch.mul(user_even, item_odd_neg).sum(dim=1)
        neg_ii = torch.mul(user_odd, item_even_neg).sum(dim=1)
        neg_ui = torch.mul(user_even, item_even_neg).sum(dim=1)
        neg_iu = torch.mul(user_odd, item_odd_neg).sum(dim=1)

        delta_uu = pos_uu - alpha_u * neg_uu
        delta_ii = pos_ii - alpha_i * neg_ii
        delta_ui = pos_ui - alpha_i * neg_ui
        delta_iu = pos_iu - alpha_u * neg_iu
        
        loss = -torch.log(10e-6 + torch.sigmoid(delta_uu + delta_ii + delta_ui + delta_iu))
        return torch.mean(loss)

    def ssm_loss(self,
        user_even, user_odd, item_even, item_odd,
        alpha_u, alpha_i, tau=0.2
    ):
        B = user_even.size(0)
    
        # global cosine vectors
        user_sum = F.normalize(user_even + user_odd, dim=-1, eps=1e-8)
        item_sum = F.normalize(item_even + item_odd, dim=-1, eps=1e-8)

        ratings = torch.matmul(user_sum.unsqueeze(1), item_sum.T).squeeze(dim=1) # BXB
        pos_ratings = torch.diag(ratings)
    
        # bilinear terms
        uu = torch.matmul(user_even, item_odd.T)
        ii = torch.matmul(user_odd,  item_even.T)
        ui = torch.matmul(user_even, item_even.T)
        iu = torch.matmul(user_odd,  item_odd.T)
    
        # shared denominator
        den = (
            (user_even + user_odd).norm(dim=1).unsqueeze(1)
            * (item_even + item_odd).norm(dim=1).unsqueeze(0)
            + 1e-8
        )
    
        C_uu = uu / den
        C_ii = ii / den
        C_ui = ui / den
        C_iu = iu / den
    
        pos_mask = torch.eye(B, device=user_even.device)
        neg_mask = 1.0 - pos_mask
    
        ratings = (
            (C_uu + C_ii + C_ui + C_iu) * pos_mask
            + (alpha_u * C_uu
             + alpha_i * C_ii
             + alpha_i * C_ui
             + alpha_u * C_iu) * neg_mask
        )
        
        logits = ratings / tau
        logits = logits - logits.max(dim=1, keepdim=True)[0]
        
        loss = -torch.mean(
            torch.diag(logits) - torch.logsumexp(logits, dim=1)
        )
        return loss

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        #all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]

        user_even_embeddings = torch.sum(torch.stack([user_all_embeddings[:,layer] for layer in range(0, self.layers+1, 2)]), dim=0)
        user_odd_embeddings = torch.sum(torch.stack([user_all_embeddings[:,layer] for layer in range(1, self.layers+1, 2)]), dim=0)
    
        item_even_embeddings = torch.sum(torch.stack([item_all_embeddings[:,layer] for layer in range(0, self.layers+1, 2)]), dim=0)
        item_odd_embeddings = torch.sum(torch.stack([item_all_embeddings[:,layer] for layer in range(1, self.layers+1, 2)]), dim=0)

        user_even_embeddings = user_even_embeddings / (self.layers + 1)
        user_odd_embeddings = user_odd_embeddings / (self.layers + 1)
        item_even_embeddings = item_even_embeddings / (self.layers + 1)
        item_odd_embeddings = item_odd_embeddings / (self.layers + 1)
        
        return user_even_embeddings, user_odd_embeddings, item_even_embeddings, item_odd_embeddings

