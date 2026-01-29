import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss,l2_reg_loss
import pickle as pkl
import os
# paper: LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR'20


class LightGCN(GraphRecommender):
    def __init__(self, conf, training_set, valid_set, test_set):
        super(LightGCN, self).__init__(conf, training_set, valid_set, test_set)
        self.n_layers = conf.n_layer
        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers)

        self.loss_type = conf.loss_type
        self.tau = conf.tau
        self.gamma = conf.gamma
        
        self.config_name = f'{conf.dataset}_{conf.model_name}_lr{conf.learning_rate}_reg{conf.reg_lambda}_dim{self.emb_size}_nl{self.n_layers}'

        if self.loss_type == 'bpr':
            self.config_name += f'_loss-bpr'
        elif self.loss_type == 'ssm':
            self.config_name += f'_loss-ssm_tau{self.tau}'
        elif self.loss_type == 'directau':
            self.config_name += f'_loss-directau_gamma{self.gamma}'

        self.config_name += f'_seed{conf.seed}'
            
        print()
        print(self.config_name)

        if os.path.exists(f'embs/{self.config_name}.pkl') or os.path.exists(f'/data/geon/PT-GCF/embs/{self.config_name}.pkl'):
            print('Exists!')
            exit(0)
    

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        
        best_valid, patience, wait_cnt = -1e10, 10, 0
        
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]

                if self.loss_type == 'bpr':
                    loss = self.bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                elif self.loss_type == 'ssm':
                    loss = self.ssm_loss(user_emb, pos_item_emb, self.tau)
                    #loss_temp = self.ssm_loss_v2(user_emb, pos_item_emb, self.tau)
                    #print(loss.item(), loss_temp.item())
                elif self.loss_type == 'directau':
                    loss = self.direct_au_loss(user_emb, pos_item_emb, self.gamma)

                reg_loss = l2_reg_loss(self.reg, model.embedding_dict['user_emb'][user_idx],model.embedding_dict['item_emb'][pos_idx],model.embedding_dict['item_emb'][neg_idx])/self.batch_size
                
                batch_loss = loss + reg_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item(), 'reg:', reg_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = model()
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

        with open(f'/data/geon/PT-GCF/embs/{self.config_name}.pkl', 'wb') as f:
            pkl.dump([self.user_emb, self.item_emb], f)

    def bpr_loss(self, user, item_pos, item_neg):
        pos = torch.mul(user, item_pos).sum(dim=1)
        neg = torch.mul(user, item_neg).sum(dim=1)
        loss = -torch.log(10e-6 + torch.sigmoid(pos - neg))
        return torch.mean(loss)

    def alignment(self,x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def uniformity(self,x, t=2):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def direct_au_loss(self, user_emb, item_emb, gamma):
        align = self.alignment(user_emb, item_emb)
        uniform = gamma * (self.uniformity(user_emb) + self.uniformity(item_emb)) / 2
        return align + uniform

    def ssm_loss(self, user_emb, pos_item_emb, tau=0.2):
        u_emb = F.normalize(user_emb, dim = -1)
        pos_emb = F.normalize(pos_item_emb, dim = -1)

        ratings = torch.matmul(u_emb.unsqueeze(1), pos_emb.T).squeeze(dim=1) # BXB
        pos_ratings = torch.diag(ratings)

        numerator = torch.exp(pos_ratings / tau)
        denominator = torch.exp(ratings / tau).sum(dim=-1)
        loss_r = torch.mean(-torch.log(numerator/denominator))
        return loss_r

    def ssm_loss_v2(self,
        user_emb, item_emb, tau=0.2
    ):
        user_emb = F.normalize(user_emb, dim = -1)
        item_emb = F.normalize(item_emb, dim = -1)
        
        B = user_emb.size(0)
    
        ratings = torch.matmul(user_emb.unsqueeze(1), item_emb.T).squeeze(dim=1) # BXB
        pos_ratings = torch.diag(ratings)
        
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
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings


