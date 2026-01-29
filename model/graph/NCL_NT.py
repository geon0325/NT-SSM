import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise, next_batch_pairwise_bidir
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
import os
import faiss
import pickle as pkl
# paper: Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning. WWW'22


class NCL_NT(GraphRecommender):
    def __init__(self, conf, training_set, valid_set, test_set):
        super(NCL_NT, self).__init__(conf, training_set, valid_set, test_set)
        self.n_layers = conf.n_layer
        self.ssl_temp = conf.tau
        self.ssl_reg = conf.ssl_reg
        self.hyper_layers = conf.hyper_layers
        self.alpha = conf.alpha
        self.proto_reg = conf.proto_reg
        self.k = conf.num_clusters
        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers)
        self.user_centroids = None
        self.user_2cluster = None
        self.item_centroids = None
        self.item_2cluster = None

        self.loss_type = conf.loss_type
        self.tau = conf.tau2
        self.gamma = conf.gamma

        self.alpha_uu = conf.alpha_uu
        self.alpha_ii = conf.alpha_ii
        self.alpha_ui = conf.alpha_ui
        self.alpha_iu = conf.alpha_iu

        self.config_name = f'{conf.dataset}_{conf.model_name}_lr{conf.learning_rate}_reg{conf.reg_lambda}_dim{self.emb_size}_ssl{self.ssl_reg}_prt{self.proto_reg}_tau{self.ssl_temp}_alpha{self.alpha}_k{self.k}_nl{self.n_layers}_uu{self.alpha_uu:.2f}_ii{self.alpha_ii:.2f}_ui{self.alpha_ui:.2f}_iu{self.alpha_iu:.2f}'

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

    def e_step(self):
        user_embeddings = self.model.embedding_dict['user_emb'].detach().cpu().numpy()
        item_embeddings = self.model.embedding_dict['item_emb'].detach().cpu().numpy()
        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings)
        self.item_centroids, self.item_2cluster = self.run_kmeans(item_embeddings)

    def run_kmeans(self, x):
        """Run K-means algorithm to get k clusters of the input tensor x        """
        kmeans = faiss.Kmeans(d=self.emb_size, k=self.k, gpu=True)
        kmeans.train(x)
        cluster_cents = kmeans.centroids
        _, I = kmeans.index.search(x, 1)
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(cluster_cents).cuda()
        node2cluster = torch.LongTensor(I).squeeze().cuda()
        return centroids, node2cluster

    def ProtoNCE_loss(self, initial_emb, user_idx, item_idx):
        user_emb, item_emb = torch.split(initial_emb, [self.data.user_num, self.data.item_num])
        user2cluster = self.user_2cluster[user_idx]
        user2centroids = self.user_centroids[user2cluster]
        proto_nce_loss_user = InfoNCE(user_emb[user_idx],user2centroids,self.ssl_temp) * self.batch_size
        item2cluster = self.item_2cluster[item_idx]
        item2centroids = self.item_centroids[item2cluster]
        proto_nce_loss_item = InfoNCE(item_emb[item_idx],item2centroids,self.ssl_temp) * self.batch_size
        proto_nce_loss = self.proto_reg * (proto_nce_loss_user + proto_nce_loss_item)
        return proto_nce_loss

    def ssl_layer_loss(self, context_emb, initial_emb, user, item):
        context_user_emb_all, context_item_emb_all = torch.split(context_emb, [self.data.user_num, self.data.item_num])
        initial_user_emb_all, initial_item_emb_all = torch.split(initial_emb, [self.data.user_num, self.data.item_num])
        context_user_emb = context_user_emb_all[user]
        initial_user_emb = initial_user_emb_all[user]
        norm_user_emb1 = F.normalize(context_user_emb)
        norm_user_emb2 = F.normalize(initial_user_emb)
        norm_all_user_emb = F.normalize(initial_user_emb_all)
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)
        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        context_item_emb = context_item_emb_all[item]
        initial_item_emb = initial_item_emb_all[item]
        norm_item_emb1 = F.normalize(context_item_emb)
        norm_item_emb2 = F.normalize(initial_item_emb)
        norm_all_item_emb = F.normalize(initial_item_emb_all)
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        ssl_loss = self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
        return ssl_loss

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        
        best_valid, patience, wait_cnt = -1e10, 10, 0
        
        for epoch in range(self.maxEpoch):
            if epoch >= 20:
                self.e_step()
                
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
                    
                model.train()

                user_even_rec, user_odd_rec, item_even_rec, item_odd_rec, emb_list = model()

                pos_user_even = user_even_rec[user_idx]
                pos_user_odd = user_odd_rec[user_idx]
                if neg_user_idx is not None:
                    neg_user_even = user_even_rec[neg_user_idx]
                    neg_user_odd = user_odd_rec[neg_user_idx]
                item_even_pos = item_even_rec[pos_idx]
                item_even_neg = item_even_rec[neg_idx]
                item_odd_pos = item_odd_rec[pos_idx]
                item_odd_neg = item_odd_rec[neg_idx]
                
                user_emb = pos_user_even + pos_user_odd
                pos_item_emb = item_even_pos + item_odd_pos
                neg_item_emb = item_even_neg + item_odd_neg
                
                #rec_user_emb, rec_item_emb, emb_list  = model()
                #user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]

                if self.loss_type == 'bpr':
                    loss_1 = self.bpr_loss(pos_user_even, pos_user_odd, item_even_pos, item_odd_pos, item_even_neg, item_odd_neg,
                                           self.alpha_iu, self.alpha_ii)
                    loss_2 = self.bpr_loss(item_even_pos, item_odd_pos, pos_user_even, pos_user_odd, neg_user_even, neg_user_odd,
                                           self.alpha_uu, self.alpha_ui)
                    rec_loss = (loss_1 + loss_2) / 2
                elif self.loss_type == 'ssm':
                    loss_1 = self.ssm_loss(pos_user_even, pos_user_odd, item_even_pos, item_odd_pos,
                                           self.alpha_iu, self.alpha_ii, self.tau)
                    loss_2 = self.ssm_loss(item_even_pos, item_odd_pos, pos_user_even, pos_user_odd, 
                                           self.alpha_uu, self.alpha_ui, self.tau)
                    rec_loss = (loss_1 + loss_2) / 2
                elif self.loss_type == 'directau':
                    rec_loss = self.direct_au_loss(pos_user_even, pos_user_odd, item_even_pos, item_odd_pos,
                                               self.alpha_uu, self.alpha_ii, self.alpha_ui, self.alpha_iu)
                    
                #rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)

                
                initial_emb = emb_list[0]
                context_emb = emb_list[self.hyper_layers*2]
                ssl_loss = self.ssl_layer_loss(context_emb,initial_emb,user_idx,pos_idx)
                warm_up_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb)/self.batch_size  + ssl_loss

                if epoch<20: #warm_up
                    optimizer.zero_grad()
                    warm_up_loss.backward()
                    optimizer.step()
                    if n % 100 == 0 and n > 0:
                        print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'ssl_loss', ssl_loss.item())
                else:
                    # Backward and optimize
                    proto_loss = self.ProtoNCE_loss(initial_emb, user_idx, pos_idx)
                    batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb) / self.batch_size + ssl_loss + proto_loss
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                    if n % 100 == 0 and n > 0:
                        print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'ssl_loss', ssl_loss.item(), 'proto_loss', proto_loss.item())
            model.eval()
            with torch.no_grad():
                user_even_rec, user_odd_rec, item_even_rec, item_odd_rec, emb_list = model()
                self.user_emb = user_even_rec + user_odd_rec
                self.item_emb = item_even_rec + item_odd_rec
                #self.user_emb, self.item_emb, _ = model()
            #self.fast_evaluation(epoch)
            
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


    def alignment(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def uniformity(self, x, alpha=1.0, t=2):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul((-t) * alpha).exp().mean().log()

    def direct_au_loss(self, 
                       user_even, user_odd, item_even, item_odd, 
                       alpha_uu, alpha_ii, alpha_ui, alpha_iu):
        user_emb = user_even + user_odd
        item_emb = item_even + item_odd
        
        align = self.alignment(user_emb, item_emb)
        
        uniform_uu = (self.uniformity(user_even, alpha_uu) + self.uniformity(item_odd, alpha_uu)) / 2
        uniform_ii = (self.uniformity(user_odd, alpha_ii) + self.uniformity(item_even, alpha_ii)) / 2
        uniform_ui = (self.uniformity(user_even, alpha_ui) + self.uniformity(item_even, alpha_ui)) / 2
        uniform_iu = (self.uniformity(user_odd, alpha_iu) + self.uniformity(item_odd, alpha_iu)) / 2
        uniform = uniform_uu + uniform_ii + uniform_ui + uniform_iu
        return align + uniform
        
    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb, _ = self.model()

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
        lgcn_all_embeddings = torch.stack(all_embeddings, dim=1)
        #all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = lgcn_all_embeddings[:self.data.user_num]
        item_all_embeddings = lgcn_all_embeddings[self.data.user_num:]

        user_even_embeddings = torch.sum(torch.stack([user_all_embeddings[:,layer] for layer in range(0, self.layers+1, 2)]), dim=0)
        user_odd_embeddings = torch.sum(torch.stack([user_all_embeddings[:,layer] for layer in range(1, self.layers+1, 2)]), dim=0)
    
        item_even_embeddings = torch.sum(torch.stack([item_all_embeddings[:,layer] for layer in range(0, self.layers+1, 2)]), dim=0)
        item_odd_embeddings = torch.sum(torch.stack([item_all_embeddings[:,layer] for layer in range(1, self.layers+1, 2)]), dim=0)

        user_even_embeddings = user_even_embeddings / (self.layers + 1)
        user_odd_embeddings = user_odd_embeddings / (self.layers + 1)
        item_even_embeddings = item_even_embeddings / (self.layers + 1)
        item_odd_embeddings = item_odd_embeddings / (self.layers + 1)
        
        return user_even_embeddings, user_odd_embeddings, item_even_embeddings, item_odd_embeddings, all_embeddings

