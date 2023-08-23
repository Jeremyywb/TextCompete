import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        # mean_embeddings.register_hook(lambda t: print(f'''===========================\n#out = self.pool_ly(hidden_states,inputs['smask']):\n {t}'''))
        mean_embeddings = torch.clamp(mean_embeddings, max=1e8)
        del sum_embeddings,input_mask_expanded
        # input_mask_expanded = attention_mask/attention_mask.sum(1).unsqueeze(1)
        # input_mask_expanded = input_mask_expanded.unsqueeze(-1).expand(last_hidden_state.size()).float()
        # mean_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        return mean_embeddings

class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = -1e9
        max_embeddings, _ = torch.max(embeddings, dim = 1)
        return max_embeddings

class MeanMax(nn.Module):
    def __init__(self):
        super(MeanMax, self).__init__()
        
        self.mean_pooler = MeanPooling()
        self.max_pooler  = MaxPooling()
        
    def forward(self, last_hidden_state, attention_mask):
        mean_pooler = self.mean_pooler( last_hidden_state ,attention_mask )
        max_pooler =  self.max_pooler( last_hidden_state ,attention_mask )
        out = torch.concat([mean_pooler ,max_pooler ] , 1)
        return out
    
class GeMText(nn.Module):
    def __init__(self, dim = 1, p=3, eps=1e-6):
        super(GeMText, self).__init__()
        self.dim = dim
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.feat_mult = 1

    def forward(self, last_hidden_state, attention_mask):
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.shape)
        x = (last_hidden_state.clamp(min=self.eps) * attention_mask_expanded).pow(self.p).sum(self.dim)
        ret = x / attention_mask_expanded.sum(self.dim).clip(min=self.eps)
        ret = ret.pow(1 / self.p)
        return ret
    
class LSTMPooling(nn.Module):
    def __init__(self, num_layers, hidden_size):
        super(LSTMPooling, self).__init__()
        self.num_hidden_layers = num_layers
        self.hidden_size = hidden_size
        self.hiddendim_lstm = 256
        self.lstm = nn.LSTM(self.hidden_size, self.hiddendim_lstm, batch_first=True)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, all_hidden_states):
        hidden_states = torch.cat([
            all_hidden_states[layer_i][:, 0, :] 
                for layer_i in range(1, self.num_hidden_layers+1)],
            dim=-1
        )
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out, _ = self.lstm(hidden_states, None)
        out = self.dropout(out[:, -1, :])
        return out

class AttentionPooling(nn.Module):
    def __init__(self, num_layers, hidden_size):
        super(AttentionPooling, self).__init__()
        self.num_hidden_layers = num_layers
        self.hidden_size = hidden_size
        self.hiddendim_fc = 256
        self.dropout = nn.Dropout(0.1)

        q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.hidden_size))
        self.q = nn.Parameter(torch.from_numpy(q_t)).float()
        w_ht = np.random.normal(loc=0.0, scale=0.1, size=(self.hidden_size, self.hiddendim_fc))
        self.w_h = nn.Parameter(torch.from_numpy(w_ht)).float()

    def forward(self, all_hidden_states):
        hidden_states = torch.cat([
            all_hidden_states[layer_i][:, 0, :] 
                for layer_i in range(1, self.num_hidden_layers+1)],
            dim=-1
        )
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out = self.attention(hidden_states)
        out = self.dropout(out)
        return out

    def attention(self, h):
        v = torch.matmul(self.q.to(device), h.transpose(-2, -1)).squeeze(1)
        v = F.softmax(v, -1)
        v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
        v = torch.matmul(self.w_h.transpose(1, 0).to(device), v_temp).squeeze(2)
        return v


class AttentionHead(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.in_features = in_features
        self.middle_features = hidden_dim
        self.W = nn.Linear(in_features, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        self.out_features = hidden_dim
        
    #========================
    # pack unmachable tensors 
    # tensors = [torch.tensor([1]).expand(5),torch.tensor([1]).expand(3),torch.tensor([1]).expand(4)]
    #========================
    
    def pack_unmachable(self,tensors):
        nt = torch.nested.nested_tensor(tensors)
        tensors = nt.to_padded_tensor(0.0)
        del nt
        return tensors

    #========================
    # document of sentance emb
    #========================
    
    def sents_avgembeddings(self,samples,labels ):
        M = torch.zeros(labels.max()+1, len(samples))
        M = M.to(samples.device)
        M[labels, torch.arange(len(samples))] = 1
        M = torch.nn.functional.normalize(M, p=1, dim=1)
        return torch.mm(M, samples)

    def documents_emb(self, last_hidden_state, labeled_input_ids):
        doc_embs = []
        for samples,labels in zip( last_hidden_state, labeled_input_ids ):
            doc_embs.append(self.sents_avgembeddings(samples,labels ))
        mask = [torch.tensor([1]).expand(emb.shape[0]) for emb in doc_embs]
        doc_embs = self.pack_unmachable(doc_embs)
        mask =torch.LongTensor(self.pack_unmachable(mask))
        mask = mask.to(last_hidden_state.device)
        #ignnored special tokens like cls,",","!"?"...
        return doc_embs[:,1:,:], mask[:,1:]

    def forward(self, last_hidden_state, labeled_input_ids):
        features,attention_mask = self.documents_emb( last_hidden_state, labeled_input_ids )
        weights_mask = attention_mask.unsqueeze(-1)
        att = torch.tanh(self.W(features))
        score = self.V(att)
        score[attention_mask==0]=-1e4
        attention_weights = torch.softmax(score, dim=1)
        context_vector = torch.sum(attention_weights*weights_mask*features, dim=1)
        return context_vector


class NLPPooling(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        if self.pooling_name =="AttentionHead":
            self.pooler = AttentionHead(self.in_features, self.out_features)
        elif self.pooling_name not in ("CLS",''):
            self.pooler = eval(self.pooling_name)(**self.params)

        print(f'Pooling: {self.pooling_name}')

    def forward(self, last_hidden_state, attention_mask_or_labelid):

        if self.pooling_name in ['MeanPooling','MaxPooling','MinPooling']:
            # Pooling between cls and sep / cls and sep embedding are not included
            # last_hidden_state = self.pooler(last_hidden_state[:,1:-1,:],attention_mask[:,1:-1])
            last_hidden_state = self.pooler(last_hidden_state,attention_mask_or_labelid)
        elif self.pooling_name=="CLS":
            # Use only cls embedding
            last_hidden_state = last_hidden_state[:,0,:]
        elif self.pooling_name=="GeMText":
            # Use Gem Pooling on all tokens
            last_hidden_state = self.pooler(last_hidden_state,attention_mask_or_labelid)
        
        elif self.pooling_name=="AttentionHead":
            # sentance pooling ,exclueded cls,sep,",","!","."
            last_hidden_state = self.pooler(last_hidden_state,attention_mask_or_labelid)
        else:
            # No pooling
            last_hidden_state = last_hidden_state
            # print(f"{self.pooling_name} not implemented")
        return last_hidden_state