# ====================================================
# CFG
# ====================================================
class CFG:
    train_out_kaggle = True
    wandb=False
    logConfig = True
    competition='CommonLit - Evaluate Student Summaries'
    _wandb_kernel='nakama'
    debug=False
    apex=True
    print_freq=60
    num_workers=4
    model="microsoft/deberta-v3-base"
    gradient_checkpointing=True
    scheduler='cosine' # ['linear', 'cosine']
    batch_scheduler=True
    num_cycles=0.5
    num_warmup_steps=0
    epochs=6
    encoder_lr=2e-5
    decoder_lr=2e-5
    min_lr=1e-6
    eps=1e-6
    betas=(0.9, 0.999)
    batch_size=32
    
    
    max_len=512
    weight_decay=0.01
    gradient_accumulation_steps=1
    max_grad_norm=1000
    target_cols=['content', 'wording']
    seed=42
    n_fold=4
    trn_fold=[0, 1, 2, 3]
    train=True
    addPrompt = False
    addExtra = False
    poolType = 'mean' #mean:last hidden,l4:last 4 1st token,lstm all hidden,attetion: all hidden
    layerType = 'l2'


if CFG.poolType == 'lstm':
    CFG.epochs = 9
    
if CFG.debug:
    CFG.epochs = 2
    CFG.trn_fold = [0]


# ====================================================
# Utils
# ====================================================
def MCRMSE(y_trues, y_preds):
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:,i]
        y_pred = y_preds[:,i]
        score = mean_squared_error(y_true, y_pred, squared=False) # RMSE
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores


def get_score(y_trues, y_preds):
    mcrmse_score, scores = MCRMSE(y_trues, y_preds)
    return mcrmse_score, scores



import random
import numpy as np

# 假设您的文章数据保存在变量 articles 中，是一个包含多篇文章的列表

# 定义函数用于计算句子的单词个数
def count_words(sentence):
    return len(sentence.split())
def choose_from_list(indicase,seed):
    np.random.seed(seed)
    return np.random.choice(np.array(indicase))
# 定义函数将文章转换成字典形式
def prepare_article(article):
    paragraphs = article.split("\n")
    article_dict = {}
    for i, paragraph in enumerate(paragraphs):
        sentences = paragraph.split(". ")
        article_dict[i] = sentences
    return article_dict
def remove_sent_from_art(art, artID,paraidx,sentidx):
    print(f"#===============\n****RM MSG\n#ARTICLE:{artID}\n#PARAID:{paraidx}\n#SENTID:{sentidx}")
    rms = art[paraidx].pop(sentidx)
    print(f"remove sentence[====]{rms}")
def paralen(paragSents):
    return sum([ count_words(s) for s in  paragSents] )
def len_of_arts(article):
    cnt = 0
    for sents in article.values():
        cnt += paralen(sents)
    return cnt
# 初始化一个空列表用于存储筛选后的文章
def proc_sentens( prompt_texts ):
    selected_articles = []

    # 循环处理每篇文章
    for i,article in enumerate(prompt_texts):
        print(f"#=======================================\n#ARTICLE:{i}")
        article_dict = prepare_article(article)
        proc_time = 0
        # 按段落分割文章为段落列表，并记录段落编号
        num_paragraphs = len(article_dict)
        paragraph_indices = list(range(num_paragraphs))
        artlen = len_of_arts( article_dict )
        if num_paragraphs == 1:
            print(f'''#========================
            #ARTICLE: {i} : NUM OF PARA:{num_paragraphs}'''
            )
            while artlen>=512:
                sentidxs = list( range( len(article_dict[0]) ) )
                if len(sentidxs)<4:
                    break
                _id = choose_from_list(sentidxs[1:-1],proc_time)
                sentidxs.remove(_id)
                remove_sent_from_art(article_dict,i,0,_id)
                artlen = len_of_arts( article_dict )
                proc_time += 1
                if proc_time>=50:
                    break
            selected_articles.append( article_dict )
            continue
        if num_paragraphs <= 3:
            print(f'''#========================
            #ARTICLE: {i} : NUM OF PARA:{num_paragraphs}'''
            )
            while artlen>=512:
                len_of_paras = [ paralen( article[sid] ) for sid in range(num_paragraphs) ]
                max_para_idx = len_of_paras.index(max(len_of_paras))
                sentidxs = list( range( len(article_dict[max_para_idx]) ) )
                if len(sentidxs)<4:
                    proc_time += 1
                    if proc_time>=50:
                        break
                    continue
                _id = choose_from_list(sentidxs[1:-1],proc_time)
                sentidxs.remove(_id)
                remove_sent_from_art(article,i,max_para_idx,_id)
                artlen = len_of_arts( article_dict )
                proc_time += 1
                if proc_time>=50:
                    break
            selected_articles.append( article_dict )
            continue
        print(f'''#========================
            #ARTICLE: {i} : NUM OF PARA:{num_paragraphs}'''
            )
        head_index = paragraph_indices[0]
        tail_index = paragraph_indices[-1]

        paraslength = sum([ paralen( article_dict[sid] ) for sid in [head_index,tail_index] ])
        print(f'''#========================
            #ARTICLE: {i} : HEAD AND TAIL :{paraslength}'''
            )

        if paraslength>=512:
            selected_articles.append( {k:article_dict[k] for k in [head_index,tail_index]} )
            continue
        # 初始化选定的段落字典
        while artlen>=512:
            para_id = choose_from_list(paragraph_indices[1:-1],proc_time)
            sentidxs = list( range( len(article_dict[para_id]) ) )

            if len(sentidxs)<4:
                proc_time += 1
                if proc_time>=50:
                    break
                continue

            _id = choose_from_list(sentidxs[1:-1],proc_time)
            print(f'''#========================
            #ARTICLE: {i} 
            #RM CHOIEC:PARA[{para_id}],SENT[{sentidxs}]
            #ABOUT TO RM LEN:{count_words( article_dict[para_id][_id] )}'''
            )
            sentidxs.remove(_id)
            remove_sent_from_art(article_dict,i,para_id,_id)
            artlen = len_of_arts( article_dict )
            proc_time += 1
            if proc_time>=50:
                break
        proc_time = 0
        while artlen>=512:

            para_id_lens = [ paralen(article_dict[idd]) for idd in range( num_paragraphs ) ]
            _mean_len = np.mean( para_id_lens )
            _over_avg = [ _idc for _idc in range( num_paragraphs ) if para_id_lens[ _idc ]> _mean_len ]
            if len(_over_avg)==1:
                para_id = _over_avg[0]
            else:
                para_id = choose_from_list(_over_avg,proc_time)
    #         para_id = para_id_lens.index(max(para_id_lens))

            print(f'''#========================
            #ARTICLE: {i} 
            #EACH PARA LENTS [{para_id_lens}],MAX ID [{para_id}]
             '''
            )
            sentidxs = list( range( len(article_dict[para_id]) ) )

            if len(sentidxs)<4:
                proc_time += 1
                if proc_time>=50:
                    break
                continue

            _id = choose_from_list(sentidxs[1:-1],proc_time)
            print(f'''#========================
            #ARTICLE: {i} 
            #LEN:{artlen}
            #RM OVER AVG CHOIEC:PARA[{para_id}],SENT[{sentidxs}]
            #ABOUT TO RM LEN:{count_words( article_dict[para_id][_id] )}'''
            )
            sentidxs.remove(_id)
            remove_sent_from_art(article_dict,i,para_id,_id)
            artlen = len_of_arts( article_dict )
            proc_time += 1
            if proc_time>=50:
                break
        selected_articles.append( article_dict )
        continue
    rt_sents = [ ]
    for o in selected_articles:
        a = '\n'.join([' '.join(para) for para in o.values()])
        rt_sents.append( a )
    return rt_sents

def prepare_prompt_input(cfg, text):
    inputs = cfg.tokenizer.encode_plus(
        text, 
        return_tensors=None, 
        add_special_tokens=True, 
        max_length=CFG.max_len,
        pad_to_max_length=True,
        truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs




# ====================================================
# Dataset
# ====================================================
def prepare_input(cfg, text):
    inputs = cfg.tokenizer.encode_plus(
        text,
        return_tensors=None,
        add_special_tokens=True,
        max_length=CFG.max_len,
        pad_to_max_length=True,
        truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs
procd_tr_prompt

class TrainDataset(Dataset):
    def __init__(self, cfg, df,prompt_dict):
        self.cfg = cfg
        self.texts = df['text'].values
        self.prompt_id = df['prompt_id'].values
        self.prompt_dict = prompt_dict #tokenized
        if self.cfg.addExtra:
            self.extra_feats = df[CFG.extra_feats].values
        self.labels = df[cfg.target_cols].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item])

        label = torch.tensor(self.labels[item], dtype=torch.float)
        if self.cfg.addExtra:
            extra_feats = self.extra_feats[item]
            return inputs, extra_feats, label
        if self.cfg.addPrompt:
            prompt_inputs = self.prompt_dict[ self.prompt_id[item] ]
            return inputs, prompt_inputs, label
        return inputs, label


def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:, :mask_len]
    return inputs


# ====================================================
# Model
# ====================================================
from models.attn import FullAttention, ProbAttention, AttentionLayer


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class LSTMPooling(nn.Module):
    def __init__(self, num_layers, hidden_size, hiddendim_lstm):
        super(LSTMPooling, self).__init__()
        self.num_hidden_layers = num_layers
        self.hidden_size = hidden_size
        self.hiddendim_lstm = hiddendim_lstm
        self.lstm = nn.LSTM(self.hidden_size, self.hiddendim_lstm, batch_first=True)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, all_hidden_states):
        ## forward
        # hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
        #                              for layer_i in range(1, self.num_hidden_layers+1)], dim=-1)
        
        # hidden_states = torch.stack(all_hidden_states) # Shape is [13, batch_size, seq_len, 768]
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
    def __init__(self, num_layers, hidden_size, hiddendim_fc):
        super(AttentionPooling, self).__init__()
        self.num_hidden_layers = num_layers
        self.hidden_size = hidden_size
        self.hiddendim_fc = hiddendim_fc
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
    
    
class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.
            if cfg.logConfig:
                LOGGER.info(self.config)
                cfg.logConfig = False
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel(self.config)
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        d_model = self.config.hidden_size
        n_heads = 8
        dropout_rate = 0.1
        # CrosAttParameters={
        #     'd_model': d_model,
        #     'd_ff': d_model*4,
        #     'n_heads': 8,
        #     'factor': 20,
        #     'dropout': 0.1,
        #     'attType': 'prob',
        #     'activation': "relu"
        # }
        # CrosConvParameters={'c_in':d_model,"d_model":d_model,"kernel_size":5}
        # self.encoders = PromptAwareEncoder(
        #                     numcross = 3,
        #                     numconv1 = 0,
        #                     d_model = d_model,
        #                     attParameter = CrosAttParameters,
        #                     downConvPara = CrosConvParameters,
        #             )
        
        self._mulheadatten = AttentionLayer(
            ProbAttention(  
                    False,
                    factor = 35, 
                    attention_dropout= dropout_rate, 
                    output_attention = False
                ), 
                d_model, 
                n_heads, 
                mix=False 
            )
        self.dropout = nn.Dropout(dropout_rate)

        self.encoder_prompt_first = True
        self.hiddendim_lstm = 256
        self.hiddendim_fc = 128
        
         #mean:last hidden,l4:last 4 1st token,lstm all hidden,attetion: all hidden
        if self.cfg.poolType == 'lstm':
            finaldim = self.hiddendim_lstm
            self.pool = LSTMPooling(self.config.num_hidden_layers, self.config.hidden_size, self.hiddendim_lstm)
        if self.cfg.poolType == 'attetion':
            finaldim = self.hiddendim_fc
            self.pool = AttentionPooling(self.config.num_hidden_layers, self.config.hidden_size, self.hiddendim_fc)
        if self.cfg.poolType == 'mean':
            self.pool = MeanPooling()
            finaldim = self.config.hidden_size
        if self.cfg.poolType == 'l4':
            finaldim = self.config.hidden_size*4
        
        if self.cfg.addPrompt:
            finaldim = self.config.hidden_size
            self.pool = MeanPooling()
        # 
#         # self.pool = MeanPooling()
        
#         # self.normpool = nn.LayerNorm(self.config.hidden_size*4)
#         # nn.BatchNorm1d(self.config.hidden_size*4)
      

#         # if CFG.poolType == 'lstm':
#         #     CFG.epochs = 8
    
        
#         # finaldim = self.hiddendim_fc
#         # finaldim = self.hiddendim_lstm
#         # finaldim = self.config.hidden_size*4
#         if self.cfg.addExtra:
#             self.extra_proj = nn.Sequential(
#                     nn.Conv1d(CFG.num_extra_feats, CFG.num_extra_feats//2, 1),
#                     nn.ReLU(),
#                     nn.Dropout(0.2),
#                     nn.Conv1d(CFG.num_extra_feats//2, 64, 1),
#                     nn.BatchNorm1d(64),
#                     nn.ELU(),
#                     nn.Dropout(0.1)
#             )
#             finaldim = finaldim + 64
            # self.config.hidden_size*4 + 64

        self.outproj = nn.Sequential(
                nn.Conv1d(finaldim, finaldim//2, 1),
                nn.BatchNorm1d(finaldim//2),
                nn.ELU(),
                nn.Dropout(0.2),
                nn.Conv1d(finaldim//2, 64, 1),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Conv1d(64, 2, 1)
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs, prompts = None):
        if prompts is not None:
            hidden_states = self.model(**inputs)[0]
            p_hidden_states = self.model(**prompts)[0]
            hidden_states = torch.max(hidden_states,dim=1).values*torch.max(p_hidden_states,dim=1).values
#             hidden_states = self.dropout(
#                     self._mulheadatten(
#                         hidden_states, p_hidden_states, p_hidden_states,
#                         attn_mask=None )[0]
#                 )
#             # [:, 0, :]
            
#             hidden_states = torch.max(hidden_states,dim=1).values
            # if self.encoder_prompt_first:
            #     # hidden_states = self.encoders(p_hidden_states, hidden_states)[:, 0, :]
            #     hidden_states = self.dropout(
            #         self._mulheadatten(
            #             hidden_states, p_hidden_states, p_hidden_states,
            #             attn_mask=cross_mask )[0]
            #     )[:, 0, :]
            # else:
            #     hidden_states = self.encoders(hidden_states, p_hidden_states)
            #     hidden_states = self.pool(hidden_states, inputs['attention_mask'])
            del p_hidden_states
        else:
            hidden_states = self.model(**inputs)[0]#0: last hidden 1:all hidden
            #################all hidden  1st TOKEN lstm#########################
            if self.cfg.poolType in ['attetion','lstm']:
                hidden_states = self.pool(hidden_states)


            #################LAST 4 LAYER 1st TOKEN#########################
            if self.cfg.poolType in ['l4','mean']:
                hidden_states = self.pool(hidden_states, inputs['attention_mask'])
        
        return hidden_states

    def forward(self, inputs, extra_feats=None,prompts = None):
        if prompts is not None:
            feature = self.feature(inputs, prompts)
        else:
            feature = self.feature(inputs)
        
        ########norm layer###############
        # feature = self.normpool(feature)
        
        if self.cfg.addExtra:
            extra_feats = self.extra_proj(extra_feats.to(torch.float32).unsqueeze(-1)).squeeze(-1)
            feature = torch.cat([feature.to(torch.float32), extra_feats.to(torch.float32)],dim=-1)
            del extra_feats
        feature = self.outproj(feature.unsqueeze(-1)).squeeze(-1)
        return feature



# ====================================================
# Loss
# ====================================================
class RMSELoss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred, y_true):
        loss = torch.sqrt(self.mse(y_pred, y_true) + self.eps)
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss



# ====================================================
# Helper functions
# ====================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def train_fn(cfg, fold, train_loader,train_loaders, model, criterion, optimizer, epoch, scheduler, device):
    # if epoch==0:
    #     model.encoder_prompt_first = True
    # else:
    #     model.encoder_prompt_first = False
        
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    step = 0
    # if cfg.addPrompt:
    if train_loaders is None:
        train_loaders = [train_loader]
    for dataloder in train_loaders:
        for o in dataloder:
    # for step, o in enumerate(train_loader):
            if cfg.addExtra:
                inputs, extra_feats, labels = o
                extra_feats = extra_feats.to(device)
            elif cfg.addPrompt:
                inputs, prompt_inputs, labels = o
                for k, v in prompt_inputs.items():
                    prompt_inputs[k] = v.to(device)
            else:
                inputs, labels = o
            inputs = collate(inputs)
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            labels = labels.to(device)
            batch_size = labels.size(0)
            with torch.cuda.amp.autocast(enabled=cfg.apex):
                if cfg.addExtra:
                    y_preds = model(inputs, extra_feats)
                elif cfg.addPrompt:
                    y_preds = model(inputs, prompts = prompt_inputs)
                else:
                    y_preds = model(inputs)
                loss = criterion(y_preds, labels)
            if cfg.gradient_accumulation_steps > 1:
                loss = loss / cfg.gradient_accumulation_steps
            losses.update(loss.item(), batch_size)
            scaler.scale(loss).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
                if cfg.batch_scheduler:
                    scheduler.step()
            end = time.time()
            if step % cfg.print_freq == 0 or step == (len(dataloder)-1):
                print('  step {0}/{1}\n        [=======================] '
                      'Elapsed {remain:s} '
                      'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                      'Grad: {grad_norm:.4f}  '
                      'LR: {lr:.8f}  '
                      .format(step, len(dataloder),
                              remain=timeSince(start, float(step+1)/len(dataloder)),
                              loss=losses,
                              grad_norm=grad_norm,
                              lr=scheduler.get_lr()[0]))
            if cfg.wandb:
                wandb.log({f"[fold{fold}] loss": losses.val,
                           f"[fold{fold}] lr": scheduler.get_lr()[0]})
            step +=1
    return losses.avg


def valid_fn(cfg, valid_loader, model, criterion, device):
    # if epoch==0:
    #     model.encoder_prompt_first = True
    # else:
    #     model.encoder_prompt_first = False
        
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, o in enumerate(valid_loader):
        if cfg.addExtra:
            inputs, extra_feats, labels = o
            extra_feats = extra_feats.to(device)
        elif cfg.addPrompt:
            inputs, prompt_inputs, labels = o
            for k, v in prompt_inputs.items():
                prompt_inputs[k] = v.to(device)
        else:
            inputs, labels = o
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            if cfg.addExtra:
                y_preds = model(inputs, extra_feats)
            elif cfg.addPrompt:
                y_preds = model(inputs, prompts = prompt_inputs)
            else:
                y_preds = model(inputs)
            loss = criterion(y_preds, labels)
        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.to('cpu').numpy())
        end = time.time()
        if step % cfg.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step+1)/len(valid_loader))))
    predictions = np.concatenate(preds)
    return losses.avg, predictions




# ====================================================
# train loop
# ====================================================
def train_loop(folds, fold, prompt_dict,uniLoader=False):

    LOGGER.info(f'''#========================================
    # Status : training
    # Fold : {fold+1}
    #========================================''')

    # ====================================================
    # loader
    # ====================================================
    if uniLoader:
        train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    else:
        train_folds = [folds[folds['fold'] == fd].reset_index(drop=True) for fd in CFG.trn_fold if fd != fold]
        print(f'NUM OF TRAIN FOLDS[====]{len(train_folds)}')
    valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)
    valid_labels = valid_folds[CFG.target_cols].values
    if uniLoader:
        train_dataset = TrainDataset(CFG, train_folds, prompt_dict) 
    else:
        train_dataset = [TrainDataset(CFG, train_fold, prompt_dict) for train_fold in  train_folds]
    valid_dataset = TrainDataset(CFG, valid_folds, prompt_dict)
    if uniLoader:
        train_loader = DataLoader(
                    train_dataset,
                    batch_size=CFG.batch_size,
                    shuffle=True,
                    num_workers=CFG.num_workers, pin_memory=True, drop_last=True
            )
    else:
        train_loaders = [
                DataLoader(
                    o_train_dataset,
                    batch_size=CFG.batch_size,
                    shuffle=True,
                    num_workers=CFG.num_workers, pin_memory=True, drop_last=True
                ) for o_train_dataset in train_dataset
        ]
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.batch_size * 2,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(CFG, config_path=None, pretrained=True)
    torch.save(model.config, OUTPUT_DIR+'config.pth')
    model.to(device)

    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
             'lr': decoder_lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters

    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=CFG.encoder_lr,
                                                decoder_lr=CFG.decoder_lr,
                                                weight_decay=CFG.weight_decay)
    optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)

    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
            )
        elif cfg.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
            )
        return scheduler

    # num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)
    num_train_steps = int( (len(folds)-len(valid_folds)) / CFG.batch_size * CFG.epochs)

    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.SmoothL1Loss(reduction='mean') # RMSELoss(reduction="mean")

    best_score = np.inf

    for epoch in range(CFG.epochs):#CCCC
        print(f'Epoch {epoch+1}/{CFG.epochs}')

        start_time = time.time()

        # train
        if uniLoader:
            avg_loss = train_fn(CFG, fold, train_loader, None, model, criterion, optimizer, epoch, scheduler, device)
        else:
            avg_loss = train_fn(CFG, fold, None,train_loaders, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, predictions = valid_fn(CFG, valid_loader, model, criterion, device)

        # scoring
        score, scores = get_score(valid_labels, predictions)

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}  Scores: {scores}')
        if CFG.wandb:
            wandb.log({f"[fold{fold}] epoch": epoch+1,
                       f"[fold{fold}] avg_train_loss": avg_loss,
                       f"[fold{fold}] avg_val_loss": avg_val_loss,
                       f"[fold{fold}] score": score})

        if best_score > score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'predictions': predictions},
                        OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth")

    predictions = torch.load(OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
                             map_location=torch.device('cpu'))['predictions']
    valid_folds[[f"pred_{c}" for c in CFG.target_cols]] = predictions

    torch.cuda.empty_cache()
    gc.collect()

    return valid_folds