import re
# import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.tokenize import sent_tokenize

# =============================
# Processor
# =============================

class PreProcess(object):
    """docstring for PreProcess"""
    def __init__(self, prompt_max_len, add_question, experiment, experiment_rate):
        super(PreProcess, self).__init__()
        self.prompt_max_len = prompt_max_len
        self.add_question = add_question
        self.experiment = experiment
        self.experiment_rate = experiment_rate

    def preprocess(self,text):
        '''
        origal : This is a sentence What .. Incredible  .
        processed : This is a sentence What ... Incredible.
        origal : This is a.....sentence What ?? Incredible.
        processed : This is a...sentence What ??? Incredible.
        '''

        pattern = r"((\.{2,})|(\?{2,})|(!{2,})|(,{2,}))"
        text = re.sub(pattern, lambda match: match.group(1)[:1] * 3, text)
        pattern = r'(?<![\.,!?]) +(?=[\.,!?])'
        replacement = '.'
        text = re.sub(pattern, replacement, text)
        text = text.strip()
        return text

    def _len_list_para(self, texts):
        return [len(s.split()) for s in texts.split('\n')]

    def _len_paragraphs(self, texts):
        return sum(self._len_list_para(texts))

    def _score_paragraphs(self, paragraphs, summaries):
        vectorizer = TfidfVectorizer()
        paragraph_vectors = vectorizer.fit_transform(paragraphs)
        summary_vectors = vectorizer.transform(summaries)
        similarities = cosine_similarity(summary_vectors, paragraph_vectors)

        return list(np.array(similarities).mean(0))

    def clean2sentances(self,text):
        text = text.replace("\r",'')
        pattern = '''(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'''
        # pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|)(?=\s)'
        sentances = re.split(pattern,text)
        sentances
        _id_senta = 0
        cleaned_sentances = []
        sentancce_id = []
        for sen in sentances:
            if len(sen)>0:
                if sen.startswith("\r\n"):
                    sen = sen.replace('\r\n','')
                    _id_senta+=1
                elif sen.startswith("\n"):
                    sen = sen.replace('\n','')
                    _id_senta+=1
                sentancce_id.append(_id_senta)
                cleaned_sentances.append(sen)
        return cleaned_sentances,sentancce_id




    def new_processor(self,summary_df,prompt_df):
        prompt_ids = prompt_df['prompt_id'].values
        prompts = prompt_df['prompt_text'].values
        SAVETEXTS = []
        for IDprompt,ptext in zip(prompt_ids, prompts):
            CleanedPsentances, IDparas =  self.clean2sentances(ptext)
            summaries = summary_df[summary_df['prompt_id']==IDprompt]['text'].values
            _scores = self._score_paragraphs( CleanedPsentances, summaries )
            Lsentance = [ len(s.split()) for s in CleanedPsentances]
            IDsentance = list(range( len(_scores) ))
            SORTINFO = list(zip( IDsentance, _scores, Lsentance, IDparas ))

            #============================================================
            #sort by score descending
            SORTINFO = sorted(SORTINFO, key=lambda x: x[1], reverse=True)
            _LEN = 0
            IDsaved = [ ]
            for SID, SC, L,_ in SORTINFO:
                _LEN+=L
                if _LEN<= self.prompt_max_len:
                    IDsaved.append(SID)
                else:
                    break
            SORTINFO = [(SID, SC, L,PID) for SID, SC, L,PID in SORTINFO if SID in IDsaved]
            SORTINFO = sorted(SORTINFO, key=lambda x: x[0], reverse=False)
            SAVETEXT = ''
            LASTPID = None
            for SID, SC, L,PID in SORTINFO:
                if LASTPID==None:
                    LASTPID = PID
                if LASTPID==PID:
                    SAVETEXT = SAVETEXT+CleanedPsentances[SID]
                else:
                    SAVETEXT = SAVETEXT+'\n'
                    SAVETEXT = SAVETEXT+CleanedPsentances[SID]
            SAVETEXTS.append( SAVETEXT )
        prompt_df['prompt_text'] = SAVETEXTS
        
        #======================================================================
        #SUMMARY TEXT 
        summary_df['text'] = summary_df['text'].map( self.preprocess )
        if self.add_question:
            summary_df = summary_df.merge(
                prompt_df[['prompt_id', 'prompt_question']],
                on = ['prompt_id'],
                how = 'left' 
             )
            summary_df['text'] = summary_df['prompt_question'] + ' [SEP] ' + summary_df['text']
        if self.experiment:
            summary_df = summary_df.groupby("prompt_id").sample(frac=self.experiment_rate, random_state=2)
        return summary_df, prompt_df

    def processor(self,summary_df,prompt_df):
        prompt_ids = prompt_df['prompt_id'].values
        prompts = prompt_df['prompt_text'].values
        _len_prompts = list(prompt_df['prompt_text'].map( self._len_paragraphs ).values)
        _proced_prompts = []
        repreoced_pid = []        
        for pid, plen, paragraphs  in zip(prompt_ids, _len_prompts, prompts):
            summaries = summary_df[summary_df['prompt_id']==pid]['text'].values
            _lens = self._len_list_para( paragraphs )
            para_list = paragraphs.split('\n')
            _scores = self._score_paragraphs( para_list, summaries )
            idx = list(range(len( _scores )))
            
            _info = list(zip(idx,_scores,_lens))
            _info = sorted(_info, key=lambda x: x[1], reverse=True)
            if plen <= self.prompt_max_len:
                _info.pop(-1)
                _info = sorted(_info, key=lambda x: x[0], reverse=False)
                para_list = [para_list[idx] for idx,_,_ in _info if len(para_list[idx])>0]
                _proced_prompts.append('\n'.join( para_list ) )
            else:
                print(f"PID[{pid}] is bigger than {self.prompt_max_len}")
                _plens = 0
                _ids = [ ]
                for idx,_,L in _info:
                    _plens += L
                    _ids.append(idx)
                    if _plens > self.prompt_max_len:
                        if _plens - self.prompt_max_len>32:
                            repreoced_pid.append( pid )
                            print(f"ID:{pid} will reproeceed")
                        break
                _ids = sorted(_ids)
                para_list = [para_list[idx] for idx in _ids]
                _proced_prompts.append('\n'.join( para_list ) )
        prompt_df['prompt_text'] = _proced_prompts
        repreoced_pid = list(set(repreoced_pid))
        # ===========================================
        # common out
        # if len(repreoced_pid)>0:
        #     print(f"REPROCED pids:{repreoced_pid}")
        #     prompt_df = self.reprocessor(summary_df,prompt_df,repreoced_pid)
        # ====================================================================
        summary_df['text'] = summary_df['text'].map( self.preprocess )
        if self.add_question:
            summary_df = summary_df.merge(
                prompt_df[['prompt_id', 'prompt_question']],
                on = ['prompt_id'],
                how = 'left' 
             )
            summary_df['text'] = summary_df['prompt_question'] + ' [SEP] ' + summary_df['text']
        if self.experiment:
            summary_df = summary_df.groupby("prompt_id").sample(frac=self.experiment_rate, random_state=2)
        return summary_df, prompt_df

    def reprocessor(self,summary_df,prompt_df,prompt_ids):
        prompts = prompt_df[prompt_df['prompt_id'].isin(prompt_ids)]['prompt_text'].values
        _len_prompts = [self._len_paragraphs(tts) for tts in prompts]
        _proced_prompts = []   
        for pid, plen, paragraphs  in zip(prompt_ids, _len_prompts, prompts):
            summaries = summary_df[summary_df['prompt_id']==pid]['text'].values
            para_list = paragraphs.split('\n')
            sentances = []
            paraids = []
            for i,paragra in enumerate(para_list) :
                sents = sent_tokenize(paragra)
                sentances+=sents
                paraids.extend( [i]*len(sents) )
            _lens = [ len(v.split()) for v in  sentances]
            _scores = self._score_paragraphs( sentances, summaries )
            idx = list(range(len( _scores )))
            _info = list(zip(idx,_scores,_lens,paraids))
            _info = sorted(_info, key=lambda x: x[1], reverse=True)
            
            # ==================================
            # ret high scored sents ordered
            _plens = 0
            _ids = [ ]
            for _ID,_,L,_SPID in _info:
                _plens += L
                _ids.append((_ID,_SPID))
                if self.prompt_max_len-_plens<32 :
                    break
            _ids = sorted(_ids, key=lambda x: x[0], reverse=False)
            newPrompt = ''
            _TMPPID = None
            for _ID,_SPID in _ids:
                if _TMPPID==None:
                    _TMPPID = _SPID
                if _SPID==_TMPPID:
                    newPrompt += sentances[_ID]
                else:
                    newPrompt += '\n'
                    newPrompt += sentances[_ID]
            print(f"PID {pid} RESET: {plen}--> {_plens} ")
            _proced_prompts.append(newPrompt)
        prompt_df.loc[prompt_df['prompt_id'].isin(prompt_ids), 'prompt_text'] = _proced_prompts
        return prompt_df




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


# ==================================================
# sentence remove

# df = pd.read_csv("/home/data/prompts_train.csv")
def common_del(text):
    if len(text.split(","))>5:
        te = splitWith(text, ",")
        text = "".join(te[:4]+[te[-1]]  )
    return text

def para_to_sentence(para2):
    stext = []
    tesDots = splitWith(para2, ".")
    for tes in tesDots:
        tesQues = splitWith(tes, "?")
        for tesQ in tesQues:
            tesNGs = splitWith(tesQ, "!")
            for tesN in tesNGs:
                stext.extend(splitWith(tesN, ";"))
    stext = [c for c in stext if len(c)>0]
    return stext

def prasenLen(paraSenteces):
    return len( paraSenteces )>8
def parasenNumWord( senAndParas ):
    return sum(len("".join(paraSenteces).split()) for paraSenteces in senAndParas)
def splitWith(text, punct):
    SPLIT_SIMPLE = "|" 
    texts = text.split(punct)
    # texts = [te.strip() for te in texts if len(te.strip())>0]
    tejoins =punct + SPLIT_SIMPLE
    output = tejoins.join(texts).split(SPLIT_SIMPLE)
    output = [c for c in output if len(c)>0]
    return output

def SentenceRemove(test):
    if len(test.split())<=512:
        print("Step 1")
        return test
    else:
        stext = []
        for para in splitWith(test,"\r\n"):
            para2s =  splitWith(para,"\n")
            for para2 in para2s:
                stext.extend( para_to_sentence(para2))
        test = "".join([ " ".join([ su for su in s.split(" ") if len(su)<=20 ]) for s in stext  ])
        if len(test.split())<=512:
            print("Step 2")
            return test
        stext = []
        for para in splitWith(test,"\r\n"):
            para2s =  splitWith(para,"\n")
            for para2 in para2s:
                stext.extend( para_to_sentence(para2))

        stext = [common_del(text) for text in stext]
        test = "".join(stext)
        if len(test.split())<=512:
            print("Step 3")
            return test
        paras = []
        
        for para in splitWith(test,"\r\n"):
            para2s =  splitWith(para,"\n")
            for para2 in para2s:
                paras.append(para_to_sentence(para2))
#         print(paras)
        is_over_8 = any([prasenLen(paraSenteces) for paraSenteces in paras])
        numwords = parasenNumWord( paras )
        _start = 0
        while (is_over_8 and numwords>512):
            findP = False
            for i,senP in enumerate(paras):
                if i==len(paras)-1:
                    _start = 0
                if i <_start:
                    continue
                if len(senP)>8:
                    findP = True
                    _start=i+1
                    break
            if findP:
                print(f"POP VALUES para{i}: ",paras[i][-2],"\n")
                paras[i].pop(-2)
            is_over_8 = any([prasenLen(paraSenteces) for paraSenteces in paras])
            numwords = parasenNumWord( paras )
        test = "".join(["".join(paraSentences) for paraSentences in paras])
        if len(test.split())<=512:
            print("Step 4")
            return test
        test = " ".join(test.split(" ")[:512])
        print("Step 5")
        return test

