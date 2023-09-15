import re
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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

        # 计算总结和段落之间的余弦相似度
        similarities = cosine_similarity(summary_vectors, paragraph_vectors)

        return list(np.array(similarities).mean(0))

    def processor(self,summary_df,prompt_df):
        prompt_ids = prompt_df['prompt_id'].values
        prompts = prompt_df['prompt_text'].values
        _len_prompts = list(prompt_df['prompt_text'].map( self._len_paragraphs ).values)
        _proced_prompts = []
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
                _plens = 0
                _ids = [ ]
                for idx,_,L in _info:
                    _plens += L
                    _ids.append(idx)
                    #===============================
                    #fillter sub text
                    if _plens > self.prompt_max_len:
                        text = para_list[idx]
                        newtext = ""
                        _plens -= L
                        nlp = spacy.load("en_core_web_sm")
                        doc = nlp(text)
                        for sent in doc.sents:
                            _plens+=len(sent.text.split())
                            newtext+=sent.text
                            if _plens > self.prompt_max_len:
                                break
                        para_list[idx] = newtext
                print(f"Pid:{pid} subset len :{_plens} getted")
                _ids = sorted(_ids)
                para_list = [para_list[idx] for idx in _ids]
                _proced_prompts.append('\n'.join( para_list ) )
        print( '#=========================' )
        print( "#proced prompt" )
        print( "#CASE BEFORE" )
        print( '#=========================\n\n' )
        print(prompt_df['prompt_text'].values[0])
        prompt_df['prompt_text'] = _proced_prompts
        print( '#=========================' )
        print( "#CASE AFTER" )
        print( '#=========================\n\n' )
        print(prompt_df['prompt_text'].values[0])
        
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
    def processor2(self, summary_df,prompt_df):
        prompt_df['prompt_text'] = proc_sentens(prompt_df['prompt_text'].values)
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
