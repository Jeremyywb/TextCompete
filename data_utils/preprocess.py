import re
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
					if _plens > self.prompt_max_len:
						break
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
			summary_df['text'] = 'question: ' + summary_df['prompt_question'] + ' [QUESSEP] ' + summary_df['text']
		if self.experiment:
			summary_df = summary_df.groupby("prompt_id").sample(frac=self.experiment_rate, random_state=2)
		return summary_df, prompt_df

