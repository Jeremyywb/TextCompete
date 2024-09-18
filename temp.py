def map_amount_to_interval(amount):
    """
    将给定的额度值映射到相应的区间标签。
    
    Args:
        amount (float): 额度值。
        
    Returns:
        str: 区间标签。
    """
    # 定义区间
    intervals_50 = [(i, i + 50) for i in range(0, 500, 50)]
    intervals_100 = [(i, i + 100) for i in range(500, 2000, 100)]
    
    # 在间隔为50的区间中查找
    for lower, upper in intervals_50:
        if lower*10e4 < amount <= upper*10e4:
            return f"({lower}-{upper}]"
    
    # 在间隔为100的区间中查找
    for lower, upper in intervals_100:
        if lower*10e4 < amount <= upper*10e4:
            return f"({lower}-{upper}]"
    
    return "Amount out of range"


df['额度区间'] = df['授信额度那个字段'].map( map_amount_to_interval )
result = df.groupby(['quarter','额度区间'])['uscc'].size().reset_index()
result.columns = ['quarter','额度区间','CNT']
res01 = df.groupby(['quarter'])['uscc'].size().to_dict()
result['quarterCNT'] = result['quarter'].map( res01 )
result['季度内额度区间占比'] = result['CNT']/result['quarterCNT']



dfs = []
for group_name, group_df in df.groupby('quarter'):

	summary = group_df['授信额度那个字段'].value_counts().reset_index()
	#normalize=True 去掉
	summary.columns = ['额度值','个数']
	total = summary['个数'].sum()
	summary['总数'] = total
	summary['占比'] = summary['个数']/
	summary['quarter'] = group_name
	dfs.append(summary)

creditSummary = pd.concat(dfs)