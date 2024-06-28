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
        if lower < amount <= upper:
            return f"({lower}-{upper}]"
    
    # 在间隔为100的区间中查找
    for lower, upper in intervals_100:
        if lower < amount <= upper:
            return f"({lower}-{upper}]"
    
    return "Amount out of range"


df['额度区间'] = df['授信额度那个字段'].map( map_amount_to_interval )
