import numpy as np

# 示例数据（净利润值，可能包含零和 NaN）
profit_data = [100, 0, np.nan, 200, np.nan, 0, 400, 500, np.nan, 0, 1000]


n = len(profit_data)
valid_indices = [i for i in range(n) if not np.isnan(profit_data[i])]
num_valid = len(valid_indices)
growth_rates = [0]+[np.nan] * (n-1)

# 第一轮迭代：计算所有非空位置间的平均增幅
for i in range(1, num_valid):
    prev_index = valid_indices[i - 1]
    curr_index = valid_indices[i]
    prev_value = profit_data[prev_index]
    curr_value = profit_data[curr_index]
    if prev_value == 0 and curr_value == 0:#都为0
        growth_rate = 0
    elif prev_value != 0:# 只是前面不为0
        growth_rate = (curr_value-prev_value)/abs(prev_value)
    elif prev_value == 0:# 只是前面为0，后面不为0
        j = i - 2
        while j >= 0 and profit_data[valid_indices[j]] == 0:
            j -= 1
        if j >= 0:
            prev_value = profit_data[valid_indices[j]]
            growth_rate = (curr_value / prev_value) ** (1 / (curr_index - valid_indices[j])) - 1
            growth_rate = np.sign(curr_value-prev_value)*abs(growth_rate)
        else:
            growth_rate = 0
    growth_rates[curr_index] = growth_rate
        


