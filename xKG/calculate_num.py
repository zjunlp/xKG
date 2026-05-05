import pandas as pd
import numpy as np # 导入numpy以使用std函数

# 1. 您的原始数据
full_data = {
    'Group': ['mech']*12 + ['tta']*12,
    'Method': ['full', 'raw-code', 'no-code', 'no-rerank']*3 + ['full', 'raw-code', 'no-code', 'no-rerank']*3,
    'Run': ['Run 1', 'Run 1', 'Run 1', 'Run 1', 'Run 2', 'Run 2', 'Run 2', 'Run 2', 'Run 3', 'Run 3', 'Run 3', 'Run 3']*2,
    'Score': [
        # mech
        0.4648, 0.4651, 0.4017, 0.3803, 0.4481, 0.4457, 0.3879, 0.4191, 0.4013, 0.3891, 0.3523, 0.3425,
        # tta
        0.5399, 0.52578, 0.4831, 0.4459, 0.5217, 0.5131, 0.4728, 0.4687, 0.4801, 0.44128, 0.4289, 0.3812
    ]
}

# 2. 将字典转换为 pandas DataFrame
df = pd.DataFrame(full_data)

# 3. 筛选出 Method 为 'full' 或 'no-rerank' 的行
filtered_df = df[df['Method'].isin(['full', 'no-rerank'])]

# 4. 按 'Group' 和 'Method' 分组，并计算 'Score' 的均值和标准差
#    .agg() 方法可以同时应用多个聚合函数
results = filtered_df.groupby(['Group', 'Method'])['Score'].agg(['mean', 'std'])

# 5. 打印结果
print("计算结果:")
print(results)
