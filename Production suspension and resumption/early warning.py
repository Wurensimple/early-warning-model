excelFileName = 'data.xlsx'
dataRange = 'A:AC'
startRow = 0
data = pd.read_excel(excelFileName, usecols=dataRange, skiprows=startRow)
filtered_data = data.loc[data['ent_status'] == '长期停产'] # 过滤第 C 列为"责令停产"的数据
filtered_data.reset_index(drop=True, inplace=True) # 重新设置索引从 1 开始排序
workbook = Workbook()
sheet = workbook.active
for row in range(1, len(filtered_data) + 1):
    for col in range(1, len(filtered_data.columns) + 1):
        sheet.cell(row=row, column=col, value=filtered_data.iloc[row-1, col-1])# 将 filtered_data 的数据保存到新的 Excel 文件中
workbook.save('data.xlsx')
print("新的 Excel 文件已保存。")
file_path = 'data.xlsx'
wb = load_workbook(file_path)
ws = wb.active
num = 0
blue_fill = PatternFill(start_color='0000FF', end_color='0000FF', fill_type='solid')
aggregated_data = filtered_data.drop(filtered_data.columns[5:30], axis=1)
aggregated_data = aggregated_data.drop(aggregated_data.columns[2], axis=1)
max_value = aggregated_data['elec_consumption'].max()
threshold = max_value * 0.18  # 10% 的阈值
new_data = []
# count = 0
count = {}
# 获取所有日期并初始化计数字典
all_dates = sorted(pd.to_datetime(aggregated_data['data_time']).unique())
for date in all_dates:
    count[date] = 0
for _, group in aggregated_data.groupby('meter_no_tm'):  # 先按测量点名称分组
    for i in range(len(group) - 2):
        first_date = group.iloc[i]['data_time']
        first_date = pd.to_datetime(first_date)

        next_date = group.iloc[i + 1]['data_time']
        next_date = pd.to_datetime(next_date)

        previous_value = group.iloc[i]['elec_consumption']
        # 检查两个日期是否连续
        if (next_date - first_date).days == 1:
            current_value = group.iloc[i + 1]['elec_consumption']
            new_data.append([group.iloc[i]['meter_no_tm'], first_date, previous_value, current_value])
        # if (next_date - first_date).days != 1:
        #     count += 1
        else:
            # 计算不连续日数并更新字典
            days_gap = (next_date - first_date).days
            if days_gap > 1:
                # 更新从 first_date 到最后一个日期的所有日期的计数
                for date in all_dates:
                    if date >= first_date:
                        count[date] += 1

new_data = pd.DataFrame(new_data)
df = new_data.drop(new_data.columns[[0, 1]], axis=1)
scaler = MinMaxScaler(feature_range=(0, 1))
data_preprocessed = scaler.fit_transform(df)
class ElectricityDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]
input_data = torch.from_numpy(data_preprocessed).float()# 假设您的输入数据是一个 24 x 1000 的矩阵
data_normalized = input_data.numpy()
dataset = ElectricityDataset(input_data)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
data_normalized = input_data.detach().numpy()
#######################################################################
# 从文件中读取参数
loaded = np.load('parameters.npz')
num_rounds = loaded['num_rounds']
max_clusters = loaded['max_clusters']
n_clusters = loaded['n_clusters']
# 读取 ave 值从文件
ave_loaded = np.load('ave_values.npy')

# 打印加载的 ave 值
print("Loaded ave values:")
for round in range(num_rounds):
    print(f"Round {round}:")
    for cluster_id in range(max_clusters):
        print(f"    Cluster {cluster_id}:")
        for i in range(n_clusters):
            print(f"    Probability of being in cluster {i}: {ave_loaded[round][cluster_id][i]:.4f}")

# 读取保存的cluster_neurons_all_rounds
with open('cluster_neurons_all_rounds.pkl', 'rb') as file:
    cluster_neurons_all_rounds = pickle.load(file)

# 打印每一轮的cluster_neurons
for l, cluster_neurons in enumerate(cluster_neurons_all_rounds):
    print(f"SOM model {l} - Cluster neurons:")
    for cluster_id, neurons in cluster_neurons.items():
        print(f"类别 {cluster_id}: {neurons}")
############################################################################
# 加载训练好的 SOM 模型
soms = []
for l in range(4):
    with open(f'som_model-tingchan_{l}.pkl', 'rb') as model_file:
        som = pickle.load(model_file)
        soms.append(som)

# 对data_normalized中的每一条数据进行预测
som_predictions_all = []
hca_result_all = []

for data_point in data_normalized:
    som_predictions = []
    for som in soms:
        winner_neuron = som.winner(data_point)  # 获取获胜神经元
        som_predictions.append(winner_neuron)

    hca_result = []
    for l, winner_neuron in enumerate(som_predictions):
        print(f"SOM model {l} - Winner neuron: {winner_neuron}")

        cluster_neurons = cluster_neurons_all_rounds[l]

        hca_cluster = None
        for cluster_id, neurons in cluster_neurons.items():
            if winner_neuron in neurons:
                hca_cluster = cluster_id
                break
        hca_result.append(hca_cluster - 1 if hca_cluster is not None else None)

    som_predictions_all.append(som_predictions)
    hca_result_all.append(hca_result)

# 计算最终概率并汇总
final_probabilities_all = []
for hca_clusters in hca_result_all:
    final_probabilities = []
    for l, hca_cluster in enumerate(hca_clusters):
        if hca_cluster is not None and 0 <= hca_cluster < ave_loaded.shape[1]:
            probabilities = ave_loaded[l][hca_cluster]
            final_probabilities.append(probabilities)
        else:
            final_probabilities.append(None)

    valid_probabilities = [p for p in final_probabilities if p is not None]
    if valid_probabilities:
        num_clusters = len(valid_probabilities[0])
        summed_probabilities = np.zeros(num_clusters)

        for probabilities in valid_probabilities:
            summed_probabilities += probabilities

        final_probabilities_all.append(summed_probabilities)
    else:
        final_probabilities_all.append(None)

# 打印最终的概率结果
for idx, probabilities in enumerate(final_probabilities_all):
    if probabilities is not None:
        print(f"Data point {idx} - Summed probabilities for each cluster:")
        for i, prob in enumerate(probabilities):
            print(f"Cluster {i}: {prob:.4f}")
    else:
        print(f"Data point {idx} - No valid probabilities to sum.")
##################################################################################################
# 假设 final_probabilities_all 和原始数据之间有相同的顺序
# 如果没有，请确保通过索引或其它方式维持两者之间的对应关系
# 获取 data_normalized 中第二列的所有数据的最大值
second_column_data = data_normalized[:, 1]
max_value_second_column = np.max(second_column_data)
threshold_value_down_1 = max_value_second_column * 0.15  # 10% 的阈值
threshold_value_down = max_value_second_column * 0.2  # 10% 的阈值

blue_fill = PatternFill(start_color='0000FF', end_color='0000FF', fill_type='solid')

# 加载原始Excel文件以便修改
wb = load_workbook('data.xlsx')
ws = wb.active

# 计算最终概率并汇总，同时检查条件
rows_to_highlight = []
for idx, probabilities in enumerate(final_probabilities_all):
    if probabilities is not None:
        # 检查最终概率条件 (假设类别索引从0开始)
        probability_condition = sum(probabilities[[0,2]]) < sum(probabilities[[1]])

        if probability_condition:
            # first_column_condition = data_normalized[idx, 0] > threshold_value_up
            # first_column_condition_1 = data_normalized[idx, 0] > data_normalized[idx, 1]
            # 检查 data_normalized 第二列数据是否小于最大值的10%
            second_column_condition = data_normalized[idx, 1] > threshold_value_down
            second_column_condition_2 = data_normalized[idx, 1] > data_normalized[idx, 0]
            second_column_condition_3 = data_normalized[idx, 0] < threshold_value_down_1
            # if first_column_condition and first_column_condition_1:
            #     rows_to_highlight.append(idx + 1)
            if second_column_condition and second_column_condition_2 and second_column_condition_3:
                # 获取当前行的日期
                current_date = pd.to_datetime(aggregated_data.iloc[idx]['data_time'])

                # 根据当前日期索引 count 字典中的累积不连续天数
                cumulative_count = count.get(current_date, 0)

                rows_to_highlight.append(idx + 2 + cumulative_count)

# 遍历要高亮的行，并在Excel中标记它们
for row in rows_to_highlight:
    ws.cell(row=row, column=1).fill = blue_fill  # 只对每行的第一列应用蓝色填充

# 保存修改后的Excel文件
wb.save('highlighted.xlsx')
print("已根据条件标蓝相关行。")
