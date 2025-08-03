# 指定Excel文件
excelFileName = 'data.xlsx'
# sheetName = '电量详情'
dataRange = 'A:AE'
startRow = 0
# 读取数据
data = pd.read_excel(excelFileName, usecols=dataRange, skiprows=startRow)
# 过滤第 C 列为"责令停产"的数据
filtered_data = data.loc[data['ent_status'] == 'Normal production state']
# 重新设置索引从 1 开始排序
filtered_data.reset_index(drop=True, inplace=True)

# 创建一个新的 Excel 文件
workbook = Workbook()
sheet = workbook.active

# 将 filtered_data 的数据保存到新的 Excel 文件中
for row in range(1, len(filtered_data) + 1):
    for col in range(1, len(filtered_data.columns) + 1):
        sheet.cell(row=row, column=col, value=filtered_data.iloc[row-1, col-1])

# 保存新的 Excel 文件
workbook.save('长期停产.xlsx')
print("新的 Excel 文件已保存。")

aggregated_data = filtered_data.drop(filtered_data.columns[5:31], axis=1)
aggregated_data = aggregated_data.drop(aggregated_data.columns[2], axis=1)
# 创建一个新的 DataFrame 用于存储结果
new_data = []
count = {}
# 获取所有日期并初始化计数字典
all_dates = sorted(pd.to_datetime(aggregated_data['data_time']).unique())
for date in all_dates:
    count[date] = 0
# 按测量点名称分组
for _, group in aggregated_data.groupby('meter_no_tm'):
    # 确定分组后的数据长度
    group_length = len(group) - 29  # 保证后续索引不会越界

    # 遍历分组中的所有行（考虑到需要连续的30天时间点）
    for i in range(group_length):
        dates = [group.iloc[i + j]['data_time'] for j in range(30)]
        consumption = [group.iloc[i + j]['elec_consumption'] for j in range(30)]

        # 检查这30天是否连续
        if all((pd.to_datetime(dates[j + 1]) - pd.to_datetime(dates[j])).days == 1 for j in range(29)):
            # 提取对应的电量消耗数据
            meter_no_tm = group.iloc[i]['meter_no_tm']
            first_date = dates[0]
            first_value = consumption[0]
            second_value = consumption[1]
            thirty_value = consumption[-1]

            # 添加到 new_data 列表
            new_data.append([
                meter_no_tm,
                first_date,
                first_value,
                *[consumption[j] for j in range(1, 30)],
                thirty_value
            ])
        else:
            # 更新从 first_date 到最后一个日期的所有日期的计数
            for date in all_dates:
                if date >= first_date:
                    count[date] += 1

# 将结果转换为 DataFrame
new_data = pd.DataFrame(new_data)
df = new_data.drop(new_data.columns[[0,1]], axis=1)
scaler = MinMaxScaler(feature_range=(0, 1))
# 对每一行数据进行归一化
data_preprocessed = scaler.fit_transform(df)
############################
class ElectricityDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

# 假设您的输入数据是一个 24 x 1000 的矩阵
input_data = torch.from_numpy(data_preprocessed).float()
data_normalized = input_data.numpy()
# 创建数据集和数据加载器
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
    with open(f'som_model-changqi_{l}.pkl', 'rb') as model_file:
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

second_column_data = data_normalized[:, 1]
max_value_second_column = np.max(second_column_data)
threshold_value_up = max_value_second_column * 0.22  # 10% 的阈值
threshold_value_down = max_value_second_column * 0.05  # 10% 的阈值

blue_fill = PatternFill(start_color='0000FF', end_color='0000FF', fill_type='solid')

# 加载原始Excel文件以便修改
wb = load_workbook('长期停产.xlsx')
ws = wb.active

# 计算最终概率并汇总，同时检查条件
rows_to_highlight = []
for idx, probabilities in enumerate(final_probabilities_all):
    if probabilities is not None:
        # 检查最终概率条件 (假设类别索引从0开始)
        sum_1 = sum(probabilities[[0]])
        sum_2 = sum(probabilities[[1,2]])
        # 对连续30天的数据进行条件检查
        all_conditions_met = True
        for i in range(30):
            if not (threshold_value_down > data_normalized[idx + i, 0] and
                    threshold_value_down > data_normalized[idx + i, 1] and
                    threshold_value_down > data_normalized[idx + i, 2] and
                    threshold_value_down > data_normalized[idx + i, 3] and
                    threshold_value_down > data_normalized[idx + i, 4]):
                all_conditions_met = False
                break

        if sum_1 < sum_2 and all_conditions_met:
            current_date = pd.to_datetime(aggregated_data.iloc[idx]['data_time'])
            cumulative_count = count.get(current_date, 0)
            rows_to_highlight.append(idx + 1 + cumulative_count)

# 遍历要高亮的行，并在Excel中标记它们
for row in rows_to_highlight:
    ws.cell(row=row, column=1).fill = blue_fill  # 只对每行的第一列应用蓝色填充
    # for cell in ws[row]:
    #     cell.fill = blue_fill

# 保存修改后的Excel文件
wb.save('highlighted.xlsx')
print("已根据条件标蓝相关行。")
