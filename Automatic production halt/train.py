# 指定Excel文件
excelFileName = 'data.xlsx'
dataRange = 'A:AC'
startRow = 0
data = pd.read_excel(excelFileName, usecols=dataRange, skiprows=startRow)
filtered_data = data.loc[data['ent_status'] == 'Normal production state']
filtered_data.reset_index(drop=True, inplace=True)
# 创建一个新的 Excel 文件
workbook = Workbook()
sheet = workbook.active

# 将 filtered_data 的数据保存到新的 Excel 文件中
for row in range(1, len(filtered_data) + 1):
    for col in range(1, len(filtered_data.columns) + 1):
        sheet.cell(row=row, column=col, value=filtered_data.iloc[row-1, col-1])

# 保存新的 Excel 文件
workbook.save('data.xlsx')
print("新的 Excel 文件已保存。")

aggregated_data = filtered_data.drop(filtered_data.columns[5:30], axis=1)
aggregated_data = aggregated_data.drop(aggregated_data.columns[2], axis=1)
new_data = []

# 按测量点名称分组
for _, group in aggregated_data.groupby('meter_no_tm'):
    # 确定分组后的数据长度
    group_length = len(group) - 4  # 保证后续索引不会越界
    # 遍历分组中的所有行（考虑到需要连续的五个时间点）
    for i in range(group_length):
        first_date = group.iloc[i]['data_time']
        second_date = group.iloc[i + 1]['data_time']
        third_date = group.iloc[i + 2]['data_time']
        fourth_date = group.iloc[i + 3]['data_time']
        fifth_date = group.iloc[i + 4]['data_time']

        # 检查这五天是否连续
        if (pd.to_datetime(second_date) - pd.to_datetime(first_date)).days == 1 and \
                (pd.to_datetime(third_date) - pd.to_datetime(second_date)).days == 1 and \
                (pd.to_datetime(fourth_date) - pd.to_datetime(third_date)).days == 1 and \
                (pd.to_datetime(fifth_date) - pd.to_datetime(fourth_date)).days == 1:
            # 提取对应的电量消耗数据
            first_value = group.iloc[i]['elec_consumption']
            second_value = group.iloc[i + 1]['elec_consumption']
            third_value = group.iloc[i + 2]['elec_consumption']
            fourth_value = group.iloc[i + 3]['elec_consumption']
            fifth_value = group.iloc[i + 4]['elec_consumption']

            # 添加到 new_data 列表
            new_data.append([
                group.iloc[i]['meter_no_tm'],
                first_date,
                first_value,
                second_value,
                third_value,
                fourth_value,
                fifth_value
            ])

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


input_data = torch.from_numpy(data_preprocessed).float()
data_normalized = input_data.numpy()
# 创建数据集和数据加载器
dataset = ElectricityDataset(input_data)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

data_normalized = input_data.detach().numpy()

np.random.seed(42)
num_samples = len(data_normalized)
num_train_samples = int(0.8 * num_samples)# 记录每个子集中的样本在原始数据中的索引位置
subset_indices = {}
for i in range(4):
    random_indices = np.random.permutation(num_samples)
    train_indices = random_indices[:num_train_samples] # 选择前80%的索引作为训练集
    train_subset = data_normalized[train_indices]
    subset_indices[f'SOM_{i}'] = {j: train_indices[j] for j in range(len(train_indices))}
    subset_filename = f'subset_{i}.npy'
    np.save(subset_filename, train_subset)
    df = pd.DataFrame(train_subset)
    subset_filename = f'subset_{i}.xlsx'
    df.to_excel(subset_filename, index=False)
    print(f"训练子集 {i} 已保存为 {subset_filename}")

consensus_matrix = np.zeros((num_samples, num_samples))
# 定义 SOM 的维度
dimension1 = 3
dimension2 = 5
# 指定每一轮次选择的聚类组
selected_clusters_per_round = [[0], [0], [0], [0]]  # 每一轮次选择的聚类组
soms = []
som_output_indices_list = []
cluster_labels_result = []
cluster_neurons_all_rounds = []
##################################################################################################################
# 初始化一个列表来存储每一轮的最优簇数(optimal_k)
optimal_ks = []
for l in range(4):
    subset_filename = f'subset_{l}.npy'
    subset = np.load(subset_filename)
    som = MiniSom(dimension1, dimension2, subset.shape[1])
    som.train_random(subset, 1000)
    som_output_indices = [som.winner(x) for x in subset]
    som_output_indices = np.array([som.winner(x) for x in subset])
    # 保存模型到文件
    # with open(f"som_model-tingche_{l}.pkl", 'wb') as model_file:
    #     pickle.dump(som, model_file)
    # # 加载已保存的SOM模型
    with open(f'som_model-zidong_{l}.pkl', 'rb') as model_file:
        som = pickle.load(model_file)
    som_output_indices = np.array([som.winner(x) for x in subset])
    print(som_output_indices)
    som_output_indices = [som.winner(x) for x in subset]
    som_data_mapping = {}
    for original_index, som_index in enumerate(som_output_indices):
        if som_index not in som_data_mapping:
            som_data_mapping[som_index] = []
        som_data_mapping[som_index].append(original_index)
    for som_index, original_indices in som_data_mapping.items():
        print(f"SOM 神经元索引 {som_index}: 对应的原始数据索引 {original_indices}")
    neuron_samples = {}
    for original_index, som_index in enumerate(som_output_indices):
        if som_index not in neuron_samples:
            neuron_samples[som_index] = []
        neuron_samples[som_index].append(original_index)
    agnes_input = som._weights.reshape(-1, som._weights.shape[2])  # 获取 SOM 输出层的权重矩阵，每行对应一个样本
    Z = linkage(agnes_input, method='complete', metric='euclidean')  # 使用complete linkage和欧氏距离
    max_clusters = 9
    T = fcluster(Z, max_clusters, criterion='maxclust')
    ch_scores = []
    for k in range(2, max_clusters + 1):  # Calinski-Harabasz score在k=2时开始有意义
        cluster_labels = fcluster(Z, k, criterion='maxclust')
        score = calinski_harabasz_score(agnes_input, cluster_labels)
        ch_scores.append(score)
        print(f"Cluster number {k}, Calinski-Harabasz Score: {score}")
    # 找出Calinski-Harabasz得分最高的簇数
    optimal_k = ch_scores.index(max(ch_scores)) + 2  # 加2是因为我们从2开始计数
    print(f"Optimal number of clusters for round {l}: {optimal_k}")
    # 将本轮的optimal_k添加到列表中
    optimal_ks.append(optimal_k)
# 使用Counter统计每个optimal_k出现的次数
optimal_k_counts = Counter(optimal_ks)
# 找出出现次数最多的optimal_k
final_optimal_k = optimal_k_counts.most_common(1)[0][0]
final_optimal_k = 9
print(f"Final optimal number of clusters selected by voting: {final_optimal_k}")
for l in range(4):
    subset_filename = f'subset_{l}.npy'
    subset = np.load(subset_filename)
    som = MiniSom(dimension1, dimension2, subset.shape[1])
    som.train_random(subset, 1000)
    som_output_indices = [som.winner(x) for x in subset]
    som_output_indices = np.array([som.winner(x) for x in subset])
    # 保存模型到文件
    # with open(f"som_model-zidong_{l}.pkl", 'wb') as model_file:
    #     pickle.dump(som, model_file)
    # # 加载已保存的SOM模型
    with open(f'som_model-zidong_{l}.pkl', 'rb') as model_file:
        som = pickle.load(model_file)
    som_output_indices = np.array([som.winner(x) for x in subset])
    print(som_output_indices)
    som_output_indices = [som.winner(x) for x in subset]
    som_data_mapping = {}
    for original_index, som_index in enumerate(som_output_indices):
        if som_index not in som_data_mapping:
            som_data_mapping[som_index] = []
        som_data_mapping[som_index].append(original_index)
    for som_index, original_indices in som_data_mapping.items():
        print(f"SOM 神经元索引 {som_index}: 对应的原始数据索引 {original_indices}")
    neuron_samples = {}
    for original_index, som_index in enumerate(som_output_indices):
        if som_index not in neuron_samples:
            neuron_samples[som_index] = []
        neuron_samples[som_index].append(original_index)
    agnes_input = som._weights.reshape(-1, som._weights.shape[2])  # 获取 SOM 输出层的权重矩阵，每行对应一个样本
    Z = linkage(agnes_input, method='complete', metric='euclidean')  # 使用complete linkage和欧氏距离
    max_clusters = 9
    T = fcluster(Z, max_clusters, criterion='maxclust')
    final_clusters = final_optimal_k
    selected_clusters = fcluster(Z, final_clusters, criterion='maxclust')
    # final_clusters = 9 # 根据需要选择最终聚类结果，例如，选择前 k 个聚类
    # selected_clusters = fcluster(Z, final_clusters, criterion='maxclust')
    print(selected_clusters)
    cluster_indices = [[] for _ in range(final_clusters)]
    for i, cluster in enumerate(selected_clusters):
        cluster_indices[cluster - 1].append(i)
    print("每个聚类类别对应的数据索引:")
    for i, indices in enumerate(cluster_indices):
        print(f"类别 {i + 1}: {indices}")
    som_index_mapping = {
        0: (0, 0),
        1: (0, 1),
        2: (0, 2),
        3: (0, 3),
        4: (0, 4),
        5: (1, 0),
        6: (1, 1),
        7: (1, 2),
        8: (1, 3),
        9: (1, 4),
        10: (2, 0),
        11: (2, 1),
        12: (2, 2),
        13: (2, 3),
        14: (2, 4),
    }
    agnes_indices = {}
    for cluster_id, som_indices in enumerate(cluster_indices):
        agnes_indices[cluster_id] = []
        for som_index in som_indices:
            if som_index in som_index_mapping:
                som_index_tuple = som_index_mapping[som_index]  # 获取对应的元组
                if som_index_tuple in som_data_mapping:
                    agnes_indices[cluster_id].extend(som_data_mapping[som_index_tuple])
    data_to_cluster = {}
    for cluster_id, indices in agnes_indices.items():
        for index in indices:
            data_to_cluster[index] = cluster_id
    # 初始化一个长度为数据点总数的数组，初始值为0
    num_data_points = len(subset)
    cluster_assignment = np.zeros(num_data_points, dtype=int)
    # 填充数组
    for index, cluster_id in data_to_cluster.items():
        cluster_assignment[index] = cluster_id
    # 将层次聚类结果转换为聚类标签
    cluster_labels = np.zeros(num_data_points, dtype=int)
    for j, cluster in enumerate(cluster_assignment):
        if cluster in selected_clusters_per_round[l]:  # 直接检查 cluster 是否在 selected_clusters_per_round[l] 中
            cluster_labels[j] = 1  # 属于选定的聚类
        else:
            cluster_labels[j] = 0  # 不属于选定的聚类
    cluster_labels_result.append((l, cluster_labels))
    #########################################################################
    for cluster_id, som_indices in enumerate(cluster_indices):
        agnes_indices[cluster_id] = []
        for som_index in som_indices:
            if som_index in som_index_mapping:
                som_index_tuple = som_index_mapping[som_index]
                if som_index_tuple in neuron_samples:
                    # 获取该神经元对应的所有样本索引
                    som_samples = neuron_samples[som_index_tuple]
                    # 将这些索引映射回原始数据中的索引
                    original_indices = [subset_indices[f'SOM_{l}'][idx] for idx in som_samples]
                    agnes_indices[cluster_id].extend(original_indices)
    print("每个聚类类别对应的原始数据索引:")
    for i, indices in agnes_indices.items():
        print(f"类别 {i + 1}: {indices}")
    ##########################################
    # 创建一个字典来存储每个神经元代表的原始样本索引
    neuron_samples = {(i, j): [] for i in range(dimension1) for j in range(dimension2)}

    # 根据神经元索引将样本索引分配到相应的神经元中
    for sample_idx, (i, j) in enumerate(som_output_indices):
        neuron_samples[(i, j)].append(sample_idx)

    # 创建一个字典来存储每个聚类包含的神经元
    cluster_neurons = {cluster_label: [] for cluster_label in range(1, final_clusters + 1)}

    # 根据聚类结果将神经元索引分配到相应的聚类中
    for neuron_idx, cluster_label in zip(neuron_samples.keys(), selected_clusters):
        cluster_neurons[cluster_label].append(neuron_idx)
    cluster_neurons_all_rounds.append(cluster_neurons)

# 保存cluster_neurons_all_rounds到文件
with open('cluster_neurons_all_rounds.pkl', 'wb') as file:
    pickle.dump(cluster_neurons_all_rounds, file)
########################################################################################################################################
file_path = 'data.xlsx'
wb = load_workbook(file_path)
ws = wb.active
num = 0
blue_fill = PatternFill(start_color='0000FF', end_color='0000FF', fill_type='solid')
max_value = aggregated_data['elec_consumption'].max()
threshold = max_value * 0.18  # 10% 的阈值
aggregated_data = filtered_data.drop(filtered_data.columns[5:30], axis=1)
aggregated_data = aggregated_data.drop(aggregated_data.columns[2], axis=1)
new_data = []

# 按测量点名称分组
for _, group in aggregated_data.groupby('meter_no_tm'):
    # 确定分组后的数据长度
    group_length = len(group) - 4  # 保证后续索引不会越界
    # 遍历分组中的所有行（考虑到需要连续的五个时间点）
    for i in range(group_length):
        first_date = group.iloc[i]['data_time']
        second_date = group.iloc[i + 1]['data_time']
        third_date = group.iloc[i + 2]['data_time']
        fourth_date = group.iloc[i + 3]['data_time']
        fifth_date = group.iloc[i + 4]['data_time']

        # 检查这五天是否连续
        if (pd.to_datetime(second_date) - pd.to_datetime(first_date)).days == 1 and \
                (pd.to_datetime(third_date) - pd.to_datetime(second_date)).days == 1 and \
                (pd.to_datetime(fourth_date) - pd.to_datetime(third_date)).days == 1 and \
                (pd.to_datetime(fifth_date) - pd.to_datetime(fourth_date)).days == 1:
            # 提取对应的电量消耗数据
            first_value = group.iloc[i]['elec_consumption']
            second_value = group.iloc[i + 1]['elec_consumption']
            third_value = group.iloc[i + 2]['elec_consumption']
            fourth_value = group.iloc[i + 3]['elec_consumption']
            fifth_value = group.iloc[i + 4]['elec_consumption']

            # 添加到 new_data 列表
            new_data.append([
                group.iloc[i]['meter_no_tm'],
                first_date,
                first_value,
                second_value,
                third_value,
                fourth_value,
                fifth_value
            ])

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
# 加载训练好的 SOM 模型
soms = []
for l in range(4):
    with open(f'som_model-zidong_{l}.pkl', 'rb') as model_file:
        som = pickle.load(model_file)
        soms.append(som)

# 初始化计数器
index_counts = np.zeros(len(data_normalized))
consensus_matrix = np.zeros((len(data_normalized), len(data_normalized)))
agnes_indices_all_rounds = []
# 将新数据映射到 SOM 网格并分配到聚类
for l in range(4):
    index_counts_num = np.zeros(len(data_normalized))
    som = soms[l]
    cluster_ids_to_select = selected_clusters_per_round[l]  # 当前轮次选择的聚类组
    som_output_indices = [som.winner(x) for x in data_normalized]
    som_data_mapping = {}
    for original_index, som_index in enumerate(som_output_indices):
        if som_index not in som_data_mapping:
            som_data_mapping[som_index] = []
        som_data_mapping[som_index].append(original_index)
    neuron_samples = {}
    for original_index, som_index in enumerate(som_output_indices):
        if som_index not in neuron_samples:
            neuron_samples[som_index] = []
        neuron_samples[som_index].append(original_index)
    agnes_input = som._weights.reshape(-1, som._weights.shape[2])  # 获取 SOM 输出层的权重矩阵，每行对应一个样本
    Z = linkage(agnes_input, method='complete', metric='euclidean')  # 使用complete linkage和欧氏距离
    max_clusters = 9
    T = fcluster(Z, max_clusters, criterion='maxclust')
    final_clusters = 9  # 根据需要选择最终聚类结果，例如，选择前 k 个聚类
    selected_clusters = fcluster(Z, final_clusters, criterion='maxclust')
    cluster_indices = [[] for _ in range(final_clusters)]
    for i, cluster in enumerate(selected_clusters):
        cluster_indices[cluster - 1].append(i)
    som_index_mapping = {
        0: (0, 0),
        1: (0, 1),
        2: (0, 2),
        3: (0, 3),
        4: (0, 4),
        5: (1, 0),
        6: (1, 1),
        7: (1, 2),
        8: (1, 3),
        9: (1, 4),
        10: (2, 0),
        11: (2, 1),
        12: (2, 2),
        13: (2, 3),
        14: (2, 4),
    }
    agnes_indices = {}
    for cluster_id, som_indices in enumerate(cluster_indices):
        agnes_indices[cluster_id] = []
        for som_index in som_indices:
            if som_index in som_index_mapping:
                som_index_tuple = som_index_mapping[som_index]  # 获取对应的元组
                if som_index_tuple in som_data_mapping:
                    agnes_indices[cluster_id].extend(som_data_mapping[som_index_tuple])
    agnes_indices_all_rounds.append(agnes_indices)
    for cluster_id in cluster_ids_to_select:  # 你感兴趣的 cluster_id
        if cluster_id in agnes_indices:
            for index in agnes_indices[cluster_id]:
                # 获取样本数据，假设数据在 new_data 的 DataFrame 中
                # sample_data = new_data.iloc[index]
                index_counts[index] += 1
                index_counts_num[index] += 1
    for d in range(len(index_counts)):
        for k in range(len(index_counts)):
            if index_counts_num[d] == 1:
                if index_counts_num[d] == index_counts_num[k] :
                    consensus_matrix[d][k] += 1

consensus_matrix /= 4
n = 4  #
kmeans_final = KMeans(n_clusters=n, random_state=42)
final_clusters = kmeans_final.fit_predict(consensus_matrix)
# 保存 KMeans 模型
with open('kmeans_model.pkl', 'wb') as model_file:
    pickle.dump(kmeans_final, model_file)
# 保存最终聚类结果
np.save('final_clusters.npy', final_clusters)
print("Final clusters:", final_clusters)

# 加载 Excel 文件
file_path = 'data.xlsx'
wb = load_workbook(file_path)
ws = wb.active
# 定义填充样式
blue_fill = PatternFill(start_color='0000FF', end_color='0000FF', fill_type='solid')
max_value = aggregated_data['elec_consumption'].max()
threshold = max_value * 0.06  # 10% 的阈值
# 根据聚类结果标记单元格
num = 0
for i in range(len(final_clusters)):
    cluster_label = final_clusters[i]
    if cluster_label in [1]:  # 只标记第1、2、3类
        sample_data = new_data.iloc[i]
        data_time = sample_data[1]  # 假设这是 pandas Timestamp 对象
        ent_name = sample_data[0]  # 企业名称
        data_time = pd.to_datetime(data_time)
        date_range = pd.date_range(data_time, periods=5)
        # 检查这五个连续的数是否都小于最大值的10%
        all_below_threshold = all(
            # aggregated_data.iloc[index + offset]['elec_consumption'] < threshold
            aggregated_data.iloc[index + offset][3] < threshold
            for offset in range(5)
        )

        if all_below_threshold:
            for target_date in date_range:
                for row in range(2, ws.max_row + 1):  # 从第2行开始遍历（假设第1行为表头）
                    cell_date = ws.cell(row=row, column=2).value
                    cell_ent_name = ws.cell(row=row, column=4).value
                    # 将 Excel 文件中的日期转换为 pandas Timestamp
                    if isinstance(data_time, datetime):
                        cell_date = pd.Timestamp(cell_date)
                    if cell_date == target_date and cell_ent_name == ent_name:
                        ws.cell(row=row, column=2).fill = blue_fill
                        num += 1
                        break  # 找到并标记后跳出循环
# 保存更改
wb.save(file_path)
print(f"标记完成，共标记了 {num} 个单元格，Excel 文件已保存。")

# 定义选择的聚类组
selected_clusters_per_round =[[0,1,2,3,4,5,6,7,8], [0,1,2,3,4,5,6,7,8], [0,1,2,3,4,5,6,7,8], [0,1,2,3,4,5,6,7,8]] #[[5], [0], [0], [4]] #[[2], [6], [1,5], [6]]

# 初始化存储概率的字典
probabilities = {}

# 遍历每一轮次
for l in range(4):
    cluster_ids_to_select = selected_clusters_per_round[l]
    probabilities[l] = {}

    # 提取每个选定聚类中的样本索引
    for cluster_id in cluster_ids_to_select:
        indices = agnes_indices_all_rounds[l].get(cluster_id, [])
        # indices = agnes_indices.get(cluster_id, [])

        # 统计这些样本在 final_clusters 中属于 [2], [3], [4] 类的频次
        final_cluster_counts = Counter(final_clusters[indices])

        # 计算概率
        total_count = len(indices)
        if total_count > 0:
            probabilities[l][cluster_id] = {
                0: final_cluster_counts[0] / total_count,
                1: final_cluster_counts[1] / total_count,
                2: final_cluster_counts[2] / total_count,
                3: final_cluster_counts[3] / total_count
            }
        else:
            probabilities[l][cluster_id] = {
                0: 0.0,
                1: 0.0,
                2: 0.0,
                3: 0.0
            }

# 打印结果
for round, clusters in probabilities.items():
    print(f"Round {round}:")
    for cluster_id, probs in clusters.items():
        print(f"  Cluster {cluster_id}:")
        print(f"    Probability of being in cluster 0: {probs[0]:.4f}")
        print(f"    Probability of being in cluster 1: {probs[1]:.4f}")
        print(f"    Probability of being in cluster 2: {probs[2]:.4f}")
        print(f"    Probability of being in cluster 3: {probs[3]:.4f}")
# 计算轮次数
num_rounds = len(selected_clusters_per_round)

# 计算最大聚类数
max_clusters = max(max(selected_clusters_per_round, key=lambda x: max(x))) + 1

# 计算最终聚类数
n_clusters = len(set(final_clusters))

total = [[[0.0] * n_clusters for _ in range(max_clusters)] for _ in range(4)]
# 初始化 sum 列表
sum = [[0.0] * n_clusters for _ in range(max_clusters)]
# 打印结果
for round, clusters in probabilities.items():
    print(f"Round {round}:")
    for cluster_id, probs in clusters.items():
        total[round][cluster_id][0] += probs[0]
        total[round][cluster_id][1] += probs[1]
        total[round][cluster_id][2] += probs[2]
        total[round][cluster_id][3] += probs[3]
        sum[cluster_id][0] += probs[0]
        sum[cluster_id][1] += probs[1]
        sum[cluster_id][2] += probs[2]
        sum[cluster_id][3] += probs[3]
        print(f"  Cluster {cluster_id}:")
        print(f"    Probability of being in cluster 0: {probs[0]:.4f}")
        print(f"    Probability of being in cluster 1: {probs[1]:.4f}")
        print(f"    Probability of being in cluster 2: {probs[2]:.4f}")
        print(f"    Probability of being in cluster 3: {probs[3]:.4f}")

ave = [[[0.0] * n_clusters for _ in range(max_clusters)] for _ in range(4)]


for round, clusters in probabilities.items():
    for cluster_id, probs in clusters.items():
        for i in range(4):
            if sum[cluster_id][i] > 0:
                ave[round][cluster_id][i] = total[round][cluster_id][i] / sum[cluster_id][i]
            else:
                ave[round][cluster_id][i] = 0.0

for round, clusters in probabilities.items():
    print(f"Round {round}:")
    for cluster_id, probs in clusters.items():
        print(f"    Cluster {cluster_id}:")
        for i in range(4):
            print(f"    Probability of being in cluster {i}: {ave[round][cluster_id][i]:.4f}")


# 将参数保存到文件
np.savez('parameters.npz', num_rounds=num_rounds, max_clusters=max_clusters, n_clusters=n_clusters)
# 保存 ave 值到文件
np.save('ave_values.npy', ave)
