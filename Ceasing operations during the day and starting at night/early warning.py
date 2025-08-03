excelFileName = 'data.xlsx'
dataRange = 'A:AC'
startRow = 0
# 读取数据
data = pd.read_excel(excelFileName, usecols=dataRange, skiprows=startRow)
# 过滤第 C 列为"责令停产"的数据
filtered_data = data.loc[data['ent_status'] == 'Order to suspend production status']
columns_to_check = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10',
                   'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'h19',
                   'h20', 'h21', 'h22', 'h23', 'h24']
filtered_data = filtered_data.dropna(subset=columns_to_check, how='any')
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
workbook.save('昼停夜开.xlsx')
print("新的 Excel 文件已保存。")

df = filtered_data.drop(filtered_data.columns[[0,1,2,3,4]], axis = 1)

scaler = MinMaxScaler(feature_range=(0, 1))
# 对每一行数据进行归一化
data_preprocessed = scaler.fit_transform(df)

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            # nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class ElectricityDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

# 假设您的输入数据是一个 24 x 1000 的矩阵
input_data = torch.from_numpy(data_preprocessed).float()

# 创建数据集和数据加载器
dataset = ElectricityDataset(input_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 定义模型、优化器和损失函数
model = Autoencoder(input_size=24, hidden_size=24)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    for data in dataloader:
        recon = model(data)
        loss = criterion(recon, data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

data_normalized = model(input_data).detach().numpy()
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
    with open(f'som_model-zhoutingyekai_{l}.pkl', 'rb') as model_file:
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

blue_fill = PatternFill(start_color='0000FF', end_color='0000FF', fill_type='solid')

# 加载原始Excel文件以便修改
wb = load_workbook('data.xlsx')
ws = wb.active

# 计算最终概率并汇总，同时检查条件
rows_to_highlight = []
for idx, probabilities in enumerate(final_probabilities_all):
    if probabilities is not None:
        # 检查最终概率条件 (假设类别索引从0开始)
        if sum(probabilities[[0, 1]]) < sum(probabilities[[2]]):
            rows_to_highlight.append(idx + 1)

# 遍历要高亮的行，并在Excel中标记它们
for row in rows_to_highlight:
    ws.cell(row=row, column=1).fill = blue_fill  # 只对每行的第一列应用蓝色填充
    # for cell in ws[row]:
    #     cell.fill = blue_fill

# 保存修改后的Excel文件
wb.save('highlighted.xlsx')
print("已根据条件标蓝相关行。")