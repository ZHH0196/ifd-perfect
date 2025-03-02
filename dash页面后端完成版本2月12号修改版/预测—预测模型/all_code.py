import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# 设置 matplotlib 使用中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 定义文件夹路径
folder_path = 'Bearing1_3'

# 初始化一个列表，用于存储每个文件的水平方向振动信号
all_signals = []

# 初始化一个列表，用于存储每个文件的均方根值
rms_values = []

# 初始化一个列表，用于存储每个文件的整流平均值
rmv_values = []

# 遍历文件夹中的所有 CSV 文件
for file_name in sorted(os.listdir(folder_path)):
    if file_name.endswith('.csv'):
        # 构建每个文件的完整路径
        file_path = os.path.join(folder_path, file_name)
        
        # 读取 CSV 文件，假设不包含表头
        data = pd.read_csv(file_path, header=None)
        
        # 提取第五列（第4列索引为3）作为水平方向的振动信号
        horizontal_signal = data.iloc[:, 4]
        
        # 将该文件的信号添加到 all_signals 列表中
        all_signals.append(horizontal_signal)
        # 计算均方根值 (RMS)
        rms = np.sqrt(np.mean(np.square(horizontal_signal)))
        
        # 将RMS值添加到 rms_values 列表中
        rms_values.append(rms)

        # 计算整流平均值 (RMV)
        rmv = np.mean(np.abs(horizontal_signal))
        
        # 将RMV值添加到 rmv_values 列表中
        rmv_values.append(rmv)
        
# 创建一个新的图形
plt.figure(figsize=(10, 6))
# 将所有的振动信号拼接成一个大的时间序列
flattened_signals = pd.concat(all_signals, ignore_index=True)
# 绘制振动信号的时间序列图
plt.plot(flattened_signals, color='blue', linewidth=0.5)
# 设置标题和标签
plt.title('水平振动信号')
plt.xlabel('样本数')
plt.ylabel('振动幅值')
# 设置纵轴范围为 -50 到 50
plt.ylim([-60, 60])
# 设置横轴范围为 0 到 2000个样本（假设每个样本对应一个文件）
plt.xlim([0, 2000 * 2560])
# 设置横轴的刻度为每 500 个样本标注一次
plt.xticks(range(0, len(all_signals) * 2560, 500 * 2560), range(0, 2000, 500))
# 取消网格线
plt.grid(True)
plt.savefig('原始数据时间序列.png')
# 显示图形
plt.close()

# 创建一个新的图形
plt.figure(figsize=(10, 6))
# 绘制均方根值的时间序列图
plt.plot(rms_values, color='blue', linewidth=0.5)
# 设置标题和标签
plt.title('水平振动信号的均方根值 (RMS)')
plt.xlabel('样本数')
plt.ylabel('RMS 值')
# 设置纵轴范围根据RMS的值自动调整
plt.ylim([min(rms_values) - 0.1, max(rms_values) + 0.1])
# 设置横轴范围为样本数（假设每个文件对应一个样本）
plt.xlim([0, 2000])
# 设置横轴的刻度为每 500 个样本标注一次
plt.xticks(range(0, len(rms_values), 500), range(0, 2000, 500))
# 取消网格线
plt.grid(True)
# 保存图形为PNG文件
plt.savefig('均方根值时间序列.png')
# 显示图形
plt.close()

# 创建一个新的图形
plt.figure(figsize=(10, 6))
# 绘制整流平均值的时间序列图
plt.plot(rmv_values, color='blue', linewidth=0.5)
# 设置标题和标签
plt.title('水平振动信号的整流平均值 (RMV)')
plt.xlabel('样本数')
plt.ylabel('RMV 值')
# 设置纵轴范围根据RMV的值自动调整
plt.ylim([min(rmv_values) - 0.1, max(rmv_values) + 0.1])
# 设置横轴范围为样本数（假设每个文件对应一个样本）
plt.xlim([0, 2000])
# 设置横轴的刻度为每 500 个样本标注一次
plt.xticks(range(0, len(rmv_values), 500), range(0, 2000, 500))
# 取消网格线
plt.grid(True)
# 保存图形为PNG文件
plt.savefig('整流平均值时间序列.png')
# 显示图形
plt.close()

rms_values = np.array(rms_values)
rmv_values = np.array(rmv_values)

# 计算 Sigmoid 变换
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 对均方根值和整流平均值应用 Sigmoid 变换
sigmoid_rms = sigmoid(rms_values)
sigmoid_arv = sigmoid(rmv_values)

# 生成退化指标
degradation_index = (sigmoid_rms + sigmoid_arv) / 2

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

sigmoid_values = (normalize(degradation_index) * 5)-1 

# 创建一个新的图形
plt.figure(figsize=(10, 6))
# 绘制均方根值的时间序列图
plt.plot(sigmoid_values, color='blue', linewidth=0.5)
# 设置标题和标签
plt.title('Sigmoid退化指标')
plt.xlabel('样本数')
plt.ylabel('退化指标值')
# 设置纵轴范围根据RMS的值自动调整
plt.ylim([-1, 5])
# 设置横轴范围为样本数（假设每个文件对应一个样本）
plt.xlim([0, 2000])
# 设置横轴的刻度为每 500 个样本标注一次
plt.xticks(range(0, len(sigmoid_values), 500), range(0, 2000, 500))
# 取消网格线
plt.grid(True)
# 保存图形为PNG文件
plt.savefig('Sigmoid退化指标.png')
# 显示图形
plt.close()



# 计算移动平均和 ts_rank
def ts_rank(signal, window=1):
    moving_average = pd.Series(signal).rolling(window).mean()
    rank = moving_average.rank(ascending=False)
    return rank.fillna(0).values

ts_rms = -ts_rank(rms_values)
ts_arv = -ts_rank(rmv_values)


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
ts_rms_normalize = normalize(ts_rms)
ts_rmv_normalize = normalize(ts_arv)
ts_rank_values = 2*(ts_rms_normalize + ts_rmv_normalize)-2
# 绘制序列图
plt.figure(figsize=(10, 6))
plt.plot(ts_rank_values, color='blue', linewidth=0.5)
plt.title('ts_rank退化指标序列图')
plt.xlabel('样本数')
plt.ylim([-2, 2])
plt.ylabel('指标值')
plt.xlim([0,2000])
plt.xticks(range(0, len(ts_rank_values), 500), range(0, 2000, 500))
plt.grid(True)
plt.savefig('ts_rank退化指标.png')
plt.close()

CDI = ts_rank_values+sigmoid_values
# 绘制序列图
plt.figure(figsize=(10, 6))
plt.plot(CDI, color='blue', linewidth=0.5)
plt.title('CDI 退化指标序列图')
plt.xlabel('样本数')
plt.ylim([-3, 6])
plt.ylabel('CDI指标值')
plt.xlim([0,2000])
plt.xticks(range(0, len(CDI), 500), range(0, 2000, 500))
plt.grid(True)
plt.savefig('CDI退化指标.png')
plt.close()

# 使用移动平均法进行平滑处理
def moving_average(signal, window_size):
    return pd.Series(signal).rolling(window=window_size, center=False).mean()

# 使用指数加权移动平均法进行平滑处理
def exponential_moving_average(signal, span):
    return pd.Series(signal).ewm(span=span, adjust=False).mean()

span_value = 10  # 指数加权移动平均的跨度

# 指数加权移动平均平滑
CDI = exponential_moving_average(CDI, span_value)


# 绘制指数加权移动平均后的 CDI 数据
plt.plot(CDI, color='blue', linewidth=0.5, label='指数加权移动平均平滑')

# 设置图形标题和标签
plt.title('CDI 退化指标序列图（平滑处理）')
plt.xlabel('样本数')
plt.ylabel('CDI 指标值')
plt.ylim([-3, 6])
plt.xlim([0, 2000])

# 设置横轴的刻度
plt.xticks(range(0, len(CDI), 500), range(0, 2000, 500))

# 显示网格
plt.grid(True)

# 添加图例
plt.legend()

# 保存图像
plt.savefig('CDI_退化指标_平滑处理.png')

# 显示图形
plt.close()


normal_data = np.array(CDI)

# 时间序列
time_series = np.arange(len(normal_data))

# =========================
# Step 2: 检测数据是否为平稳状态
# =========================

# 阈值，判断是否开始退化 (可以根据实际数据调整)
threshold = 0.15
is_degrading = np.any(normal_data > threshold)

if not is_degrading:
    print("检测到轴承处于正常状态，暂时没有明显退化迹象，预测轴承仍处于健康状态。")
    print("在当前状态下，剩余使用寿命 (RUL) 可能较长，暂时无需维修。")
else:
    # =========================
    # Step 3: 使用维纳过程和卡尔曼滤波预测RUL
    # =========================

    # 定义维纳过程参数
    def wiener_process(time, drift, diffusion, x0=0):
        """
        维纳过程模拟
        :param time: 时间序列
        :param drift: 漂移系数
        :param diffusion: 扩散系数
        :param x0: 初始值
        :return: 退化过程
        """
        dt = np.diff(time)  # 时间间隔
        x = np.zeros(len(time))
        x[0] = x0

        for i in range(1, len(time)):
            x[i] = x[i - 1] + drift * dt[i - 1] + diffusion * np.sqrt(dt[i - 1]) * np.random.randn()

        return x

    # 模拟维纳过程
    drift = 0.05  # 漂移系数
    diffusion = 0.02  # 扩散系数
    wiener_data = wiener_process(time_series, drift, diffusion)

    # =========================
    # Step 4: 自适应卡尔曼滤波
    # =========================

    class AdaptiveKalmanFilter:
        def __init__(self, A, B, C, Q, R, P, x0):
            """
            自适应卡尔曼滤波器
            :param A: 系统矩阵
            :param B: 控制矩阵
            :param C: 观测矩阵
            :param Q: 过程噪声协方差
            :param R: 观测噪声协方差
            :param P: 初始误差协方差
            :param x0: 初始状态
            """
            self.A = A  # 状态转移矩阵
            self.B = B  # 控制矩阵
            self.C = C  # 观测矩阵
            self.Q = Q  # 过程噪声协方差
            self.R = R  # 观测噪声协方差
            self.P = P  # 误差协方差
            self.x = x0  # 初始状态

        def predict(self, u=0):
            # 预测阶段
            self.x = np.dot(self.A, self.x) + np.dot(self.B, u)
            self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

        def update(self, z):
            # 更新阶段
            y = z - np.dot(self.C, self.x)  # 计算残差
            S = np.dot(self.C, np.dot(self.P, self.C.T)) + self.R  # 残差协方差
            K = np.dot(np.dot(self.P, self.C.T), np.linalg.inv(S))  # 卡尔曼增益

            self.x = self.x + np.dot(K, y)  # 更新状态估计
            self.P = self.P - np.dot(K, np.dot(self.C, self.P))  # 更新误差协方差

            return self.x

    # 初始化卡尔曼滤波器的参数
    A = np.array([[1]])  # 状态转移矩阵
    B = np.array([[0]])  # 控制矩阵
    C = np.array([[1]])  # 观测矩阵
    Q = np.array([[0.01]])  # 过程噪声协方差
    R = np.array([[0.1]])  # 观测噪声协方差
    P = np.array([[1]])  # 初始误差协方差
    x0 = np.array([normal_data[0]])  # 初始状态 (使用CDI的第一个值作为初始状态)

    # 创建卡尔曼滤波器
    kf = AdaptiveKalmanFilter(A, B, C, Q, R, P, x0)

    # =========================
    # Step 5: 预测轴承剩余使用寿命 (RUL)
    # =========================

    # 进行卡尔曼滤波预测
    rul_predictions = []
    for z in normal_data:
        kf.predict()
        predicted_state = kf.update(z)
        rul_predictions.append(predicted_state[0])

    # =========================
    # Step 6: 结果可视化
    # =========================

    # 绘制CDI数据与RUL预测结果
    plt.figure(figsize=(10, 6))
    plt.plot(time_series, normal_data, label='CDI')
    plt.plot(time_series, rul_predictions, label='剩余寿命预测')
    plt.xlabel('时间')
    plt.ylabel('CDI与RUL')
    plt.title('CDI与RUL预测')
    plt.legend()
    plt.grid(True)
    plt.savefig('CDI_RUL_预测.png')
    plt.close()

    # 输出最后时刻的RUL预测值
    print(f"最后时刻的轴承剩余使用寿命 (RUL) 预测值: {rul_predictions[-1]}")
    
    # =========================
# Step 7: 通过线性外推预测RUL
# =========================

# 设定一个退化的故障阈值
failure_threshold = 5.8

  # 计算当前的退化速度（简单的差分）
degradation_rate = (rul_predictions[-1] - rul_predictions[0]) / len(rul_predictions)

 # 每个样本代表的时间长度是 10 秒
time_per_sample = 10  # 每个样本的时间间隔为 10 秒

    # 如果退化速率为正，我们可以通过线性外推预测轴承达到故障阈值的剩余时间
if degradation_rate > 0:
        # 剩余时间 (秒)
    remaining_life_seconds = (failure_threshold - rul_predictions[-1]) / degradation_rate * time_per_sample

        # 将剩余时间转换为小时
    remaining_life_hours = remaining_life_seconds / 3600

    print(f"预测的剩余使用寿命 (RUL): {remaining_life_hours} 小时")
else:
    print("无法预测RUL，退化率为零或负值。")