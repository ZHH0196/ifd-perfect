import pandas as pd
import numpy as np
from keras import models
import os

# 新数据预处理函数 (确保数据符合模型的输入格式)
def PreprocessNewData(new_data_df, samples_per_block):
    # 提取时间序列数据 (假设在第一列)
    new_data_series = new_data_df.iloc[:, 0]  # 提取时间序列数据

    # 确保数据长度足够
    if len(new_data_series) < samples_per_block:
        raise ValueError(f"新数据长度不足，必须至少包含 {samples_per_block} 个数据点")

    # 将数据调整为模型期望的形状 (1, 样本长度, 1)
    processed_data = new_data_series[:samples_per_block].values.reshape(1, samples_per_block, 1)

    return processed_data

# 新数据检测函数
def DetectNewDataFromDataFrame(new_data_df, model_path, samples_per_block, state_labels):
    
    # 加载保存的模型
    model = models.load_model(model_path)

    # 预处理新数据
    processed_data = PreprocessNewData(new_data_df, samples_per_block)

    # 使用模型进行预测
    predictions = model.predict(processed_data)

    # 获取预测结果（取最大概率对应的类别索引）
    predicted_label_idx = np.argmax(predictions, axis=1)[0]

    # 打印预测的故障状态标签
    print(f"预测的故障状态标签: {state_labels[predicted_label_idx]}")


new_data_df = pd.read_csv('驱动端外圈右边0.007断层.csv')

# 设置参数
samples_per_block = 1681  
model_path = "12K2马力1750电机速驱动端.h5"  # 模型

# 训练时的状态标签列表，按顺序列出
state_labels = ['正常状态驱动端', '驱动端内圈0.007断层', '驱动端内圈0.014断层', '驱动端内圈0.021断层', '驱动端内圈0.028断层', '驱动端外圈中心0.007断层', '驱动端外圈中心0.014断层', '驱动端外圈中心0.021断层', '驱动端外圈右边0.007断层', '驱动端外圈右边0.021断层', '驱动端外圈左边0.007断层', '驱动端外圈左边0.021断层', '驱动端球体0.007断层', '驱动端球体0.014断层', '驱动端球体0.021断层', '驱动端球体0.028断层']
# 对新数据进行故障检测
DetectNewDataFromDataFrame(new_data_df, model_path, samples_per_block, state_labels)