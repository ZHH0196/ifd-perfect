import os
import pathlib
import dash
from dash import dash_table, Input, Output, State, html, dcc, callback,Dash
import plotly.graph_objs as go
import dash_daq as daq
import dash_mantine_components as dmc
import pandas as pd
import plotly.express as px
import io
import base64
from keras import models
import numpy as np
from flask import Flask
import time
from openai import OpenAI
server = Flask(__name__)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 配置通义千问API客户端
client = OpenAI(
    api_key="sk-977e9f40cc884b79a37f9071e54acde9",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# AI分析函数
def get_ai_analysis(current_state, prediction_str, data_features):
    prompt = f"""# 轴承故障诊断分析报告

## 基本信息
- 当前状态：{current_state}
- 预测结果：{prediction_str}

## 数据分析
- 数据特征：
{data_features}

请作为专业的轴承故障诊断专家，从以下几个方面进行分析：

1. 当前轴承的健康状况评估
2. 可能存在的问题和潜在风险
3. 建议采取的维护措施
4. 预防性维护建议

请用专业的工程师语言给出分析结果。"""

    try:
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"AI分析生成失败: {str(e)}"
app = dash.Dash(
    __name__,
    server=server,
    url_base_pathname='/dashboard/',
    suppress_callback_exceptions=True, # 允许动态回调
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Industrial Fault Detection Dashboard"
app.config["suppress_callback_exceptions"] = True

APP_PATH = str(pathlib.Path(__file__).parent.resolve())
# df = pd.read_csv(os.path.join(APP_PATH, os.path.join("data", "spc_data.csv"))) # 修改为上传数据

# 页头编辑
def build_banner():
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.Div(
                id="banner-text",
                children=[
                    html.H5("工业故障检测与寿命预测仪表盘"),
                    html.H6("实时监控工业产品健康状况"),
                ],
            ),
        ],
    )

# 标签编辑
def build_tabs():
    return html.Div(
        id="tabs",
        className="tabs",
        children=[
            dcc.Tabs(
                id="app-tabs",
                value="tab2",
                className="custom-tabs",
                children=[
                    dcc.Tab(
                        id="pre-tab",
                        label="故障检测",
                        value="tab2",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                    dcc.Tab(
                        id="throw-tab",
                        label="评定失效阈值",
                        value="tab3",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                    dcc.Tab(
                        id="life-tab",
                        label="寿命检测",
                        value="tab4",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                ],
            )
        ],
    )

# 构建参数设置模块
def build_quick_stats_panel(multiple=True):
    return html.Div(
        id="quick-stats",
        className="row",
        children=[
            # 添加型号选择模块
            html.Div(
                id="model-select-container",
                children=[
                    html.Label(
                        "选择轴承型号",
                        style={
                            "fontSize": "16px",
                            "marginBottom": "10px",
                            "display": "block",
                            "color": "white",
                        },
                    ),
                    dcc.Dropdown(
                        id="model-select-dropdown",
                        options=[
                            {"label": "12K0马力1797电机速驱动端", "value": "12K0马力1797电机速驱动端"},
                            {"label": "12K0马力1797电机速风扇端", "value": "12K0马力1797电机速风扇端"},
                            {"label": "12K1马力1772电机速驱动端", "value": "12K1马力1772电机速驱动端"},
                            {"label": "12K1马力1772电机速风扇端", "value": "12K1马力1772电机速风扇端"},
                            {"label": "12K2马力1750电机速驱动端", "value": "12K2马力1750电机速驱动端"},
                            {"label": "12K2马力1750电机速风扇端", "value": "12K2马力1750电机速风扇端"},
                            {"label": "12K3马力1730电机速驱动端", "value": "12K3马力1730电机速驱动端"},
                            {"label": "12K3马力1730电机速风扇端", "value": "12K3马力1730电机速风扇端"},
                        ],
                        placeholder="请选择型号",

                    ),
                ],
                style={
                    "width": "100%",
                    "padding": "10px",
                    "marginBottom": "20px",
                    "backgroundColor": "#2e2f3f",
                    "borderRadius": "5px",
                    "boxShadow": "0px 2px 4px rgba(0, 0, 0, 0.2)",
                },
            ),
            html.Div(
                id="card-1",
                children=[
                    html.P("请上传文件",                        
                    style={
                            "fontSize": "16px",
                            "marginBottom": "10px",
                            "display": "block",
                            "color": "white",
                        },),
                    
                    dcc.Upload(
                        id='upload-data',
                        children=html.Button('拖拽或点击上传文件'),
                        style={
                            'width': '100%',
                            'height': '75px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        },
                        multiple=multiple  # Allow multiple files to be uploaded
                    ),
                ],
                style={
                    "width": "100%",
                    "padding": "30px",
                    "marginBottom": "20px",
                    "backgroundColor": "#2e2f3f",
                    "borderRadius": "5px",
                    "boxShadow": "0px 2px 4px rgba(0, 0, 0, 0.2)",
                },
            ),
            html.Div(
                id="card-2",
                children=[
                    html.P("故障状态标签"),
                    html.Div(id="output-lab"), 
                ],
                style={
                    "width": "100%",
                    "padding": "30px",
                    "marginBottom": "20px",
                    "backgroundColor": "#2e2f3f",
                    "borderRadius": "5px",
                    "boxShadow": "0px 2px 4px rgba(0, 0, 0, 0.2)",
                },
            ),
            
        ],
    )

# 构建参数设置模块
def build_quick_stats_panel3(multiple=True):
    return html.Div(
        id="quick-stats",
        className="row",
        children=[
            html.Div(
                id="card-1",
                children=[
                    html.P("请上传文件",                        
                    style={
                            "fontSize": "16px",
                            "marginBottom": "10px",
                            "display": "block",
                            "color": "white",
                        },),
                    
                    dcc.Upload(
                        id='upload-data3',
                        children=html.Button('拖拽或点击上传文件'),
                        style={
                            'width': '100%',
                            'height': '75px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        },
                        multiple=multiple  # Allow multiple files to be uploaded
                    ),
                ],
                style={
                    "width": "100%",
                    "padding": "30px",
                    "marginBottom": "20px",
                    "backgroundColor": "#2e2f3f",
                    "borderRadius": "5px",
                    "boxShadow": "0px 2px 4px rgba(0, 0, 0, 0.2)",
                },
            ),
            
        ],
    )

# 构建参数设置模块
def build_quick_stats_panel4(multiple=True):
    return html.Div(
        id="quick-stats",
        className="row",
        children=[
            html.Div(
                children=[dcc.Input(id="value-name-input", type='text', placeholder='输入该轴承的失效阈值')],
                style={
                    "width": "100%",
                    "padding": "10px",
                    "marginBottom": "20px",
                    "backgroundColor": "#2e2f3f",
                    "borderRadius": "5px",
                    "boxShadow": "0px 2px 4px rgba(0, 0, 0, 0.2)",
                },
            ),
            html.Div(
                id="card-1",
                children=[
                    html.P("请上传文件",                        
                    style={
                            "fontSize": "16px",
                            "marginBottom": "10px",
                            "display": "block",
                            "color": "white",
                        },),
                    
                    dcc.Upload(
                        id='upload-data4',
                        children=html.Button('拖拽或点击上传文件'),
                        style={
                            'width': '100%',
                            'height': '75px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        },
                        multiple=multiple  # Allow multiple files to be uploaded
                    ),
                ],
                style={
                    "width": "100%",
                    "padding": "30px",
                    "marginBottom": "20px",
                    "backgroundColor": "#2e2f3f",
                    "borderRadius": "5px",
                    "boxShadow": "0px 2px 4px rgba(0, 0, 0, 0.2)",
                },
            ),
            
        ],
    )

# 生成顶部模块
def generate_section_banner(title):
    return html.Div(className="section-banner", children=title)

# 构建顶部模块
def build_top_panel(stopped_interval):
    return html.Div(
        id="top-section-container",
        className="row",
        children=[
            # Metrics summary
            html.Div(
                id="metric-summary-session",
                className="eight columns",
                children=[
                    generate_section_banner("该轴承数据的时间序列图"),
                    dcc.Graph(id='metric-rows')
                ],
            ),
            # Piechart
            html.Div(
                id="ooc-piechart-outer",
                className="four columns",
                children=[
                    generate_section_banner("选择型号的训练模型混淆矩阵"),
                    html.Div(id="img_opt"),
                ],
            ),
        ],
    )

img_options={
        "12K0马力1797电机速风扇端": "assets/12K0马力1797电机速风扇端.png",
        "12K0马力1797电机速驱动端": "assets/12K0马力1797电机速驱动端.png",
        "12K1马力1772电机速风扇端": "assets/12K1马力1772电机速风扇端.png",
        "12K1马力1772电机速驱动端": "assets/12K1马力1772电机速驱动端.png",
        "12K2马力1750电机速风扇端": "assets/12K2马力1750电机速风扇端.png",
        "12K2马力1750电机速驱动端": "assets/12K2马力1750电机速驱动端.png",
        "12K3马力1730电机速风扇端": "assets/12K3马力1730电机速风扇端.png",
        "12K3马力1730电机速驱动端": "assets/12K3马力1730电机速驱动端.png",
}

text_options={
        "12K0马力1797电机速风扇端": ['风扇端内圈0.007断层', '风扇端内圈0.014断层', '风扇端内圈0.021断层', '风扇端外圈中心0.007断层', '风扇端外圈中心0.014断层', '风扇端外圈右边0.007断层', '风扇端外圈左边0.007断层', '风扇端外圈左边0.014断层', '风扇端外圈左边0.021断层', '风扇端正常状态', '风扇端球体0.007断层', '风扇端球体0.014断层', '风扇端球体0.021断层'],
        "12K0马力1797电机速驱动端": ['驱动端内圈0.007断层', '驱动端内圈0.014断层', '驱动端内圈0.021断层', '驱动端内圈0.028断层', '驱动端外圈中心0.007断层', '驱动端外圈中心0.014断层', '驱动端外圈中心0.021断层', '驱动端外圈右边0.007断层', '驱动端外圈右边0.021断层', '驱动端外圈左边0.007断层', '驱动端外圈左边0.021断层', '驱动端正常状态', '驱动端球体0.007断层', '驱动端球体0.014断层', '驱动端球体0.021断层', '驱动端球体0.028断层'],
        "12K1马力1772电机速风扇端": ['正常状态风扇端', '风扇端内圈0.007断层', '风扇端内圈0.014断层', '风扇端内圈0.021断层', '风扇端外圈中心0.007断层', '风扇端外圈右边0.007断层', '风扇端外圈左边0.007断层', '风扇端外圈左边0.014断层', '风扇端外圈左边0.021断层', '风扇端球体0.007断层', '风扇端球体0.014断层', '风扇端球体0.021断层'],
        "12K1马力1772电机速驱动端": ['正常状态驱动端', '驱动端内圈0.007断层', '驱动端内圈0.014断层', '驱动端内圈0.021断层', '驱动端内圈0.028断层', '驱动端外圈中心0.007断层', '驱动端外圈中心0.014断层', '驱动端外圈中心0.021断层', '驱动端外圈右边0.007断层', '驱动端外圈右边0.021断层', '驱动端外圈左边0.007断层', '驱动端外圈左边0.021断层', '驱动端球体0.007断层', '驱动端球体0.014断层', '驱动端球体0.021断层', '驱动端球体0.028断层'],
        "12K2马力1750电机速风扇端": ['正常状态风扇端', '风扇端内圈0.007断层', '风扇端内圈0.014断层', '风扇端内圈0.021断层', '风扇端外圈中心0.007断层', '风扇端外圈右边0.007断层', '风扇端外圈左边0.007断层', '风扇端外圈左边0.014断层', '风扇端外圈左边0.021断层', '风扇端球体0.007断层', '风扇端球体0.014断层', '风扇端球体0.021断层'],
        "12K2马力1750电机速驱动端": ['正常状态驱动端', '驱动端内圈0.007断层', '驱动端内圈0.014断层', '驱动端内圈0.021断层', '驱动端内圈0.028断层', '驱动端外圈中心0.007断层', '驱动端外圈中心0.014断层', '驱动端外圈中心0.021断层', '驱动端外圈右边0.007断层', '驱动端外圈右边0.021断层', '驱动端外圈左边0.007断层', '驱动端外圈左边0.021断层', '驱动端球体0.007断层', '驱动端球体0.014断层', '驱动端球体0.021断层', '驱动端球体0.028断层'],
        "12K3马力1730电机速风扇端": ['正常状态风扇端', '风扇端内圈0.007断层', '风扇端内圈0.014断层', '风扇端内圈0.021断层', '风扇端外圈中心0.007断层', '风扇端外圈右边0.007断层', '风扇端外圈左边0.007断层', '风扇端外圈左边0.014断层', '风扇端外圈左边0.021断层', '风扇端球体0.007断层', '风扇端球体0.014断层', '风扇端球体0.021断层'],
        "12K3马力1730电机速驱动端": ['正常状态驱动端', '驱动端内圈0.007断层', '驱动端内圈0.014断层', '驱动端内圈0.021断层', '驱动端内圈0.028断层', '驱动端外圈中心0.007断层', '驱动端外圈中心0.014断层', '驱动端外圈中心0.021断层', '驱动端外圈右边0.007断层', '驱动端外圈右边0.021断层', '驱动端外圈左边0.007断层', '驱动端外圈左边0.021断层', '驱动端球体0.007断层', '驱动端球体0.014断层', '驱动端球体0.021断层', '驱动端球体0.028断层'],
}


# 构建顶部模块
def build_top_panel3(stopped_interval):
    return html.Div(
        id="top-section-container",
        className="row",
        children=[
            # Metrics summary
            html.Div(
                id="metric-summary-session",
                className="eight columns",
                children=[
                    generate_section_banner("该轴承数据的时间序列图"),
                    dcc.Loading(
                        id="loading-die-row",
                        type="default",  # 可以选择 "circle", "dot", 或者 "default"
                        children=[
                            dcc.Graph(id='die-row')
                        ],
                    ),
                ],
            ),
            # Piechart
            html.Div(
                id="ooc-piechart-outer",
                className="four columns",
                children=[
                    generate_section_banner("该轴承数据的CDI退化指标序列图"),
                    dcc.Loading(
                        id="loading-img-opt3",
                        type="default",
                        children=[
                            dcc.Graph(id='img-opt3')
                        ],
                    ),
                ],
            ),
        ],
    )



# 构建顶部模块
def build_top_panel4(stopped_interval):
    return html.Div(
        id="top-section-container",
        className="row",
        children=[
            # Metrics summary
            html.Div(
                id="metric-summary-session",
                className="eight columns",
                children=[
                    generate_section_banner("该轴承数据的时间序列图"),
                    dcc.Loading(
                        id="loading-life-row",
                        type="default",
                        children=[
                            dcc.Graph(id='life-row')
                        ],
                    ),

                ],
            ),
            # Piechart
            html.Div(
                id="ooc-piechart-outer",
                className="four columns",
                children=[
                    generate_section_banner("该轴承数据的CDI退化指标序列图"),
                    dcc.Loading(
                        id="loading-img_opt4",
                        type="default",
                        children=[
                            dcc.Graph(id="img_opt4")
                        ],
                    ),
                    
                ],
            ),
        ],
    )

@callback(
    [Output(component_id='output-lab', component_property='children'),
     Output(component_id='img_opt', component_property='children')],
    [Input(component_id='model-select-dropdown', component_property='value'),
     Input(component_id='upload-data', component_property='contents')]
)
def update_output_div(model_value, file_contents):
    img = html.Img(src=img_options.get(model_value))
    output_cont = text_options.get(model_value)
    return f'{output_cont}', img


def parse_contents(contents):
    # 解码文件内容
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    # 假设上传的是 CSV 文件，且有表头，使用 skiprows 跳过表头
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), skiprows=1, header=None)  # 跳过表头
    return df



# 计算 Sigmoid 变换
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# 计算移动平均和 ts_rank
def ts_rank(signal, window=1):
    moving_average = pd.Series(signal).rolling(window).mean()
    rank = moving_average.rank(ascending=False)
    return rank.fillna(0).values

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# 使用移动平均法进行平滑处理
def moving_average(signal, window_size):
    return pd.Series(signal).rolling(window=window_size, center=False).mean()

# 使用指数加权移动平均法进行平滑处理
def exponential_moving_average(signal, span):
    return pd.Series(signal).ewm(span=span, adjust=False).mean()

all_signals = []  # 存储所有文件的振动信号
rms_values = []   # 存储所有文件的 RMS 值
rmv_values = []   # 存储所有文件的 RMV 值

def read_data(content):
    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    # 读取 CSV 数据
    data = pd.read_csv(io.StringIO(decoded.decode('utf-8')), header=None)
    column_len = data.shape[1]
    if column_len > 4:
        # 提取第五列（索引为 4）作为水平方向的振动信号
        horizontal_signal = data.iloc[:, 4]
        
        # 添加到 all_signals 列表
        all_signals.append(horizontal_signal)
                
        # 计算均方根值 (RMS)
        rms = np.sqrt(np.mean(np.square(horizontal_signal)))
        rms_values.append(rms)
                
        # 计算整流平均值 (RMV)
        rmv = np.mean(np.abs(horizontal_signal))
        rmv_values.append(rmv)
    return all_signals,rms_values, rmv_values


def die_data_analysis(rms_values, rmv_values):
    rms_values = np.array(rms_values)
    rmv_values = np.array(rmv_values)
        
    # 对均方根值和整流平均值应用 Sigmoid 变换
    sigmoid_rms = sigmoid(rms_values)
    sigmoid_arv = sigmoid(rmv_values)
    # 生成退化指标
    degradation_index = (sigmoid_rms + sigmoid_arv) / 2
    sigmoid_values = (normalize(degradation_index) * 5)-1 
    ts_rms = -ts_rank(rms_values)
    ts_arv = -ts_rank(rmv_values)
    ts_rms_normalize = normalize(ts_rms)
    ts_rmv_normalize = normalize(ts_arv)
    ts_rank_values = 2*(ts_rms_normalize + ts_rmv_normalize)-2
    CDI = ts_rank_values+sigmoid_values
    span_value = 10  # 指数加权移动平均的跨度
    # 指数加权移动平均平滑
    CDI = exponential_moving_average(CDI, span_value)
    normal_data = np.array(CDI)
    # 时间序列
    time_series = np.arange(len(normal_data))
    
    # 阈值，判断是否开始退化 (可以根据实际数据调整)
    threshold = 0.15
    is_degrading = np.any(normal_data > threshold)
    return is_degrading,time_series,normal_data

def wiener_process(time, drift, diffusion, x0=0):
    dt = np.diff(time)  # 时间间隔
    x = np.zeros(len(time))
    x[0] = x0

    for i in range(1, len(time)):
        x[i] = x[i - 1] + drift * dt[i - 1] + diffusion * np.sqrt(dt[i - 1]) * np.random.randn()

    return x

class AdaptiveKalmanFilter:
    def __init__(self, A, B, C, Q, R, P, x0):
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

@app.callback(
    [Output(component_id='life-row', component_property='figure'),
    Output(component_id='img_opt4', component_property='figure'),
    Output(component_id='stitu-life', component_property='children'),
    Output(component_id='life-life', component_property='children')],
    [Input(component_id='value-name-input', component_property='value'),
    Input(component_id='upload-data4', component_property='contents'),]
)
def update_output_div(input_value,contents):
    if contents is None:
        return dash.no_update  # 如果没有上传内容，不更新图表
    if input_value is None:
        return dash.no_update

    # 遍历每个文件内容
    for content in contents:
        all_signals,rms_values, rmv_values = read_data(content)
    is_degrading,time_series,normal_data = die_data_analysis(rms_values,rmv_values)
    if not is_degrading:
        print("检测到轴承处于正常状态，暂时没有明显退化迹象，预测轴承仍处于健康状态。")
        print("在当前状态下，剩余使用寿命 (RUL) 可能较长，暂时无需维修。")
    else:
        # 模拟维纳过程
        drift = 0.05  # 漂移系数
        diffusion = 0.02  # 扩散系数
        wiener_data = wiener_process(time_series, drift, diffusion)
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
        # 进行卡尔曼滤波预测
        rul_predictions = []
        for z in normal_data:
            kf.predict()
            predicted_state = kf.update(z)
            rul_predictions.append(predicted_state[0])
            # 设定一个退化的故障阈值
    failure_threshold = float(input_value)
    rul_predictions[-1] = float(rul_predictions[-1])
    rul_predictions[0] = float(rul_predictions[0])
    print('当前轴承的状态值为',rul_predictions[-1])
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

    all_signals = np.array(all_signals)
    all_signals = [item for sublist in all_signals for item in sublist]

    # 创建时间序列数据
    x_values = list(range(len(normal_data)))  # 序号（时间序列）
    y_values = normal_data  # 提取的所有数据
    # 使用plotly绘图
    fig = go.Figure(data=[go.Scatter(
        x=x_values,
        y=y_values,
        mode='lines',
        name='CDI图'
    )])
    # 更新布局
    fig.update_layout(
        margin=dict(l=50, r=50, b=50, t=50),
        height=350,
        width=600,
        autosize=True,
        title='CDI图',
        xaxis_title='序号',
        yaxis_title='振动信号',
        xaxis=dict(
            showgrid=True,
            gridcolor='#e0e0e0',
            showline=True,
            linewidth=0.5,
            linecolor='#333333'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#e0e0e0',
            showline=True,
            linewidth=0.5,
            linecolor='#333333'
        ),
        font=dict(family='Arial, sans-serif'),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff'
    )
    # 创建时间序列数据
    x_valuesall = list(range(len(all_signals)))  # 序号（时间序列）
    y_valuesall = all_signals  # 提取的所有数据

    # 使用plotly绘图
    figall = go.Figure(data=[go.Scatter(
        x=x_valuesall,
        y=y_valuesall,
        mode='lines',
        name='时间序列图'
    )])

    # 更新布局
    figall.update_layout(
        margin=dict(l=50, r=50, b=50, t=50),
        height=350,
        width=600,
        autosize=True,
        title='时间序列图',
        xaxis_title='序号',
        yaxis_title='振动信号',
        xaxis=dict(
            showgrid=True,
            gridcolor='#e0e0e0',
            showline=True,
            linewidth=0.5,
            linecolor='#333333'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#e0e0e0',
            showline=True,
            linewidth=0.5,
            linecolor='#333333'
        ),
        font=dict(family='Arial, sans-serif'),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff'
    )

    # 返回生成的图和其他结果
    # 添加AI分析
    current_state = f"当前状态值: {round(rul_predictions[-1],2)}"
    prediction_str = f"预测剩余寿命: {round(remaining_life_hours,2)}小时"
    data_features = {
        "数据点数": len(all_signals),
        "振动信号均值": np.mean(all_signals),
        "振动信号标准差": np.std(all_signals),
        "RMS均值": np.mean(rms_values),
        "RMV均值": np.mean(rmv_values)
    }
    
    ai_analysis = get_ai_analysis(current_state, prediction_str, data_features)
    
    # 返回结果时添加AI分析
    analysis_content = dcc.Markdown(
        f"""## 🎯 寿命预测结果

### 📊 数据统计
- **数据点数**: {data_features['数据点数']}
- **振动信号均值**: {data_features['振动信号均值']}
- **振动信号标准差**: {data_features['振动信号标准差']}
- **RMS均值**: {data_features['RMS均值']}
- **RMV均值**: {data_features['RMV均值']}

### 💡 预测的剩余寿命
**{round(remaining_life_hours,2)} 小时**

### 📝 AI专家分析报告
{ai_analysis}""",
        style={
            'backgroundColor': '#2e2f3f',
            'padding': '20px',
            'borderRadius': '8px',
            'color': 'white',
            'fontFamily': 'Arial, sans-serif',
            'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
            'lineHeight': '1.6'
        }
    )
    return figall, fig, analysis_content, round(remaining_life_hours,2)

@app.callback(
    [Output(component_id='die-row', component_property='figure'),
    Output(component_id='img-opt3', component_property='figure'),
    Output(component_id='die-value', component_property='children')],
    Input(component_id='upload-data3', component_property='contents'),
)
def update_output_div(contents):
    if contents is None:
        return dash.no_update  # 如果没有上传内容，不更新图表


    # 遍历每个文件内容
    for content in contents:
        all_signals,rms_values,rmv_values = read_data(content)
    rms_values = np.array(rms_values)
    rmv_values = np.array(rmv_values)
        
    # 对均方根值和整流平均值应用 Sigmoid 变换
    sigmoid_rms = sigmoid(rms_values)
    sigmoid_arv = sigmoid(rmv_values)
    # 生成退化指标
    degradation_index = (sigmoid_rms + sigmoid_arv) / 2
    sigmoid_values = (normalize(degradation_index) * 5)-1 
    ts_rms = -ts_rank(rms_values)
    ts_arv = -ts_rank(rmv_values)
    ts_rms_normalize = normalize(ts_rms)
    ts_rmv_normalize = normalize(ts_arv)
    ts_rank_values = 2*(ts_rms_normalize + ts_rmv_normalize)-2
    CDI = ts_rank_values+sigmoid_values
    span_value = 10  # 指数加权移动平均的跨度
    # 指数加权移动平均平滑
    CDI = exponential_moving_average(CDI, span_value)
    normal_data = np.array(CDI)
    # 时间序列
    time_series = np.arange(len(normal_data))
    
    # 阈值，判断是否开始退化 (可以根据实际数据调整)
    threshold = 0.15
    is_degrading = np.any(normal_data > threshold)

    if not is_degrading:
        print("检测到轴承处于正常状态，暂时没有明显退化迹象，预测轴承仍处于健康状态。")
        print("在当前状态下，剩余使用寿命 (RUL) 可能较长，暂时无需维修。")
    else:
        # 模拟维纳过程
        drift = 0.05  # 漂移系数
        diffusion = 0.02  # 扩散系数
        wiener_data = wiener_process(time_series, drift, diffusion)
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
        # 进行卡尔曼滤波预测
        rul_predictions = []
        for z in normal_data:
            kf.predict()
            predicted_state = kf.update(z)
            rul_predictions.append(predicted_state[0])
    rul_predictions[-1] = float(rul_predictions[-1])
    rul_predictions[0] = float(rul_predictions[0])
    print('当前轴承的失效阈值为',rul_predictions[-1])


    all_signals = np.array(all_signals)
    all_signals = [item for sublist in all_signals for item in sublist]

    # 创建时间序列数据
    x_values = list(range(len(normal_data)))  # 序号（时间序列）
    y_values = normal_data  # 提取的所有数据

        # 使用plotly绘图
    fig = go.Figure(data=[go.Scatter(
        x=x_values,
        y=y_values,
        mode='lines',
        name='CDI图'
    )])

    # 更新布局
    fig.update_layout(
        margin=dict(l=50, r=50, b=50, t=50),
        height=350,
        width=600,
        autosize=True,
        title='CDI图',
        xaxis_title='序号',
        yaxis_title='振动信号',
        xaxis=dict(
            showgrid=True,
            gridcolor='#e0e0e0',
            showline=True,
            linewidth=0.5,
            linecolor='#333333'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#e0e0e0',
            showline=True,
            linewidth=0.5,
            linecolor='#333333'
        ),
        font=dict(family='Arial, sans-serif'),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff'
    )
    # 创建时间序列数据
    x_valuesall = list(range(len(all_signals)))  # 序号（时间序列）
    y_valuesall = all_signals  # 提取的所有数据

    # 使用plotly绘图
    figall = go.Figure(data=[go.Scatter(
        x=x_valuesall,
        y=y_valuesall,
        mode='lines',
        name='时间序列图'
    )])

    # 更新布局
    figall.update_layout(
        margin=dict(l=50, r=50, b=50, t=50),
        height=350,
        width=600,
        autosize=True,
        title='时间序列图',
        xaxis_title='序号',
        yaxis_title='振动信号',
        xaxis=dict(
            showgrid=True,
            gridcolor='#e0e0e0',
            showline=True,
            linewidth=0.5,
            linecolor='#333333'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#e0e0e0',
            showline=True,
            linewidth=0.5,
            linecolor='#333333'
        ),
        font=dict(family='Arial, sans-serif'),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff'
    )
    # 返回生成的图和其他结果
    # 添加AI分析
    current_state = f"当前失效阈值: {round(rul_predictions[-1],2)}"
    prediction_str = "阈值评估中"
    data_features = {
        "数据点数": len(all_signals),
        "振动信号均值": round(np.mean(all_signals), 4),
        "振动信号标准差": round(np.std(all_signals), 4),
        "RMS均值": round(np.mean(rms_values), 4),
        "RMV均值": round(np.mean(rmv_values), 4)
    }
    
    ai_analysis = get_ai_analysis(current_state, prediction_str, data_features)
    
    # 返回结果时添加AI分析
    # 使用dcc.Markdown组件来格式化显示分析结果
    analysis_content = dcc.Markdown(
        f"""## 🎯 失效阈值评估结果

### 📊 数据统计
- **数据点数**: {data_features['数据点数']}
- **振动信号均值**: {data_features['振动信号均值']}
- **振动信号标准差**: {data_features['振动信号标准差']}
- **RMS均值**: {data_features['RMS均值']}
- **RMV均值**: {data_features['RMV均值']}

### 💡 建议的失效阈值
**{round(rul_predictions[-1],2)}**

### 📝 AI专家分析报告
{ai_analysis}""",
        style={
            'backgroundColor': '#2e2f3f',
            'padding': '20px',
            'borderRadius': '8px',
            'color': 'white',
            'fontFamily': 'Arial, sans-serif',
            'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
            'lineHeight': '1.6'
        }
    )
    return fig, figall, analysis_content
@app.callback(
    Output('metric-rows', 'figure'),
    Input('upload-data', 'contents')
)
def update_output(contents):
    if contents is None:
        return dash.no_update  # 如果没有上传内容，不更新图表

    # 解析文件内容
    df = parse_contents(contents)

    # 检查数据是否为空
    if df.empty:
        return dash.no_update  # 如果数据为空，不更新图表

    # 获取数据长度
    length = len(df)

    # 创建序号列表，以500为间隔
    x_values = list(range(0, length))  # 使用所有索引
    y_values = df[0].values  # 取第一列数据

    # 仅使用每500的索引
    x_values_filtered = x_values[::500]  # 以500为间隔
    y_values_filtered = y_values[::500]  # 以500为间隔

    # 使用plotly绘图
    fig = go.Figure(data=[go.Scatter(
        x=x_values_filtered,
        y=y_values_filtered,
        mode='lines',
        name='时间序列图'
    )])

    # 更新布局
    fig.update_layout(
        margin=dict(l=50, r=50, b=50, t=50),
        height=350,
        width=600,
        autosize=True,
        title='时间序列图',
        xaxis_title='序号',
        yaxis_title='振动信号',
        xaxis=dict(
            showgrid=True,
            gridcolor='#e0e0e0',
            showline=True,
            linewidth=0.5,
            linecolor='#333333'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#e0e0e0',
            showline=True,
            linewidth=0.5,
            linecolor='#333333'
        ),
        font=dict(family='Arial, sans-serif'),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff'
    )
    return fig
def build_chart_panel():
    return html.Div(
        id="control-chart-container",
        className="twelve columns",
        children=[
            generate_section_banner("故障状态概率及检测结果"),
            dcc.Loading(
                id="loading-result1",
                type="default",
                style={"position": "relative","top":"200px"},
                children=[
                    html.Div(id="result_pre",children=[]),
                    html.Div(id="result_ts")
                ],
            ),
            
        ],
    )
def build_chart_panel4():
    return html.Div(
            id="control-chart-container",
            className="twelve columns",
            children=[
                generate_section_banner("该轴承的当前状态与剩余寿命"),
                dcc.Loading(
                    id="loading-result2",
                    type="default",
                    style={"position": "relative","top":"200px"},
                    children=[
                        html.Div(id="stitu-life"),
                        html.Div(id="life-life")
                    ],
                ),
                ],
            )
def build_chart_panel3():
    return html.Div(
            id="control-chart-container",
            className="twelve columns",
            children=[
                generate_section_banner("该轴承的失效阈值"),
                dcc.Loading(
                    id="loading-result3",
                    type="default",
                    style={"position": "relative","top":"200px"},
                    children=[
                        html.Div(id="die-value"),
                    ],
                ),
                
                ],
            )


h5_options={
        "12K0马力1797电机速驱动端": "assets/12K0马力1797电机速驱动端.h5",
        "12K0马力1797电机速风扇端": "assets/12K0马力1797电机速风扇端.h5",
        "12K1马力1772电机速驱动端": "assets/12K1马力1772电机速驱动端.h5",
        "12K1马力1772电机速风扇端": "assets/12K1马力1772电机速风扇端.h5",
        "12K2马力1750电机速驱动端": "assets/12K2马力1750电机速驱动端.h5",
        "12K2马力1750电机速风扇端": "assets/12K2马力1750电机速风扇端.h5",
        "12K3马力1730电机速驱动端": "assets/12K3马力1730电机速驱动端.h5",
        "12K3马力1730电机速风扇端": "assets/12K3马力1730电机速风扇端.h5",
}

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
    pre_ap = []
    # 使用模型进行预测
    predictions = model.predict(processed_data)
    predicted_label_idx = np.argmax(predictions, axis=1)[0]
    for i in range(len(state_labels[0])):
        print(f'{state_labels[0][i]}的概率为:{str(predictions[0][i]*100)[:5]}%')
        pre_ap.append(f'{state_labels[0][i]}概率为:{str(predictions[0][i]*100)[:5]}%')
    # 打印检测到的故障状态标签
    print(f"检测的故障状态标签: {state_labels[0][predicted_label_idx]}")
    return pre_ap,str(state_labels[0][predicted_label_idx])
@app.callback(
    [Output(component_id='result_pre', component_property='children'),
    Output(component_id='result_ts', component_property='children')],
    [Input(component_id='model-select-dropdown', component_property='value'),
     Input(component_id='upload-data', component_property='contents')]  # 注意这里是 'contents'
)
def update_output(input_value, contents):
    # 设置参数
    samples_per_block = 1681 
    model_path = h5_options.get(input_value)
    lab_cont = text_options.get(input_value)
    # 训练时的状态标签列表，按顺序列出
    state_labels = [lab_cont]
    
    # 检查上传数据
    if contents is None:
        return ["请上传数据文件。"], []  # 确保返回值为 list 或 tuple
    if input_value is None:
        return ["请选择一个模型。"], []  # 确保返回值为 list 或 tuple

    # 解码上传的文件内容
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    # 读取 CSV 数据
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    # 转换为 DataFrame
    new_data_df = pd.DataFrame(df)
    
    # 对新数据进行故障检测
    pre_ap, result = DetectNewDataFromDataFrame(new_data_df, model_path, samples_per_block, state_labels)

    # 添加AI分析
    current_state = f"当前状态: {result}"
    prediction_str = "故障检测完成"
    data_features = {
        "数据点数": len(df),
        "振动信号均值": np.mean(df.values),
        "振动信号标准差": np.std(df.values)
    }
    
    ai_analysis = get_ai_analysis(current_state, prediction_str, data_features)
    
    # 在结果中添加AI分析
    analysis_content = dcc.Markdown(
        f"""## 🎯 故障检测结果

### 📊 数据统计
- **数据点数**: {data_features['数据点数']}
- **振动信号均值**: {data_features['振动信号均值']}
- **振动信号标准差**: {data_features['振动信号标准差']}

### 💡 检测到的故障状态
**{result}**

### 📝 AI专家分析报告
{ai_analysis}""",
        style={
            'backgroundColor': '#2e2f3f',
            'padding': '20px',
            'borderRadius': '8px',
            'color': 'white',
            'fontFamily': 'Arial, sans-serif',
            'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
            'lineHeight': '1.6'
        }
    )
    
    return analysis_content, result


app.layout = html.Div(
    id="big-app-container",
    children=[
        build_banner(),
        dcc.Interval(
            id="interval-component",
            interval=2 * 1000,  # in milliseconds
            n_intervals=50,  # start at batch 50
            disabled=True,
        ),
        html.Div(
            id="app-container",
            children=[
                build_tabs(),
                # Main app
                html.Div(id="app-content"),
            ],
        ),
        # dcc.Store(id="value-setter-store", data=init_value_setter_store()),
        dcc.Store(id="n-interval-stage", data=50),
        # generate_modal(),
    ],
)

@app.callback(
    [Output("app-content", "children"), Output("interval-component", "n_intervals")],
    [Input("app-tabs", "value")],
    [State("n-interval-stage", "data")],
)
# 回调函数
def render_tab_content(tab_switch, stopped_interval):
    if tab_switch == "tab2":
        # 第二个标签的内容
        return (
            html.Div(
                id="status-container",
                children=[
                    build_quick_stats_panel(multiple=False),
                    html.Div(
                        id="graphs-container",
                        children=[build_top_panel(stopped_interval), build_chart_panel()],
                    ),
                ],
            ),
            stopped_interval,
        )
    elif tab_switch == "tab3":
        # 第三个标签的内容
        return (
            html.Div(
                id="status-container",
                children=[
                    build_quick_stats_panel3(multiple=True),
                    html.Div(
                        id="graphs-container",
                        children=[build_top_panel3(stopped_interval), build_chart_panel3()],
                    ),
                ],
            ),
            stopped_interval,
        )
    elif tab_switch == "tab4":
        # 第四个标签的内容
       return (
            html.Div(
                id="status-container",
                children=[
                    build_quick_stats_panel4(multiple=True),
                    html.Div(
                        id="graphs-container",
                        children=[build_top_panel4(stopped_interval), build_chart_panel4()],
                    ),
                ],
            ),
            stopped_interval,
        )



if __name__ == "__main__":
    app.run_server(debug=True, port=8050)