
# Time Series Transformer: Three Web Implementations

使用同一套 PyTorch 时序 Transformer，分别通过 Streamlit、Flask 和 FastAPI 实现网页训练与预测。

## 项目目标

对比三种 Python Web 开发方式：

- Streamlit：用 Python 快速生成数据应用界面；
- Flask：HTML、CSS、JavaScript 与后端 API；
- FastAPI：类型校验、OpenAPI 和 SSE 实时推送。

三个版本共用：

```text
shared/transformer_core.py
```

因此模型和数据完全相同，区别只在 Web 实现方式。

## 项目结构

```text
transformer-three-ways/
├── shared/
│   ├── __init__.py
│   └── transformer_core.py
│
├── streamlit_version/
│   └── app.py
│
├── flask_version/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── app.js
│       └── style.css
│
├── fastapi_version/
│   ├── app.py
│   └── static/
│       └── index.html
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 功能

- 生成具有周期、动量和随机噪声的模拟时序；
- 使用滑动窗口制作训练样本；
- 训练 PyTorch Transformer；
- 实时展示 Epoch、Loss 和 Accuracy；
- 预测下一个时间点上涨或下跌；
- 展示置信度；
- 表格标记预测正确或错误。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements.txt
```

## 测试算法核心

```bash
python -m shared.transformer_core
```

## Streamlit 版本

启动：

```bash
python -m streamlit run \
  streamlit_version/app.py \
  --server.port 8501
```

访问：

```text
http://127.0.0.1:8501
```

特点：

- 页面主要由 Python 生成；
- Streamlit 管理界面更新；
- 适合模型演示和数据工具。

## Flask 版本

启动：

```bash
python -m flask \
  --app flask_version.app \
  run \
  --debug \
  --host 127.0.0.1 \
  --port 5000
```

访问：

```text
http://127.0.0.1:5000
```

主要接口：

```text
GET  /api/health
GET  /api/preview
POST /api/train
GET  /api/status/<job_id>
```

特点：

- HTML 负责页面结构；
- CSS 负责样式；
- JavaScript 调用 Flask API；
- 浏览器轮询训练状态。

## FastAPI 版本

启动：

```bash
python -m uvicorn \
  fastapi_version.app:app \
  --reload \
  --host 127.0.0.1 \
  --port 8000
```

访问：

```text
http://127.0.0.1:8000
```

API 文档：

```text
http://127.0.0.1:8000/docs
```

主要接口：

```text
GET  /api/health
GET  /api/preview
POST /api/train
GET  /api/events/<job_id>
```

特点：

- Pydantic 自动校验请求参数；
- 自动生成 OpenAPI 文档；
- 使用 SSE 主动推送训练进度。

## 三种实现的通信方式

```text
Streamlit
浏览器 ↔ Streamlit内部通信 ↔ Python页面代码

Flask
浏览器 → POST创建任务
浏览器 → 定时GET查询训练状态

FastAPI
浏览器 → POST创建任务
服务器 → 通过SSE持续推送训练状态
```

## 本地端口

```text
127.0.0.1:8501 → Streamlit
127.0.0.1:5000 → Flask
127.0.0.1:8000 → FastAPI
```

IP 用来找到电脑，端口用来找到这台电脑上的具体服务。

## 注意事项

- 当前项目用于学习，不适合直接作为生产系统；
- 训练任务只保存在内存中；
- Flask 和 FastAPI 版本没有用户认证；
- 不要直接暴露到公网；
- 不要在仓库中保存 API Key、Token 或敏感数据；
- 模拟数据上的准确率不代表真实业务效果。
