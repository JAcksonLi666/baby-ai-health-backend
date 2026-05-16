# 宝宝健康档案 AI 助手 - 后端服务

基于 Python FastAPI 构建的婴幼儿健康档案管理后端服务，提供 OCR 识别、向量数据库存储和智能问答功能。

## 📋 功能特性

- 📤 **OCR 识别**：支持图片和 PDF 的文字提取（PaddleOCR）
- 🔍 **向量数据库**：ChromaDB 持久化存储健康档案
- 🤖 **智能问答**：基于 RAG（Retrieval-Augmented Generation）的问答系统
- 🌐 **RESTful API**：完整的 CRUD 接口
- 🔒 **隐私保护**：支持数据脱敏处理
- 📊 **健康趋势分析**：基于历史数据的指标趋势分析

## 🛠️ 技术栈

- **框架**: FastAPI 0.109.0
- **OCR**: PaddleOCR 2.7.3
- **向量数据库**: ChromaDB 0.4.22
- **LLM 框架**: LangChain 0.1.4
- **嵌入模型**: Sentence Transformers
- **本地模型**: Ollama (Qwen2.5:7B)
- **语言**: Python 3.10+

## 📁 项目结构

```
backend/
├── main.py             # FastAPI 应用入口，定义 API 路由
├── config.py           # 配置管理（环境变量、路径、参数）
├── models.py           # Pydantic 数据模型定义
├── ocr_service.py      # OCR 服务（图片/PDF文字提取）
├── vector_db.py        # ChromaDB 向量数据库操作
├── llm_service.py      # 大语言模型服务（本地/Ollama）
├── rag_service.py      # RAG 问答服务（检索+生成）
├── requirements.txt    # Python 依赖列表
├── .env.example        # 环境变量示例
└── __pycache__/        # Python 编译缓存
```

## 🚀 快速开始

### 环境要求

- Python 3.10+
- NVIDIA GPU（推荐 RTX 4060 8GB+）
- Ollama（本地模型支持）

### 安装依赖

```bash
cd backend

# 创建虚拟环境（推荐）
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 安装 Ollama 并下载模型

```bash
# 安装 Ollama
# Windows: https://github.com/ollama/ollama/releases
# macOS/Linux: https://ollama.ai/download

# 下载推荐模型
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

# 启动 Ollama 服务
ollama serve
```

### 配置环境变量

```bash
copy .env.example .env
```

编辑 `.env` 文件：

```env
# 服务器配置
HOST=0.0.0.0
PORT=8000

# Ollama 配置
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b

# 存储配置
UPLOAD_DIR=data/uploads
CHROMA_DIR=data/chroma

# 文件限制
MAX_UPLOAD_SIZE=10485760  # 10MB

# 日志配置
LOG_LEVEL=INFO
```

### 启动服务

```bash
# 开发模式
python main.py

# 或使用 uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

服务将在 http://localhost:8000 启动

## 📡 API 接口

### 健康检查

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | / | 基础健康检查 |
| GET | /health | 详细健康检查（含 Ollama、ChromaDB 状态） |

### 文件上传

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | /upload | 上传化验单图片或 PDF |

**请求参数：**
- `file`: UploadFile - 化验单文件
- `record_date`: str (可选) - 记录日期 YYYY-MM-DD
- `record_type`: str (可选) - 记录类型: blood_test, urine_test, general

**响应示例：**
```json
{
  "success": true,
  "file_id": "20240101120000_abc12345",
  "filename": "20240101120000_abc12345.jpg",
  "message": "上传成功，已完成 OCR 识别和向量化存储",
  "extracted_text": "..."
}
```

### 智能问答

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | /ask | 基于历史档案的智能问答 |

**请求体：**
```json
{
  "question": "宝宝最近的体重增长情况如何？",
  "top_k": 3,
  "use_cloud": false
}
```

**响应示例：**
```json
{
  "success": true,
  "answer": "根据历史记录分析...",
  "sources": [...],
  "model_used": "qwen2.5:7b",
  "cloud_used": false
}
```

### 档案管理

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | /records | 获取所有档案列表 |
| GET | /record/{record_id} | 获取指定档案详情 |
| DELETE | /record/{record_id} | 删除指定档案 |

### 健康分析

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | /analyze-trend | 分析特定健康指标趋势 |

### 模型管理

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | /models | 获取可用的 AI 模型列表 |

## 🧩 核心模块

### 1. OCR 服务 (`ocr_service.py`)

- `extract_text_from_image()` - 从图片提取文字
- `extract_text_from_pdf()` - 从 PDF 提取文字
- `extract_health_indicators()` - 提取健康指标（体重、身高、体温等）

### 2. 向量数据库 (`vector_db.py`)

- `add_record()` - 添加记录到向量库
- `search_similar()` - 相似性检索
- `get_all_records()` - 获取所有记录
- `get_record()` - 获取单条记录
- `delete_record()` - 删除记录

### 3. LLM 服务 (`llm_service.py`)

- `generate_local()` - 使用 Ollama 本地模型生成
- `generate_cloud()` - 使用云端 API 生成（预留）
- `check_ollama_health()` - 检查 Ollama 服务状态
- `get_available_models()` - 获取可用模型列表

### 4. RAG 服务 (`rag_service.py`)

- `build_medical_prompt()` - 构建医疗问答 Prompt
- `answer_question()` - 基于 RAG 回答问题
- `analyze_health_trend()` - 分析健康指标趋势

## 📖 API 文档

启动服务后访问：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## ⚠️ 注意事项

1. **医疗免责声明**：系统输出仅供参考，不能替代专业医疗诊断
2. **数据安全**：敏感信息已脱敏处理，请勿上传未脱敏的原始文件
3. **GPU 要求**：建议 8GB 以上显存以获得最佳性能
4. **Ollama 服务**：启动后端前请确保 Ollama 服务正在运行

## 📄 许可证

MIT License
