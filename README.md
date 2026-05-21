# 宝宝健康档案 AI 助手 - 后端服务

基于 Python FastAPI 构建的婴幼儿健康档案管理后端服务，提供 OCR 识别、向量数据库存储、智能问答、症状自查、化验单解析等功能。

## 📋 功能特性

### 核心功能
- 📤 **OCR 识别**：支持图片和 PDF 的文字提取（PaddleOCR）
- 🔍 **向量数据库**：ChromaDB 持久化存储健康档案
- 🤖 **智能问答**：基于 RAG（Retrieval-Augmented Generation）的问答系统
- 🌐 **RESTful API**：完整的 CRUD 接口
- 🔒 **隐私保护**：支持数据脱敏处理
- 📊 **健康趋势分析**：基于历史数据的指标趋势分析
- 📅 **日期自动识别**：从化验单自动提取日期信息

### v1.2.0 新增
- 😴 **睡眠记录管理**：记录宝宝的入睡、醒来时间、睡眠质量
- 💩 **排泄记录管理**：记录尿布类型、颜色、便便状态
- 😭 **哭声记录管理**：记录哭闹类型、强度、持续时间和可能原因
- 🌐 **双 AI 层架构**：本地 RAG 检索 + 云端大模型（蚂蚁·安诊儿）
- 📚 **国家卫健委知识库**：内置婴幼儿照护指南知识库
- 🔧 **流式输出支持**：云端和本地模型均支持流式输出

### v1.3.0 新增
- 🍼 **喂养记录管理**：记录母乳/配方奶/辅食/喝水，支持哺乳侧和奶量
- 📏 **生长发育记录**：记录体重、身高、头围、体温
- 📈 **WHO 生长曲线**：内置 WHO 标准生长数据，支持百分位计算
- 📊 **今日汇总仪表盘**：睡眠/排泄/哭声/喂养/生长发育数据概览 + AI 洞察
- 📚 **知识库动态管理**：支持知识条目的增删查
- 🐳 **Docker 部署**：提供 Dockerfile 和 docker-compose 配置
- 📱 **PWA 支持**：Service Worker + manifest.json

### v1.4.0 新增
- 🧪 **化验单 AI 解析**：LLM 智能解析化验单文本，提取结构化指标数据
- 📋 **化验单智能评估**：根据年龄段参考范围自动评估指标状态（正常/偏低/偏高/危急）
- 🩺 **症状自查**：8 大类 40+ 症状选择，AI 分析可能原因并给出注意事项
- 💬 **对话历史管理**：AI 问答会话持久化存储，支持查看历史对话
- 🔍 **混合检索**：BM25 + 向量相似度混合搜索知识库
- 🚀 **性能优化**：预计算 embedding 缓存、线程安全文件锁、async 非阻塞 LLM 调用
- 🌍 **枚举校验**：ReportType / MessageRole 等枚举类型确保 API 参数安全

## 🛠️ 技术栈

- **框架**: FastAPI 0.109.0
- **OCR**: PaddleOCR 2.7.3
- **向量数据库**: ChromaDB 0.4.22
- **LLM 框架**: LangChain 0.1.4
- **嵌入模型**: Ollama nomic-embed-text（推荐）/ Sentence Transformers（备选）
- **本地模型**: Ollama (Qwen2.5:7B)
- **语言**: Python 3.10+

## 📁 项目结构

```
backend/
├── main.py                 # FastAPI 应用入口，定义所有 API 路由
├── config.py               # 配置管理（环境变量、路径、参数）
├── models.py               # Pydantic 数据模型定义（含枚举校验）
├── ocr_service.py          # OCR 服务（图片/PDF 文字提取）
├── vector_db.py            # ChromaDB 向量数据库操作
├── llm_service.py          # 大语言模型服务（本地/Ollama/云端）
├── rag_service.py          # RAG 问答服务（检索+生成）
├── daily_records.py        # 日常记录服务（睡眠/排泄/哭声/喂养/生长发育）
├── knowledge_base.py       # 知识库服务（BM25 + 向量混合检索）
├── growth_standards.py     # WHO 生长发育标准数据与百分位计算
├── lab_report_parser.py    # 化验单 AI 解析服务（v1.4.0）
├── symptom_checker.py      # 症状自查服务（v1.4.0）
├── chat_history.py         # 对话历史服务（v1.4.0）
├── desensitization.py      # 数据脱敏处理
├── requirements.txt        # Python 依赖列表
├── .env.example            # 环境变量示例
├── Dockerfile              # Docker 镜像配置
└── __pycache__/            # Python 编译缓存
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

# 下载模型
ollama pull qwen2.5:7b        # 对话模型
ollama pull nomic-embed-text  # 嵌入模型（用于向量检索）

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
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b

# 云端大模型配置（第二层AI分析 - 可选）
# 推荐：蚂蚁·安诊儿（AntAngelMed）- 医疗专业大模型
CLOUD_API_KEY=sk-studio-xxxxxxxxxxxx
CLOUD_API_BASE=https://api.ant-ling.com/v1/

# 存储配置
UPLOAD_DIR=data/uploads
VECTOR_DB_DIR=data/vector_db
MODEL_CACHE_DIR=data/models

# 文件限制
MAX_UPLOAD_SIZE=10485760  # 10MB

# 日志配置
LOG_LEVEL=INFO
```

### 云端大模型配置

系统支持双 AI 层架构：
1. **第一层（本地）**：使用 Ollama 本地模型进行 RAG 检索和初步回答
2. **第二层（云端）**：可选使用云端大模型进行增强分析

**推荐云端模型**：蚂蚁·安诊儿（Ling-2.6-1T）- 专为医疗健康场景优化

申请地址：https://modelscope.cn/studios/MedAIBase/AntAngelMed

### 启动服务

```bash
# 开发模式（推荐）
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
| GET | /health | 快速健康检查（含 ChromaDB、Embedding 状态） |

### 文件上传

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | /upload | 上传化验单图片或 PDF |
| POST | /upload/preview | 预览识别结果（不保存到数据库） |

**请求参数：**
- `file`: UploadFile - 化验单文件
- `record_date`: str (可选) - 记录日期 YYYY-MM-DD
- `record_type`: str (可选) - 记录类型: blood_test, urine_test, general, other

### 智能问答

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | /ask | 基于历史档案的智能问答 |
| GET | /ask/stream | 基于历史档案的智能问答（流式） |

**POST /ask 请求体：**
```json
{
  "question": "宝宝最近的体重增长情况如何？",
  "top_k": 3,
  "use_cloud": true,
  "model": "auto"
}
```

### 档案管理

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | /records | 获取所有档案列表 |
| GET | /records/filter | 按类型/日期/关键词筛选档案 |
| GET | /records/types | 获取所有记录类型统计 |
| GET | /record/{record_id} | 获取指定档案详情 |
| PUT | /record/{record_id} | 更新档案信息 |
| DELETE | /record/{record_id} | 删除指定档案 |

### 健康分析

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | /analyze-trend | 分析特定健康指标趋势 |

### 模型管理

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | /models | 获取可用的 AI 模型列表 |

### 联网搜索

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | /search/online | 使用 Tavily 进行联网搜索 |
| GET | /search/status | 获取联网搜索功能状态 |

### 日常记录管理

#### 睡眠记录
| 方法 | 路径 | 描述 |
|------|------|------|
| GET | /api/sleep | 获取睡眠记录列表 |
| POST | /api/sleep | 创建睡眠记录 |
| GET | /api/sleep/ongoing | 获取进行中的睡眠 |
| GET | /api/sleep/{id} | 获取单条睡眠记录 |
| PUT | /api/sleep/{id} | 更新睡眠记录 |
| DELETE | /api/sleep/{id} | 删除睡眠记录 |

#### 排泄记录
| 方法 | 路径 | 描述 |
|------|------|------|
| GET | /api/diaper | 获取排泄记录列表 |
| POST | /api/diaper | 创建排泄记录 |
| GET | /api/diaper/{id} | 获取单条排泄记录 |
| PUT | /api/diaper/{id} | 更新排泄记录 |
| DELETE | /api/diaper/{id} | 删除排泄记录 |

#### 哭声记录
| 方法 | 路径 | 描述 |
|------|------|------|
| GET | /api/cry | 获取哭声记录列表 |
| POST | /api/cry | 创建哭声记录 |
| GET | /api/cry/ongoing | 获取进行中的哭闹 |
| GET | /api/cry/analyze | AI 分析哭声原因 |
| GET | /api/cry/{id} | 获取单条哭声记录 |
| PUT | /api/cry/{id} | 更新哭声记录 |
| DELETE | /api/cry/{id} | 删除哭声记录 |

#### 喂养记录 (v1.3.0)
| 方法 | 路径 | 描述 |
|------|------|------|
| GET | /api/feeding | 获取喂养记录列表 |
| POST | /api/feeding | 创建喂养记录（母乳/配方奶/辅食/喝水） |
| GET | /api/feeding/{id} | 获取单条喂养记录 |
| PUT | /api/feeding/{id} | 更新喂养记录 |
| DELETE | /api/feeding/{id} | 删除喂养记录 |

#### 生长发育记录 (v1.3.0)
| 方法 | 路径 | 描述 |
|------|------|------|
| GET | /api/growth | 获取生长发育记录列表 |
| POST | /api/growth | 创建生长发育记录（体重/身高/头围/体温） |
| GET | /api/growth/latest | 获取最新生长发育记录 |
| GET | /api/growth/{id} | 获取单条生长发育记录 |
| PUT | /api/growth/{id} | 更新生长发育记录 |
| DELETE | /api/growth/{id} | 删除生长发育记录 |

#### 生长发育标准 (v1.3.0)
| 方法 | 路径 | 描述 |
|------|------|------|
| GET | /api/growth/standards | 获取指定年龄的 WHO 生长标准值 |
| GET | /api/growth/percentile | 计算生长指标百分位 |
| GET | /api/growth/age-groups | 获取年龄段定义 |
| GET | /api/growth/metrics | 获取可用的生长指标列表 |

### 仪表盘与知识库

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | /api/today/summary | 获取今日汇总（睡眠/排泄/哭声/喂养/生长发育 + AI 洞察） |
| GET | /api/knowledge/search | 搜索知识库（BM25 + 向量混合检索） |
| GET | /api/knowledge/status | 获取知识库状态 |
| GET | /api/knowledge/list | 列出知识库条目（支持分类过滤） |
| GET | /api/knowledge/{entry_id} | 获取单条知识条目 |
| POST | /api/knowledge | 添加知识条目 |
| DELETE | /api/knowledge/{entry_id} | 删除知识条目 |

### AI - 化验单解析 (v1.4.0)

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | /api/lab-report/parse | 使用 LLM 将化验单文本解析为结构化 JSON |
| POST | /api/lab-report/evaluate | 解析并评估化验单指标（根据年龄段参考范围） |

**POST /api/lab-report/evaluate 请求体：**
```json
{
  "report_type": "blood_routine",
  "month_age": 6,
  "indicators": [
    {"name": "白细胞", "value": "8.5", "unit": "10^9/L"},
    {"name": "血红蛋白", "value": "120", "unit": "g/L"}
  ]
}
```

### AI - 症状自查 (v1.4.0)

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | /api/symptom/analyze | 分析婴幼儿症状，返回分类、可能原因和注意事项 |
| GET | /api/symptom/categories | 获取所有可用症状分类及症状列表 |

**POST /api/symptom/analyze 请求体：**
```json
{
  "symptoms": [
    {"key": "cough", "category": "呼吸系统", "name": "咳嗽"},
    {"key": "fever", "category": "发热", "name": "发热"}
  ],
  "month_age": 6
}
```

### AI - 对话历史 (v1.4.0)

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | /api/chat/sessions | 创建新对话会话 |
| GET | /api/chat/sessions | 列出所有对话会话 |
| GET | /api/chat/sessions/{session_id}/messages | 获取会话消息历史 |
| POST | /api/chat/sessions/{session_id}/messages | 向会话添加消息 |
| DELETE | /api/chat/sessions/{session_id} | 删除对话会话及其所有消息 |

## 🧩 核心模块

### 1. OCR 服务 (`ocr_service.py`)

- `extract_text_from_image()` - 从图片提取文字
- `extract_text_from_pdf()` - 从 PDF 提取文字
- `extract_health_indicators()` - 提取健康指标（体重、身高、体温等）
- `extract_date()` - 从文本中提取日期信息

### 2. 向量数据库 (`vector_db.py`)

- `add_record()` - 添加记录到向量库
- `search_similar()` - 相似性检索
- `get_all_records()` - 获取所有记录
- `get_record()` - 获取单条记录
- `delete_record()` - 删除记录
- `update_record()` - 更新记录信息
- `generate_embedding()` - 生成文本嵌入向量

**嵌入模型支持（自动选择）**：
1. 优先使用 Ollama nomic-embed-text（推荐）
2. 备选使用 Sentence Transformers
3. 最后降级到简单哈希向量

### 3. LLM 服务 (`llm_service.py`)

- `generate_local()` - 使用 Ollama 本地模型生成
- `generate_local_stream()` - 使用 Ollama 本地模型流式生成
- `generate_cloud()` - 使用云端 API 生成（非流式）
- `generate_cloud_stream()` - 使用云端 API 流式生成
- `check_ollama_health()` - 检查 Ollama 服务状态
- `get_available_models()` - 获取可用模型列表
- `select_smartest_model()` - 自动选择最优模型

**支持的云端 API：**
- 蚂蚁·安诊儿（AntAngelMed）- 医疗专业大模型
- DeepSeek
- OpenAI 兼容格式

### 4. RAG 服务 (`rag_service.py`)

- `build_medical_prompt()` - 构建医疗问答 Prompt
- `answer_question()` - 基于 RAG 回答问题
- `analyze_health_trend()` - 分析健康指标趋势

### 5. 知识库服务 (`knowledge_base.py`)

- `search()` - 搜索知识库（BM25 + 向量混合检索）
- `hybrid_search()` - 混合检索（BM25 文本匹配 + 向量相似度）
- `vector_search()` - 纯向量检索（带预计算缓存）
- `get_entry()` / `add_entry()` / `delete_entry()` - 知识条目 CRUD
- `_precompute_embeddings()` - 预计算所有条目的嵌入向量

### 6. 化验单解析服务 (`lab_report_parser.py`) - v1.4.0

- `parse_with_llm()` - 使用 LLM 解析化验单文本
- `evaluate_indicators()` - 根据年龄段参考范围评估指标
- `_parse_via_llm()` - LLM 解析实现（async 非阻塞）
- `_parse_via_regex()` - 正则表达式回退解析

### 7. 症状自查服务 (`symptom_checker.py`) - v1.4.0

- `analyze_symptoms()` - 分析症状，返回分类和可能原因
- `get_categories()` - 获取所有症状分类定义
- `_search_knowledge()` - 检索相关知识库条目

### 8. 对话历史服务 (`chat_history.py`) - v1.4.0

- `create_session()` - 创建新对话会话
- `add_message()` - 添加消息（线程安全）
- `get_session_history()` - 获取会话消息历史
- `list_sessions()` - 列出所有会话
- `delete_session()` - 删除会话
- `get_context_messages()` - 获取 LLM 上下文格式消息

## 📖 API 文档

启动服务后访问：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## ⚠️ 注意事项

1. **医疗免责声明**：系统输出仅供参考，不能替代专业医疗诊断
2. **数据安全**：敏感信息已脱敏处理，请勿上传未脱敏的原始文件
3. **GPU 要求**：建议 8GB 以上显存以获得最佳性能（CPU 也可运行但速度较慢）
4. **Ollama 服务**：启动后端前请确保 Ollama 服务正在运行
5. **嵌入模型**：首次启动会自动检查/下载 nomic-embed-text 模型
6. **版本兼容**：PaddlePaddle 3.x 与 PaddleOCR 2.x 不兼容，需使用 PaddlePaddle 2.6.2 + PaddleOCR 2.7.3
7. **云端模型**：使用云端大模型需要在 `.env` 中配置 `CLOUD_API_KEY` 和 `CLOUD_API_BASE`
8. **知识库**：国家卫健委婴幼儿照护指南知识库会在首次启动时自动加载
9. **线程安全**：对话历史和知识库操作使用文件锁保护，支持并发访问
10. **症状自查**：本工具仅提供症状分类参考，不构成就医建议

## 🔄 版本历史

- **v1.4.0 (2026-05-21)** - AI 增强版本
  - ✅ 化验单 AI 智能解析（LLM + 正则回退）
  - ✅ 化验单指标评估（年龄特异性参考范围）
  - ✅ 症状自查功能（8 大类 40+ 症状）
  - ✅ 对话历史管理（会话持久化存储）
  - ✅ 知识库混合检索（BM25 + 向量相似度）
  - ✅ 预计算 embedding 缓存优化
  - ✅ 线程安全文件锁（chat_history / knowledge_base）
  - ✅ async 非阻塞 LLM 调用
  - ✅ ReportType / MessageRole 枚举校验
  - ✅ 前端国际化完善（LabReportParser / SymptomChecker / ChatHistory）
  - ✅ 移动端响应式设计优化

- **v1.3.0 (2026-05-20)** - 成长管理版本
  - ✅ 添加喂养记录管理功能（母乳/配方奶/辅食/喝水）
  - ✅ 添加生长发育记录功能（体重/身高/头围/体温）
  - ✅ 添加 WHO 生长曲线标准数据
  - ✅ 添加生长指标百分位计算
  - ✅ 今日汇总仪表盘集成喂养和生长发育数据
  - ✅ 知识库动态管理（增删查）
  - ✅ Docker 部署配置（Dockerfile + docker-compose）
  - ✅ PWA 支持（manifest.json + Service Worker）
  - ✅ 代码优化：moment.js 替换为 dayjs、EventSource 泄漏修复等

- **v1.2.0 (2026-05-19)** - 完整功能版本
  - ✅ 添加睡眠记录管理功能（CRUD + 进行中状态）
  - ✅ 添加排泄记录管理功能（CRUD）
  - ✅ 添加哭声记录管理功能（CRUD + AI分析原因）
  - ✅ 添加今日汇总仪表盘（睡眠/排泄/哭声数据概览）
  - ✅ 配置云端大模型（蚂蚁·安诊儿 Ling-2.6-1T）
  - ✅ 双 AI 层架构（本地 RAG + 云端大模型）
  - ✅ 添加国家卫健委婴幼儿照护指南知识库
  - ✅ 添加云端模型流式输出支持
  - ✅ 修复流式输出时忽略 use_cloud 参数的问题
  - ✅ 添加每日记录 API 端点

- **v1.1.0 (2026-05-18)** - 功能增强版本
  - ✅ 添加档案更新接口（PUT /record/{id}）
  - ✅ 添加预识别接口（POST /preview）
  - ✅ 支持日期自动识别
  - ✅ 支持记录类型筛选
  - ✅ 修复 PaddleOCR 版本兼容问题
  - ✅ 修复 OCR 服务脱敏方法缺失问题

- **v1.0.0 (2026-05-16)** - MVP 版本
  - ✅ 基础 OCR 识别功能
  - ✅ 向量数据库存储
  - ✅ RAG 智能问答
  - ✅ 基础 CRUD 接口

## 📄 许可证

MIT License
