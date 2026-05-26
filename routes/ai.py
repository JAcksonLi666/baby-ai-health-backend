from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse
from models import AskRequest, AskResponse, LabReportParseRequest, LabReportResponse, SymptomCheckRequest
from rag_service import rag_service
from lab_report_parser import lab_report_parser
from symptom_checker import symptom_checker
from chat_history import chat_history_service
from typing import Dict
import json
import time
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["AI 服务"])


# ==================== Rate Limiter ====================
_rate_limit_store: Dict[str, list] = {}
_RATE_LIMIT_WINDOW = 60
_RATE_LIMIT_MAX_REQUESTS = 60
_RATE_LIMIT_AI_MAX = 10


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """基于历史健康档案的智能问答"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    try:
        result = rag_service.answer_question(
            question=request.question,
            top_k=request.top_k,
            use_cloud=request.use_cloud,
            model=request.model
        )
        if result.get("success"):
            return AskResponse(
                success=True,
                answer=result["answer"],
                sources=result.get("sources", []),
                model_used=result.get("model_used", "unknown"),
                cloud_used=result.get("cloud_used", False)
            )
        else:
            return AskResponse(
                success=False,
                answer="",
                sources=[],
                model_used=result.get("model_used", "unknown"),
                cloud_used=request.use_cloud
            )
    except Exception as e:
        logger.error(f"问答处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")


def generate_stream_response(question: str, top_k: int = 3, use_cloud: bool = False, model: str = "auto"):
    """生成流式响应"""
    try:
        result_gen = rag_service.answer_question_stream(
            question=question,
            top_k=top_k,
            use_cloud=use_cloud,
            model=model
        )

        for token in result_gen:
            if token is None:
                continue

            if token.startswith("错误:"):
                yield f"data: {json.dumps({'error': token}, ensure_ascii=False)}\n\n"
                continue

            yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"

    except Exception as e:
        logger.error(f"流式生成失败: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"


@router.get("/ask/stream")
async def ask_question_stream(
    question: str = Query(..., description="问题内容"),
    top_k: int = Query(3, ge=1, le=10),
    use_cloud: bool = Query(False),
    model: str = Query("auto")
):
    """基于历史健康档案的智能问答（流式输出）"""
    if not question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    return StreamingResponse(
        generate_stream_response(question=question, top_k=top_k, use_cloud=use_cloud, model=model),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/analyze-trend")
async def analyze_health_trend(
    metric_name: str = Query(...),
    time_range: str = Query("all")
):
    """分析特定健康指标的历史趋势"""
    try:
        result = rag_service.analyze_health_trend(metric_name, time_range)
        if result.get("success"):
            return result
        else:
            return {"success": False, "message": result.get("message", "分析失败")}
    except Exception as e:
        logger.error(f"趋势分析失败: {str(e)}")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")


@router.get("/models")
async def get_available_models():
    """获取可用的 AI 模型列表"""
    try:
        ollama_status = rag_service.llm.check_ollama_health()
        if ollama_status:
            models = rag_service.llm.get_available_models()
            model_names = [m["name"] for m in models]
            auto_model = rag_service.llm.select_smartest_model()
            return {
                "success": True,
                "ollama_online": True,
                "models": model_names,
                "model_details": models,
                "default_model": rag_service.llm.default_model,
                "auto_model": auto_model,
                "message": f"Auto 模式将自动选择: {auto_model}"
            }
        else:
            return {
                "success": False,
                "ollama_online": False,
                "models": [],
                "message": "Ollama 服务未启动，请先运行 `ollama serve`"
            }
    except Exception as e:
        logger.error(f"获取模型列表失败: {str(e)}")
        return {
            "success": False,
            "ollama_online": False,
            "error": str(e)
        }


@router.get("/search/online")
async def online_search(
    query: str = Query(...),
    max_results: int = Query(5, ge=1, le=10)
):
    """使用 Tavily 进行联网搜索"""
    try:
        if not rag_service.tavily_enabled:
            return {
                "success": False,
                "error": "Tavily API Key 未配置，无法使用联网搜索功能",
                "results": []
            }

        result = rag_service.search_online(query, max_results)
        return result
    except Exception as e:
        logger.error(f"联网搜索失败: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "results": []
        }


@router.get("/search/status")
async def search_status():
    """获取联网搜索功能状态"""
    return {
        "enabled": rag_service.tavily_enabled,
        "provider": "Tavily Search" if rag_service.tavily_enabled else None,
        "quota": "每天 1000 次" if rag_service.tavily_enabled else None
    }


# --- 化验单解析 ---
@router.post("/api/lab-report/parse", response_model=LabReportResponse)
async def parse_lab_report(request: LabReportParseRequest):
    """解析化验单数据"""
    try:
        result = lab_report_parser.parse_report(request.report_type, request.indicators, request.month_age)
        return result
    except Exception as e:
        logger.error(f"化验单解析失败: {str(e)}")
        raise HTTPException(status_code=500, detail="化验单解析失败")


@router.post("/api/lab-report/evaluate")
async def evaluate_lab_report(request: LabReportParseRequest):
    """评估化验单指标"""
    try:
        result = lab_report_parser.evaluate_report(request.report_type, request.indicators, request.month_age)
        return result
    except Exception as e:
        logger.error(f"化验单评估失败: {str(e)}")
        raise HTTPException(status_code=500, detail="化验单评估失败")


# --- 症状自查 ---
@router.post("/api/symptom/analyze")
async def analyze_symptoms(request: SymptomCheckRequest):
    """分析症状"""
    try:
        result = symptom_checker.analyze(request.symptoms, request.month_age)
        return result
    except Exception as e:
        logger.error(f"症状分析失败: {str(e)}")
        raise HTTPException(status_code=500, detail="症状分析失败")


@router.get("/api/symptom/categories")
async def get_symptom_categories():
    """获取症状分类"""
    try:
        categories = symptom_checker.get_categories()
        return {"success": True, "categories": categories}
    except Exception as e:
        logger.error(f"获取症状分类失败: {str(e)}")
        raise HTTPException(status_code=500, detail="获取症状分类失败")


# --- 对话历史 ---
@router.post("/api/chat/sessions")
async def create_chat_session(data: dict):
    """创建对话会话"""
    try:
        session = chat_history_service.create_session(data.get("title", ""))
        return {"success": True, "session": session}
    except Exception as e:
        logger.error(f"创建对话会话失败: {str(e)}")
        raise HTTPException(status_code=500, detail="创建会话失败")


@router.get("/api/chat/sessions")
async def list_chat_sessions():
    """获取对话会话列表"""
    try:
        sessions = chat_history_service.list_sessions()
        return {"success": True, "sessions": sessions}
    except Exception as e:
        logger.error(f"获取对话列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail="获取对话列表失败")


@router.get("/api/chat/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    """获取会话消息"""
    try:
        messages = chat_history_service.get_messages(session_id)
        return {"success": True, "messages": messages}
    except Exception as e:
        logger.error(f"获取对话消息失败: {str(e)}")
        raise HTTPException(status_code=500, detail="获取消息失败")


@router.delete("/api/chat/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    """删除对话会话"""
    try:
        success = chat_history_service.delete_session(session_id)
        if success:
            return {"success": True, "message": "会话已删除"}
        raise HTTPException(status_code=500, detail="删除失败")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除对话会话失败: {str(e)}")
        raise HTTPException(status_code=500, detail="删除会话失败")


# --- 知识库 ---
@router.get("/api/knowledge/search")
async def search_knowledge(
    query: str = Query(...),
    n_results: int = Query(3, ge=1, le=10)
):
    """搜索知识库"""
    try:
        results = rag_service.search_knowledge(query, n_results)
        return {"success": True, "results": results}
    except Exception as e:
        logger.error(f"搜索知识库失败: {str(e)}")
        raise HTTPException(status_code=500, detail="搜索知识库失败")


@router.get("/api/knowledge/status")
async def get_knowledge_status():
    """获取知识库状态"""
    try:
        status = rag_service.get_knowledge_status()
        return {"success": True, **status}
    except Exception as e:
        logger.error(f"获取知识库状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail="获取知识库状态失败")
