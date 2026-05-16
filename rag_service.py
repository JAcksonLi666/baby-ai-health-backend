import logging
from typing import List, Dict
from vector_db import vector_db_service
from llm_service import llm_service
from config import OLLAMA_MODEL
import re

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        self.vector_db = vector_db_service
        self.llm = llm_service
        logger.info("RAG 服务初始化完成")

    def build_medical_prompt(
        self,
        question: str,
        context_records: List[Dict],
        include_history: bool = True
    ) -> str:
        """构建医疗问答 Prompt"""
        
        system_prompt = """你是一个专业的儿科医疗健康助手。请根据提供的健康档案信息，回答用户的问题。

重要提示：
1. 你的回答仅供参考，不能替代执业医师的诊断和治疗
2. 如果涉及具体医疗决策，请务必咨询专业医生
3. 请基于客观数据进行分析，避免主观臆断
4. 对于婴儿健康问题，建议优先咨询儿科医生

"""

        user_prompt = f"用户问题：{question}\n\n"

        if include_history and context_records:
            user_prompt += "相关健康档案信息：\n"
            user_prompt += "-" * 50 + "\n"
            
            for i, record in enumerate(context_records, 1):
                date = record.get("metadata", {}).get("date", "未知日期")
                record_type = record.get("metadata", {}).get("record_type", "一般记录")
                text = record.get("text", "")
                similarity = record.get("similarity", 0)
                
                user_prompt += f"\n【记录 {i}】日期：{date}，类型：{record_type}，相关度：{similarity:.2%}\n"
                user_prompt += f"内容：{text}\n"
            
            user_prompt += "-" * 50 + "\n\n"
        else:
            user_prompt += "（暂无相关历史档案）\n\n"

        user_prompt += """请基于以上信息，提供：
1. 简要分析
2. 可能的解读
3. 建议（需明确标注仅供参考）
"""

        full_prompt = system_prompt + user_prompt
        logger.info(f"构建 Prompt 成功，上下文记录数: {len(context_records)}")
        
        return full_prompt

    def answer_question(
        self,
        question: str,
        top_k: int = 3,
        use_cloud: bool = False,
        model: str = None
    ) -> Dict:
        """基于 RAG 回答问题"""
        try:
            context_records = self.vector_db.search_similar(question, top_k=top_k)
            
            prompt = self.build_medical_prompt(question, context_records)
            
            if use_cloud:
                llm_result = self.llm.generate_cloud(prompt)
            else:
                llm_result = self.llm.generate_local(prompt, model=model)
            
            if llm_result.get("success"):
                return {
                    "success": True,
                    "answer": llm_result["response"],
                    "sources": [
                        {
                            "id": record["id"],
                            "date": record.get("metadata", {}).get("date", ""),
                            "type": record.get("metadata", {}).get("record_type", ""),
                            "similarity": record.get("similarity", 0)
                        }
                        for record in context_records
                    ],
                    "model_used": llm_result.get("model", model or OLLAMA_MODEL),
                    "cloud_used": use_cloud,
                    "context_count": len(context_records)
                }
            else:
                return {
                    "success": False,
                    "error": llm_result.get("error", "生成失败"),
                    "sources": [],
                    "model_used": model or OLLAMA_MODEL,
                    "cloud_used": use_cloud
                }
                
        except Exception as e:
            logger.error(f"RAG 回答失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "sources": [],
                "model_used": model or OLLAMA_MODEL,
                "cloud_used": use_cloud
            }

    def analyze_health_trend(
        self,
        metric_name: str,
        time_range: str = "all"
    ) -> Dict:
        """分析特定健康指标的趋势"""
        try:
            query = f"{metric_name} 指标趋势分析"
            records = self.vector_db.search_similar(query, top_k=20)
            
            if not records:
                return {
                    "success": False,
                    "error": "未找到相关历史数据",
                    "metric": metric_name
                }
            
            analysis_prompt = f"""请分析以下{metric_name}的历史变化趋势：

"""
            for record in records:
                text = record.get("text", "")
                if metric_name.lower() in text.lower():
                    analysis_prompt += f"- {text}\n"
            
            analysis_prompt += """
请给出：
1. 总体趋势（上升/下降/稳定）
2. 是否在正常范围内
3. 建议

注意：仅供参考，请咨询专业医生。
"""
            
            result = self.llm.generate_local(analysis_prompt)
            
            if result.get("success"):
                return {
                    "success": True,
                    "analysis": result["response"],
                    "records_analyzed": len(records),
                    "metric": metric_name,
                    "time_range": time_range
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "分析失败"),
                    "metric": metric_name
                }
                
        except Exception as e:
            logger.error(f"趋势分析失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "metric": metric_name
            }


rag_service = RAGService()
