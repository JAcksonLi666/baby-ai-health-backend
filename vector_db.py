import chromadb
from chromadb.config import Settings
import numpy as np
import logging
import requests
import json
from typing import List, Dict, Optional, Tuple
from config import VECTOR_DB_DIR, EMBEDDING_MODEL, CHROMA_COLLECTION_NAME, MODEL_CACHE_DIR, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)


class VectorDBService:
    def __init__(self, preload_embedding: bool = False):
        self.client = chromadb.PersistentClient(
            path=str(VECTOR_DB_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.embedding_model = None
        self.embedding_initialized = False
        self.use_ollama_embedding = False
        
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"description": "宝宝健康档案向量数据库"}
        )
        
        logger.info(f"ChromaDB 初始化完成，集合名称: {CHROMA_COLLECTION_NAME}")
        
        if preload_embedding:
            self._init_embedding_model()
    
    def _check_ollama_embedding(self) -> bool:
        """检查 Ollama 是否有嵌入模型可用"""
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                if any("embed" in name.lower() for name in model_names):
                    logger.info(f"发现 Ollama 嵌入模型: {model_names}")
                    return True
                else:
                    logger.info("未发现 Ollama 嵌入模型，尝试下载 nomic-embed-text")
                    # 尝试自动下载
                    try:
                        requests.post(f"{OLLAMA_BASE_URL}/api/pull", 
                                    json={"name": "nomic-embed-text"}, 
                                    timeout=120)
                        logger.info("Ollama 嵌入模型下载成功")
                        return True
                    except:
                        return False
            return False
        except Exception as e:
            logger.warning(f"检查 Ollama 失败: {str(e)}")
            return False
    
    def _init_embedding_model(self):
        if self.embedding_initialized:
            return
            
        logger.info("正在初始化 embedding 模型...")
        
        # 优先使用 Ollama 嵌入
        if self._check_ollama_embedding():
            self.use_ollama_embedding = True
            self.embedding_initialized = True
            logger.info("使用 Ollama 嵌入模型")
            return
        
        # 尝试使用 sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(
                EMBEDDING_MODEL, 
                model_kwargs={"cache_dir": str(MODEL_CACHE_DIR)}
            )
            self.use_ollama_embedding = False
            self.embedding_initialized = True
            logger.info("Embedding 模型初始化完成 (sentence-transformers)")
        except Exception as e:
            logger.warning(f"sentence-transformers 模型初始化失败，将使用 Ollama 或简单向量: {str(e)}")
            self.embedding_model = None
            self.embedding_initialized = True

    def generate_embedding(self, text: str) -> List[float]:
        """生成文本嵌入向量"""
        if not text:
            logger.warning("输入文本为空，返回零向量")
            return [0.0] * 128
            
        try:
            self._init_embedding_model()
            
            # 使用 Ollama 嵌入
            if self.use_ollama_embedding:
                try:
                    response = requests.post(
                        f"{OLLAMA_BASE_URL}/api/embeddings",
                        json={"model": "nomic-embed-text", "prompt": text},
                        timeout=30
                    )
                    if response.status_code == 200:
                        result = response.json()
                        return result.get("embedding", [])
                    else:
                        logger.warning(f"Ollama 嵌入请求失败: {response.status_code}")
                        raise ValueError("Ollama 嵌入失败")
                except Exception as e:
                    logger.warning(f"Ollama 嵌入失败: {str(e)}")
                    raise
            
            # 使用 sentence-transformers
            if self.embedding_model is None:
                raise ValueError("Embedding 模型未加载")
                
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.warning(f"生成嵌入向量失败，使用简单向量: {str(e)}")
            import hashlib
            vec = [float(b) / 255.0 for b in hashlib.md5(text.encode()).digest()[:128]]
            vec.extend([0.0] * (128 - len(vec)))
            return vec

    def add_record(
        self,
        record_id: str,
        text: str,
        metadata: Dict,
        date: Optional[str] = None
    ) -> bool:
        """添加健康记录到向量数据库"""
        try:
            embedding = self.generate_embedding(text)
            
            doc_metadata = {
                "record_id": record_id,
                "date": date or "",
                "record_type": metadata.get("type", "general"),
                "filename": metadata.get("filename", ""),
                **metadata
            }
            
            self.collection.add(
                ids=[record_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[doc_metadata]
            )
            
            logger.info(f"成功添加记录到向量数据库: {record_id}")
            return True
            
        except Exception as e:
            logger.error(f"添加记录失败: {str(e)}")
            return False

    def search_similar(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """检索相似记录"""
        try:
            query_embedding = self.generate_embedding(query)
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_metadata,
                include=["documents", "metadatas", "distances"]
            )
            
            similar_records = []
            if results['ids'] and len(results['ids']) > 0:
                for i in range(len(results['ids'][0])):
                    record = {
                        "id": results['ids'][0][i],
                        "text": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "distance": results['distances'][0][i],
                        "similarity": 1 - results['distances'][0][i]
                    }
                    similar_records.append(record)
            
            logger.info(f"检索到 {len(similar_records)} 条相似记录")
            return similar_records
            
        except Exception as e:
            logger.error(f"检索失败: {str(e)}")
            return []

    def get_record(self, record_id: str) -> Optional[Dict]:
        """获取指定记录"""
        try:
            result = self.collection.get(
                ids=[record_id],
                include=["documents", "metadatas"]
            )
            
            if result['ids'] and len(result['ids']) > 0:
                return {
                    "id": result['ids'][0],
                    "text": result['documents'][0],
                    "metadata": result['metadatas'][0]
                }
            return None
            
        except Exception as e:
            logger.error(f"获取记录失败: {str(e)}")
            return None

    def delete_record(self, record_id: str) -> bool:
        """删除指定记录"""
        try:
            self.collection.delete(ids=[record_id])
            logger.info(f"成功删除记录: {record_id}")
            return True
        except Exception as e:
            logger.error(f"删除记录失败: {str(e)}")
            return False

    def get_all_records(self, limit: int = 100) -> List[Dict]:
        """获取所有记录"""
        try:
            result = self.collection.get(
                limit=limit,
                include=["documents", "metadatas"]
            )
            
            records = []
            if result['ids']:
                for i in range(len(result['ids'])):
                    records.append({
                        "id": result['ids'][i],
                        "text": result['documents'][i],
                        "metadata": result['metadatas'][i]
                    })
            
            logger.info(f"获取到 {len(records)} 条记录")
            return records
            
        except Exception as e:
            logger.error(f"获取记录列表失败: {str(e)}")
            return []

    def get_collection_stats(self) -> Dict:
        """获取数据库统计信息"""
        try:
            count = self.collection.count()
            return {
                "total_records": count,
                "collection_name": CHROMA_COLLECTION_NAME,
                "embedding_model": EMBEDDING_MODEL
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {str(e)}")
            return {}


vector_db_service = VectorDBService(preload_embedding=False)
