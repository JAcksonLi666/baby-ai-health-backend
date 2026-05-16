import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from config import VECTOR_DB_DIR, EMBEDDING_MODEL, CHROMA_COLLECTION_NAME, MODEL_CACHE_DIR

logger = logging.getLogger(__name__)


class VectorDBService:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(VECTOR_DB_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL, cache_folder=str(MODEL_CACHE_DIR))
        
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"description": "宝宝健康档案向量数据库"}
        )
        
        logger.info(f"ChromaDB 初始化完成，集合名称: {CHROMA_COLLECTION_NAME}")

    def generate_embedding(self, text: str) -> List[float]:
        """生成文本嵌入向量"""
        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"生成嵌入向量失败: {str(e)}")
            raise

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


vector_db_service = VectorDBService()
