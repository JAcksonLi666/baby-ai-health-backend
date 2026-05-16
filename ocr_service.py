import re
from typing import List, Dict, Optional
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import logging
from config import MODEL_CACHE_DIR

logger = logging.getLogger(__name__)


class OCRService:
    def __init__(self):
        self.ocr = None
        logger.info("PaddleOCR 服务延迟初始化，将在第一次使用时加载")
    
    def _init_ocr(self):
        if self.ocr is None:
            logger.info("正在初始化 PaddleOCR 服务...")
            self.ocr = PaddleOCR(use_angle_cls=True, lang='ch')
            logger.info("PaddleOCR 服务初始化完成")

    def extract_text_from_image(self, image_path: str) -> str:
        """从图片中提取文字"""
        try:
            self._init_ocr()
            result = self.ocr.ocr(image_path)
            
            if not result or not result[0]:
                return ""
            
            texts = []
            for line in result[0]:
                if len(line) >= 2:
                    text_info = line[1]
                    if isinstance(text_info, tuple) and len(text_info) >= 1:
                        texts.append(text_info[0])
            
            return "\n".join(texts)
        except Exception as e:
            logger.error(f"OCR 识别失败: {e}")
            return ""

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """从 PDF 中提取文字"""
        try:
            import fitz
            doc = fitz.open(pdf_path)
            all_text = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                all_text.append(text)
            return "\n".join(all_text)
        except ImportError:
            logger.warning("PyMuPDF 未安装，无法处理 PDF")
            return ""
        except Exception as e:
            logger.error(f"PDF 解析失败: {e}")
            return ""

    def extract_health_indicators(self, text: str) -> Dict[str, str]:
        """从 OCR 结果中提取健康指标"""
        indicators = {}
        
        weight_match = re.search(r'(?:体重|weight)\s*[:：]?\s*([\d.]+)\s*(kg|公斤)?', text, re.I)
        if weight_match:
            indicators['weight'] = weight_match.group(1)
        
        height_match = re.search(r'(?:身高|height)\s*[:：]?\s*([\d.]+)\s*(cm|厘米)?', text, re.I)
        if height_match:
            indicators['height'] = height_match.group(1)
        
        temp_match = re.search(r'(?:体温|temperature)\s*[:：]?\s*([\d.]+)\s*[°Cc]?', text, re.I)
        if temp_match:
            indicators['temperature'] = temp_match.group(1)
        
        return indicators


ocr_service = OCRService()
