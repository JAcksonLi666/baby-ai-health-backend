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
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='ch',
            use_gpu=True,
            show_log=False,
            rec_model_dir=str(MODEL_CACHE_DIR / "rec"),
            det_model_dir=str(MODEL_CACHE_DIR / "det"),
            cls_model_dir=str(MODEL_CACHE_DIR / "cls")
        )
        logger.info("PaddleOCR 服务初始化完成")

    def extract_text_from_image(self, image_path: str) -> str:
        """从图片中提取文字"""
        try:
            result = self.ocr.ocr(image_path, cls=True)
            
            if not result or not result[0]:
                logger.warning(f"未能在图片 {image_path} 中检测到文字")
                return ""
            
            extracted_lines = []
            for line in result[0]:
                if line and len(line) >= 2:
                    text = line[1][0]
                    confidence = line[1][1]
                    if confidence > 0.5:
                        extracted_lines.append(text)
            
            full_text = '\n'.join(extracted_lines)
            logger.info(f"成功从 {image_path} 提取 {len(extracted_lines)} 行文字")
            return full_text
            
        except Exception as e:
            logger.error(f"OCR 处理失败: {str(e)}")
            raise

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """从 PDF 中提取文字（需要先转换为图片）"""
        try:
            from pdf2image import convert_from_path
            
            images = convert_from_path(pdf_path, dpi=200)
            all_text = []
            
            for i, image in enumerate(images):
                temp_image_path = f"/tmp/pdf_page_{i}.png"
                image.save(temp_image_path, "PNG")
                text = self.extract_text_from_image(temp_image_path)
                all_text.append(text)
            
            full_text = '\n\n'.join(all_text)
            logger.info(f"成功从 PDF {pdf_path} 提取 {len(images)} 页文字")
            return full_text
            
        except ImportError:
            logger.warning("pdf2image 未安装，尝试使用其他方法")
            return ""
        except Exception as e:
            logger.error(f"PDF 处理失败: {str(e)}")
            return ""

    def desensitize_text(self, text: str) -> str:
        """脱敏处理：移除个人身份信息"""
        desensitized = text
        
        desensitized = re.sub(r'姓名[：:]\s*[^\n]+', '姓名：[已脱敏]', desensitized)
        desensitized = re.sub(r'身份证[号]?[：:]\s*[^\n]+', '身份证：[已脱敏]', desensitized)
        desensitized = re.sub(r'电话[：:]\s*[^\n]+', '电话：[已脱敏]', desensitized)
        desensitized = re.sub(r'手机[：:]\s*[^\n]+', '手机：[已脱敏]', desensitized)
        desensitized = re.sub(r'\d{11}', '[手机号]', desensitized)
        desensitized = re.sub(r'\d{17}[\dXx]', '[身份证号]', desensitized)
        desensitized = re.sub(r'\d{3,4}[-\s]?\d{3,4}[-\s]?\d{3,4}', '[电话号码]', desensitized)
        
        logger.info("文字脱敏处理完成")
        return desensitized

    def parse_health_metrics(self, text: str) -> List[Dict]:
        """解析健康指标"""
        metrics = []
        
        patterns = [
            r'([\u4e00-\u9fa5a-zA-Z]+)[\s:：]*(\d+\.?\d*)\s*([\u00μ%/g/L/mL]+)?',
            r'([\u4e00-\u9fa5a-zA-Z]+)[\s:：]*(↑|↓|正常|异常)',
            r'(白细胞|红细胞|血红蛋白|血小板|中性粒细胞|淋巴细胞)\s*[:：]?\s*(\d+\.?\d*)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                metric_name = match.group(1).strip()
                metric_value = match.group(2).strip() if len(match.groups()) >= 2 else "N/A"
                metric_unit = match.group(3).strip() if len(match.groups()) >= 3 else ""
                
                metrics.append({
                    "name": metric_name,
                    "value": float(metric_value) if metric_value.replace('.', '').isdigit() else metric_value,
                    "unit": metric_unit
                })
        
        logger.info(f"解析出 {len(metrics)} 个健康指标")
        return metrics


ocr_service = OCRService()
