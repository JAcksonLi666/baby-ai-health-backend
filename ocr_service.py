import re
from typing import List, Dict, Optional
import logging
from config import MODEL_CACHE_DIR
import os

logger = logging.getLogger(__name__)

# 尝试导入不同的OCR库
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    logger.warning("PaddleOCR 未安装")

try:
    import pytesseract
    from PIL import Image, ImageEnhance
    
    # 设置 Tesseract 路径（优先使用环境变量，其次自动检测）
    from config import TESSERACT_CMD

    tesseract_cmd = TESSERACT_CMD
    if tesseract_cmd and os.path.exists(tesseract_cmd):
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        logger.info(f"Tesseract 路径已从环境变量加载: {tesseract_cmd}")
    else:
        # 自动检测常见安装路径
        common_paths = []
        if os.name == 'nt':  # Windows
            common_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            ]
            # 尝试从 PATH 或注册表查找
            import shutil
            which_result = shutil.which('tesseract')
            if which_result:
                common_paths.insert(0, which_result)
        else:  # Linux / macOS
            common_paths = ['/usr/bin/tesseract', '/usr/local/bin/tesseract']

        for path in common_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                logger.info(f"Tesseract 路径已自动检测: {path}")
                break
    
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("Tesseract 未安装")

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV 未安装")


class OCRService:
    def __init__(self):
        self.paddle_ocr = None
        self.use_tesseract_fallback = False
        logger.info("OCR 服务初始化完成，支持 PaddleOCR 和 Tesseract")
    
    def _init_paddle_ocr(self):
        """初始化 PaddleOCR"""
        if self.paddle_ocr is None and PADDLE_AVAILABLE:
            logger.info("正在初始化 PaddleOCR...")
            try:
                self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='ch')
                logger.info("PaddleOCR 初始化成功")
            except Exception as e:
                logger.error(f"PaddleOCR 初始化失败: {e}")
                self.use_tesseract_fallback = True

    def _preprocess_image_cv2(self, image_path: str) -> str:
        """使用 OpenCV 进行图片预处理"""
        if not CV2_AVAILABLE:
            return image_path
            
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"无法读取图片: {image_path}")
                return image_path
            
            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 自适应阈值二值化
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # 形态学操作
            kernel = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # 保存预处理后的图片
            processed_path = image_path.replace('.', '_processed.')
            cv2.imwrite(processed_path, cleaned)
            logger.info(f"图片预处理完成: {processed_path}")
            return processed_path
            
        except Exception as e:
            logger.error(f"图片预处理失败: {e}")
            return image_path

    def _preprocess_image_pil(self, image_path: str) -> str:
        """使用 PIL 进行图片预处理"""
        if not TESSERACT_AVAILABLE:
            return image_path
            
        try:
            img = Image.open(image_path)
            
            # 转换为灰度图
            if img.mode != 'L':
                img = img.convert('L')
            
            # 增强对比度
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.0)
            
            # 二值化处理
            threshold = 127
            img = img.point(lambda p: p > threshold and 255)
            
            # 保存预处理后的图片
            processed_path = image_path.replace('.', '_processed_pil.')
            img.save(processed_path)
            logger.info(f"PIL图片预处理完成: {processed_path}")
            return processed_path
            
        except Exception as e:
            logger.error(f"PIL图片预处理失败: {e}")
            return image_path

    def _extract_with_paddle(self, image_path: str) -> str:
        """使用 PaddleOCR 提取文字"""
        if not PADDLE_AVAILABLE:
            return ""
            
        try:
            self._init_paddle_ocr()
            if self.paddle_ocr is None:
                return ""
                
            result = self.paddle_ocr.ocr(image_path)
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
            logger.error(f"PaddleOCR 识别失败: {e}")
            self.use_tesseract_fallback = True
            return ""

    def _extract_with_tesseract(self, image_path: str) -> str:
        """使用 Tesseract 提取文字"""
        if not TESSERACT_AVAILABLE:
            return ""
            
        try:
            # 先进行预处理
            processed_path = self._preprocess_image_pil(image_path)
            
            # 使用 Tesseract 识别
            custom_config = r'--oem 3 --psm 6 -l chi_sim+eng'
            text = pytesseract.image_to_string(Image.open(processed_path), config=custom_config)
            
            return text.strip()
        except Exception as e:
            logger.error(f"Tesseract 识别失败: {e}")
            return ""

    def extract_text_from_image(self, image_path: str) -> str:
        """从图片中提取文字，优先使用 PaddleOCR，失败时回退到 Tesseract"""
        # 检查文件是否存在
        if not os.path.exists(image_path):
            logger.error(f"图片文件不存在: {image_path}")
            return ""
        
        # 尝试多种识别方式
        results = []
        
        # 方式1: 使用 PaddleOCR + OpenCV 预处理
        if PADDLE_AVAILABLE and not self.use_tesseract_fallback:
            processed_path = self._preprocess_image_cv2(image_path)
            result = self._extract_with_paddle(processed_path)
            if result:
                results.append(result)
        
        # 方式2: 如果 PaddleOCR 失败或已切换，使用原始图片
        if not results and PADDLE_AVAILABLE and not self.use_tesseract_fallback:
            result = self._extract_with_paddle(image_path)
            if result:
                results.append(result)
        
        # 方式3: 使用 Tesseract（备选方案）
        if not results and TESSERACT_AVAILABLE:
            result = self._extract_with_tesseract(image_path)
            if result:
                results.append(result)
        
        # 返回最长的识别结果
        if results:
            best_result = max(results, key=len)
            logger.info(f"OCR 识别成功，识别到 {len(best_result)} 个字符")
            return best_result
        
        logger.warning("所有 OCR 方式均未能识别到文字")
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

    def desensitize_text(self, text: str) -> str:
        """脱敏处理：去除姓名、身份证号、手机号等敏感信息"""
        if not text:
            return text
        
        desensitized = text
        
        # 脱敏姓名（保留姓氏，隐藏名字）
        name_patterns = [
            r'姓名[：:\s]*([\u4e00-\u9fa5]{2,4})',
            r'姓\s*名[：:\s]*([\u4e00-\u9fa5]{2,4})',
        ]
        for pattern in name_patterns:
            matches = re.findall(pattern, desensitized)
            for match in matches:
                if len(match) >= 2:
                    masked_name = match[0] + '*' * (len(match) - 1)
                    desensitized = desensitized.replace(match, masked_name)
        
        # 脱敏身份证号（只显示前三位和后四位）
        id_pattern = r'\b(\d{3})\d{11}(\d{4})\b'
        desensitized = re.sub(id_pattern, r'\1************\2', desensitized)
        
        # 脱敏手机号（只显示前三位和后四位）
        phone_pattern = r'\b(\d{3})\d{4}(\d{4})\b'
        desensitized = re.sub(phone_pattern, r'\1****\2', desensitized)
        
        # 脱敏地址（只保留省市区）
        address_pattern = r'([^省市区县]{2,6}省[^市区县]{2,6}市[^区县]{2,6}区?)(.*?)(?=邮编|$)'
        desensitized = re.sub(address_pattern, r'\1***', desensitized)
        
        return desensitized

    def parse_health_metrics(self, text: str) -> List[Dict]:
        """从脱敏后的文本中解析健康指标"""
        if not text:
            return []
        
        metrics = []
        
        # 常见血液检测指标模式
        blood_patterns = {
            'WBC': (r'白细胞.*?(\d+\.?\d*)\s*(?:10\^9/L|10\*9/L)', '白细胞计数'),
            'RBC': (r'红细胞.*?(\d+\.?\d*)\s*(?:10\^12/L|10\*12/L)', '红细胞计数'),
            'HGB': (r'血红蛋白.*?(\d+\.?\d*)\s*(?:g/L|g/L)', '血红蛋白'),
            'PLT': (r'血小板.*?(\d+\.?\d*)\s*(?:10\^9/L|10\*9/L)', '血小板计数'),
            'MCV': (r'平均红细胞体积.*?(\d+\.?\d*)\s*(?:fL|fl)', '平均红细胞体积'),
            'MCH': (r'平均红细胞血红蛋白量.*?(\d+\.?\d*)\s*(?:pg)', '平均红细胞血红蛋白量'),
            'MCHC': (r'平均红细胞血红蛋白浓度.*?(\d+\.?\d*)\s*(?:g/L)', '平均红细胞血红蛋白浓度'),
            'LY': (r'淋巴细胞.*?(\d+\.?\d*)\s*(?:10\^9/L|%)', '淋巴细胞'),
            'NEUT': (r'中性粒细胞.*?(\d+\.?\d*)\s*(?:10\^9/L|%)', '中性粒细胞'),
        }
        
        for key, (pattern, name) in blood_patterns.items():
            matches = re.findall(pattern, text, re.I)
            for match in matches:
                value = float(match) if match else None
                if value:
                    metrics.append({
                        'name': name,
                        'code': key,
                        'value': value,
                        'type': 'blood'
                    })
        
        # 常见尿液检测指标
        urine_patterns = {
            'URO': (r'尿胆原.*?(\d+\.?\d*)', '尿胆原'),
            'BIL': (r'胆红素.*?(\d+\.?\d*)', '胆红素'),
            'KET': (r'酮体.*?(\d+\.?\d*)', '酮体'),
            'BLD': (r'潜血.*?(\d+\.?\d*)', '潜血'),
            'PRO': (r'蛋白质.*?(\d+\.?\d*)', '蛋白质'),
            'NIT': (r'亚硝酸盐.*?(\d+\.?\d*)', '亚硝酸盐'),
            'WBC_U': (r'白细胞.*?(\d+\.?\d*)\s*(?:/HP|/μL)', '尿白细胞'),
        }
        
        for key, (pattern, name) in urine_patterns.items():
            matches = re.findall(pattern, text, re.I)
            for match in matches:
                value = float(match) if match else None
                if value:
                    metrics.append({
                        'name': name,
                        'code': key,
                        'value': value,
                        'type': 'urine'
                    })
        
        return metrics

    def extract_date_from_text(self, text: str) -> Optional[str]:
        """从文本中识别日期，优先返回最可能的日期"""
        if not text:
            return None
        
        # 日期模式列表（按优先级排序）
        date_patterns = [
            # YYYY-MM-DD 格式（最常见）
            r'(\d{4})[\-/年](\d{1,2})[\-/月](\d{1,2})[日号]?',
            # YYYY/MM/DD 格式
            r'(\d{4})/(\d{1,2})/(\d{1,2})',
            # 采样时间/检验时间等
            r'(?:采样|检验|检测|日期).*?(\d{4})[\-/年](\d{1,2})[\-/月](\d{1,2})[日号]?',
            # 纯数字日期
            r'\b(\d{4})(\d{2})(\d{2})\b',
            # 中文日期
            r'(\d{4})年(\d{1,2})月(\d{1,2})日?'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    year = int(match[0])
                    month = int(match[1])
                    day = int(match[2])
                    
                    # 验证日期有效性
                    if 2000 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                        return f"{year:04d}-{month:02d}-{day:02d}"
                except (ValueError, IndexError):
                    continue
        
        return None

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
