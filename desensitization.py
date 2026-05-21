"""
数据脱敏模块
对敏感信息进行脱敏处理，保护用户隐私
"""
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DataDesensitizer:
    """数据脱敏器"""

    # 脱敏规则
    PATTERNS = {
        'name': {
            'pattern': r'([\u4e00-\u9fa5]{2,4})',
            'description': '中文姓名',
        },
        'id_card': {
            'pattern': r'(\d{6})(\d{4})(\d{4})(\d{3}[\dXx])',
            'description': '身份证号',
        },
        'phone': {
            'pattern': r'(1[3-9]\d)(\d{4})(\d{4})',
            'description': '手机号',
        },
        'address': {
            'pattern': r'([\u4e00-\u9fa5]{2,6}(省|市|区|县|镇|乡|村|路|号|幢|栋|单元|室|楼))',
            'description': '地址信息',
        },
        'email': {
            'pattern': r'(\w{1,3})(\w*)(@\w+\.\w+)',
            'description': '邮箱地址',
        },
        'bank_card': {
            'pattern': r'(\d{4})(\d+)(\d{4})',
            'description': '银行卡号',
        },
    }

    @classmethod
    def desensitize(cls, text: str, rules: Optional[list] = None) -> str:
        """
        对文本进行脱敏处理

        Args:
            text: 原始文本
            rules: 需要应用的脱敏规则列表，None 表示应用所有规则

        Returns:
            脱敏后的文本
        """
        if not text or not isinstance(text, str):
            return text

        result = text
        active_rules = rules if rules else list(cls.PATTERNS.keys())

        for rule_name in active_rules:
            if rule_name not in cls.PATTERNS:
                continue
            result = cls._apply_rule(result, rule_name)

        return result

    @classmethod
    def _apply_rule(cls, text: str, rule_name: str) -> str:
        """应用单个脱敏规则"""
        rule = cls.PATTERNS[rule_name]
        pattern = rule['pattern']

        try:
            if rule_name == 'name':
                # 姓名脱敏: 张三 -> 张*
                result = re.sub(
                    pattern,
                    lambda m: m.group(1)[0] + '*' * (len(m.group(1)) - 1),
                    text
                )
            elif rule_name == 'id_card':
                # 身份证脱敏: 110101199001011234 -> 110101****011234
                result = re.sub(
                    pattern,
                    r'\1****\3\4',
                    text
                )
            elif rule_name == 'phone':
                # 手机号脱敏: 13812345678 -> 138****5678
                result = re.sub(
                    pattern,
                    r'\1****\3',
                    text
                )
            elif rule_name == 'address':
                # 地址脱敏: 北京市朝阳区XX路XX号 -> 北京市朝阳区****
                result = re.sub(
                    pattern,
                    lambda m: m.group(1)[:3] + '****',
                    text
                )
            elif rule_name == 'email':
                # 邮箱脱敏: abc@example.com -> a**@example.com
                result = re.sub(
                    pattern,
                    r'\1**\3',
                    text
                )
            elif rule_name == 'bank_card':
                # 银行卡脱敏: 6222021234567890123 -> 6222********0123
                result = re.sub(
                    pattern,
                    r'\1********\3',
                    text
                )
            else:
                result = text
            return result
        except re.error:
            logger.warning(f"脱敏规则 {rule_name} 正则匹配失败")
            return text

    @classmethod
    def detect_sensitive_info(cls, text: str) -> list:
        """
        检测文本中的敏感信息

        Returns:
            包含敏感信息类型的列表
        """
        if not text or not isinstance(text, str):
            return []

        detected = []
        for rule_name, rule in cls.PATTERNS.items():
            try:
                if re.search(rule['pattern'], text):
                    detected.append({
                        'type': rule_name,
                        'description': rule['description'],
                    })
            except re.error:
                continue

        return detected

    @classmethod
    def desensitize_log(cls, message: str) -> str:
        """脱敏日志消息"""
        return cls.desensitize(message, rules=['phone', 'id_card', 'email', 'bank_card'])


# 便捷函数
def desensitize_text(text: str, rules: Optional[list] = None) -> str:
    """脱敏文本"""
    return DataDesensitizer.desensitize(text, rules)


def detect_sensitive(text: str) -> list:
    """检测敏感信息"""
    return DataDesensitizer.detect_sensitive_info(text)


def desensitize_log(message: str) -> str:
    """脱敏日志"""
    return DataDesensitizer.desensitize_log(message)
