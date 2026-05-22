"""
Symptom Checker Service - Infant and toddler symptom classification
Provides symptom categorization and knowledge base retrieval.
Does NOT provide medical advice, only symptom classification and related knowledge lookup.
"""
import hashlib
import json
import logging
from typing import List, Dict, Optional

from knowledge_base import knowledge_service

logger = logging.getLogger(__name__)

# Built-in symptom classification tree for common infant/toddler symptoms
SYMPTOM_CATEGORIES = {
    "fever": {
        "name": "发热",
        "name_en": "Fever",
        "description": "婴幼儿体温异常升高，是身体对抗感染的一种防御反应",
        "symptoms": {
            "低热": {
                "description": "体温 37.3-38.5°C，可能伴有轻微不适",
                "possible_causes": [
                    "环境温度过高或穿着过多",
                    "轻度上呼吸道感染",
                    "疫苗接种后反应",
                ],
                "severity": "mild",
                "related_knowledge": ["kb_health_001", "kb_health_002"],
                "precautions": [
                    "保持室内通风，适当减少衣物",
                    "多喂水或奶，注意补充水分",
                    "定时测量体温，记录变化趋势",
                    "3个月以下婴儿发热(>=38°C)应立即就医",
                ],
            },
            "中热": {
                "description": "体温 38.5-39.5°C，可能伴有精神不振、食欲下降",
                "possible_causes": [
                    "病毒性上呼吸道感染（感冒）",
                    "幼儿急疹",
                    "中耳炎或尿路感染",
                ],
                "severity": "moderate",
                "related_knowledge": ["kb_health_001", "kb_health_002"],
                "precautions": [
                    "体温>=38.5°C或明显不适时可使用退热药",
                    "对乙酰氨基酚适用于2个月以上婴儿",
                    "布洛芬适用于6个月以上婴儿",
                    "不可同时使用两种退热药",
                    "发热超过3天或伴有其他严重症状应及时就医",
                ],
            },
            "高热": {
                "description": "体温 >39.5°C，可能伴有寒战、面色潮红或苍白",
                "possible_causes": [
                    "细菌感染（如肺炎、化脓性扁桃体炎）",
                    "病毒感染（如流感、手足口病）",
                    "泌尿系统感染",
                ],
                "severity": "severe",
                "related_knowledge": ["kb_health_001", "kb_health_002"],
                "precautions": [
                    "高热时应立即采取降温措施",
                    "注意观察是否有抽搐、意识模糊等表现",
                    "体温超过40°C应立即就医",
                    "记录发热时间、体温变化和伴随症状",
                    "就医时向医生详细描述发热规律和用药情况",
                ],
            },
        },
    },
    "respiratory": {
        "name": "呼吸系统",
        "name_en": "Respiratory",
        "description": "婴幼儿呼吸系统尚未发育成熟，容易受到外界病原体侵袭",
        "symptoms": {
            "咳嗽": {
                "description": "呼吸道受到刺激时产生的防御性反射动作",
                "possible_causes": [
                    "上呼吸道感染（感冒）",
                    "过敏性咳嗽或哮喘",
                    "气道异物",
                ],
                "severity": "moderate",
                "related_knowledge": ["kb_health_002"],
                "precautions": [
                    "保持室内空气湿润，使用加湿器",
                    "1岁以下不使用蜂蜜止咳",
                    "注意观察咳嗽的频率、时间和痰液情况",
                    "如咳嗽持续超过2周或伴有呼吸困难应就医",
                ],
            },
            "流鼻涕": {
                "description": "鼻腔分泌物增多，可为清水样、粘稠或脓性",
                "possible_causes": [
                    "普通感冒",
                    "过敏性鼻炎",
                    "鼻腔异物",
                ],
                "severity": "mild",
                "related_knowledge": ["kb_health_002"],
                "precautions": [
                    "使用生理盐水滴鼻软化鼻涕",
                    "可用吸鼻器帮助清除鼻涕",
                    "保持室内湿度适宜",
                    "脓性鼻涕持续超过10天应就医",
                ],
            },
            "打喷嚏": {
                "description": "鼻腔黏膜受到刺激后的反射动作",
                "possible_causes": [
                    "鼻腔刺激（灰尘、冷空气、光线）",
                    "过敏性鼻炎",
                    "感冒初期症状",
                ],
                "severity": "mild",
                "related_knowledge": [],
                "precautions": [
                    "注意观察是否伴有其他症状",
                    "保持室内清洁，减少灰尘和过敏原",
                    "频繁打喷嚏伴鼻痒、眼痒需考虑过敏",
                ],
            },
            "鼻塞": {
                "description": "鼻腔通道受阻，呼吸不畅，影响进食和睡眠",
                "possible_causes": [
                    "感冒或上呼吸道感染",
                    "过敏性鼻炎",
                    "腺样体肥大",
                ],
                "severity": "mild",
                "related_knowledge": [],
                "precautions": [
                    "使用生理盐水滴鼻缓解鼻塞",
                    "喂奶前先清理鼻腔，有助于进食",
                    "睡觉时可适当抬高头部",
                    "持续鼻塞影响睡眠和进食应就医",
                ],
            },
            "喘息": {
                "description": "呼吸时伴有高调的哮鸣音，呼气时更为明显",
                "possible_causes": [
                    "毛细支气管炎（常见于2岁以下）",
                    "哮喘",
                    "异物吸入",
                ],
                "severity": "severe",
                "related_knowledge": ["kb_health_002"],
                "precautions": [
                    "喘息伴有呼吸困难应立即就医",
                    "注意观察呼吸频率和面色",
                    "记录喘息发作的时间和诱因",
                    "避免接触烟雾、粉尘等刺激物",
                ],
            },
        },
    },
    "digestive": {
        "name": "消化系统",
        "name_en": "Digestive",
        "description": "婴幼儿消化系统发育不完善，容易出现消化功能紊乱",
        "symptoms": {
            "腹泻": {
                "description": "大便次数增多、性状变稀，可伴有腹痛和呕吐",
                "possible_causes": [
                    "病毒性肠炎（轮状病毒等）",
                    "细菌性肠炎",
                    "食物过敏或不耐受",
                ],
                "severity": "moderate",
                "related_knowledge": ["kb_health_003", "kb_diaper_002", "kb_diaper_003"],
                "precautions": [
                    "预防脱水是关键，使用口服补液盐III",
                    "腹泻期间继续喂养，不应禁食",
                    "注意观察尿量和精神状态",
                    "月龄<6月或大便带血应及时就医",
                ],
            },
            "呕吐": {
                "description": "胃内容物经口排出，可为溢奶（少量）或喷射性呕吐",
                "possible_causes": [
                    "喂养不当（过饱、吞入空气）",
                    "胃肠型感冒",
                    "肠套叠或幽门狭窄",
                ],
                "severity": "moderate",
                "related_knowledge": ["kb_feed_002"],
                "precautions": [
                    "少量多次喂养，喂奶后竖抱拍嗝",
                    "呕吐后暂停喂养30分钟再少量试喂",
                    "注意观察呕吐物的颜色和量",
                    "喷射性呕吐或呕吐物带血/胆汁应立即就医",
                ],
            },
            "便秘": {
                "description": "排便次数减少、粪便干硬、排便困难",
                "possible_causes": [
                    "饮食中纤维摄入不足",
                    "水分摄入不够",
                    "功能性便秘",
                ],
                "severity": "mild",
                "related_knowledge": ["kb_diaper_002", "kb_diaper_003"],
                "precautions": [
                    "增加水分摄入，适当添加富含纤维的辅食",
                    "顺时针按摩腹部促进肠蠕动",
                    "已添加辅食的婴儿可吃西梅泥、梨泥",
                    "便秘持续超过2周或伴腹痛、血便应就医",
                ],
            },
            "腹胀": {
                "description": "腹部膨隆，可能伴有肠鸣音亢进或排气增多",
                "possible_causes": [
                    "吞入过多空气（哭闹、喂奶姿势不当）",
                    "消化不良",
                    "肠绞痛（常见于3个月以内）",
                ],
                "severity": "mild",
                "related_knowledge": ["kb_cry_001", "kb_cry_002"],
                "precautions": [
                    "喂奶后充分拍嗝，减少空气吞入",
                    "顺时针按摩腹部",
                    "做排气操（蹬自行车动作）帮助排气",
                    "严重腹胀伴呕吐、不排便应警惕肠梗阻",
                ],
            },
            "食欲不振": {
                "description": "进食量明显减少或拒绝进食",
                "possible_causes": [
                    "口腔不适（出牙、口腔溃疡）",
                    "消化不良或积食",
                    "疾病前兆（感冒初期）",
                ],
                "severity": "mild",
                "related_knowledge": ["kb_feed_001", "kb_feed_004"],
                "precautions": [
                    "不强迫进食，尊重婴儿食欲",
                    "检查口腔是否有溃疡或出牙迹象",
                    "提供多样化食物，尝试不同口味和质地",
                    "食欲持续下降伴体重不增应就医",
                ],
            },
        },
    },
    "skin": {
        "name": "皮肤",
        "name_en": "Skin",
        "description": "婴幼儿皮肤薄嫩，容易受到外界刺激和过敏原影响",
        "symptoms": {
            "湿疹": {
                "description": "皮肤红斑、丘疹、渗出、结痂，伴有剧烈瘙痒",
                "possible_causes": [
                    "遗传过敏体质",
                    "环境因素（干燥、过热、刺激物）",
                    "食物过敏原（牛奶、鸡蛋等）",
                ],
                "severity": "moderate",
                "related_knowledge": ["kb_health_004"],
                "precautions": [
                    "保湿是基础，每日多次涂抹无刺激润肤剂",
                    "沐浴水温32-37°C，时间5-10分钟",
                    "穿纯棉宽松衣物，避免羊毛和化纤",
                    "严重湿疹应在医生指导下使用药物",
                ],
            },
            "荨麻疹": {
                "description": "皮肤出现风团（红色隆起皮疹），伴有瘙痒，时起时消",
                "possible_causes": [
                    "食物过敏（鸡蛋、牛奶、海鲜等）",
                    "药物过敏",
                    "病毒感染",
                ],
                "severity": "moderate",
                "related_knowledge": ["kb_health_004"],
                "precautions": [
                    "记录饮食和接触物，排查过敏原",
                    "避免搔抓，可冷敷缓解瘙痒",
                    "注意观察是否伴有呼吸急促或面部肿胀",
                    "出现呼吸困难或喉头水肿应立即就医",
                ],
            },
            "皮疹": {
                "description": "皮肤出现各种形态的异常改变（红斑、丘疹、水疱等）",
                "possible_causes": [
                    "病毒感染（幼儿急疹、手足口病、水痘）",
                    "药物过敏",
                    "接触性皮炎",
                ],
                "severity": "moderate",
                "related_knowledge": ["kb_health_004"],
                "precautions": [
                    "观察皮疹的形态、分布和变化",
                    "注意是否伴有发热或其他全身症状",
                    "不随意涂抹药膏，避免加重病情",
                    "皮疹迅速扩散或伴高热应就医",
                ],
            },
            "红臀": {
                "description": "臀部皮肤发红，严重时出现糜烂和溃疡",
                "possible_causes": [
                    "尿布更换不及时，皮肤长时间接触排泄物",
                    "真菌或细菌感染",
                    "腹泻期间频繁排便刺激皮肤",
                ],
                "severity": "mild",
                "related_knowledge": ["kb_diaper_001"],
                "precautions": [
                    "勤换尿布，保持臀部清洁干燥",
                    "每次更换尿布时用温水清洗，轻轻拍干",
                    "可涂抹含氧化锌的护臀膏",
                    "适当让臀部暴露在空气中晾干",
                ],
            },
        },
    },
    "sleep": {
        "name": "睡眠问题",
        "name_en": "Sleep Issues",
        "description": "婴幼儿睡眠模式与成人不同，容易出现各种睡眠相关问题",
        "symptoms": {
            "入睡困难": {
                "description": "难以在合理时间内入睡，需要长时间安抚",
                "possible_causes": [
                    "过度疲劳或过度刺激",
                    "作息不规律",
                    "分离焦虑（常见于8-12个月）",
                ],
                "severity": "mild",
                "related_knowledge": ["kb_sleep_001", "kb_sleep_002", "kb_sleep_003"],
                "precautions": [
                    "建立固定的睡前仪式（洗澡、抚触、讲故事）",
                    "注意捕捉犯困信号（揉眼、打哈欠）",
                    "保持卧室安静、昏暗、温度适宜",
                    "避免睡前过度兴奋的活动",
                ],
            },
            "夜醒频繁": {
                "description": "夜间醒来次数多，难以自行重新入睡",
                "possible_causes": [
                    "睡眠周期转换困难",
                    "饥饿或口渴",
                    "出牙不适或生病",
                ],
                "severity": "mild",
                "related_knowledge": ["kb_sleep_001", "kb_sleep_003"],
                "precautions": [
                    "排除生理需求（饥饿、尿布）",
                    "夜间喂奶保持安静，不开大灯",
                    "先观察几分钟再决定是否干预",
                    "避免过度响应影响自主入睡能力",
                ],
            },
            "惊厥": {
                "description": "睡眠中突然出现意识丧失、四肢抽动、双眼上翻",
                "possible_causes": [
                    "热性惊厥（高热时发生）",
                    "癫痫",
                    "低钙血症",
                ],
                "severity": "severe",
                "related_knowledge": ["kb_health_002"],
                "precautions": [
                    "保持冷静，记录惊厥持续时间和表现",
                    "让婴儿侧卧，防止呕吐物误吸",
                    "不要强行约束肢体或往嘴里塞东西",
                    "惊厥持续超过5分钟或频繁发作应立即就医",
                ],
            },
        },
    },
    "oral": {
        "name": "口腔",
        "name_en": "Oral",
        "description": "婴幼儿口腔问题较为常见，多与出牙和感染有关",
        "symptoms": {
            "出牙不适": {
                "description": "牙齿萌出期间牙龈肿胀、流涎增多、烦躁",
                "possible_causes": [
                    "乳牙萌出刺激牙龈",
                    "局部牙龈炎症",
                ],
                "severity": "mild",
                "related_knowledge": ["kb_oral_001"],
                "precautions": [
                    "提供安全的牙胶或冰镇毛巾让婴儿咬",
                    "用干净纱布轻轻按摩牙龈",
                    "流涎多时及时擦干，预防口水疹",
                    "出牙期间低热属正常现象，但高热需排查其他原因",
                ],
            },
            "口腔溃疡": {
                "description": "口腔黏膜出现圆形或椭圆形溃疡，伴有疼痛",
                "possible_causes": [
                    "病毒感染（如疱疹性口炎）",
                    "创伤（咬伤、硬物划伤）",
                    "营养不良或免疫力下降",
                ],
                "severity": "moderate",
                "related_knowledge": ["kb_oral_001"],
                "precautions": [
                    "保持口腔清洁，饭后用温水漱口",
                    "提供柔软、常温的食物，避免刺激性食物",
                    "多喂水，保持口腔湿润",
                    "溃疡超过2周不愈或反复发作应就医",
                ],
            },
            "鹅口疮": {
                "description": "口腔黏膜出现白色斑膜，不易擦除，强行擦除后可见红色创面",
                "possible_causes": [
                    "白色念珠菌感染",
                    "长期使用抗生素后菌群失调",
                    "奶嘴或奶瓶消毒不彻底",
                ],
                "severity": "moderate",
                "related_knowledge": ["kb_oral_001"],
                "precautions": [
                    "注意奶嘴、奶瓶和玩具的清洁消毒",
                    "哺乳前清洁乳头",
                    "遵医嘱使用抗真菌药物",
                    "不要强行擦除白色斑膜",
                ],
            },
        },
    },
    "eye": {
        "name": "眼部",
        "name_en": "Eye",
        "description": "婴幼儿眼部问题需引起重视，及时发现异常有助于早期干预",
        "symptoms": {
            "分泌物增多": {
                "description": "眼部分泌物（眼屎）明显增多，可为水样、粘液样或脓性",
                "possible_causes": [
                    "鼻泪管阻塞（常见于新生儿）",
                    "结膜炎（细菌性或病毒性）",
                    "上呼吸道感染伴随症状",
                ],
                "severity": "mild",
                "related_knowledge": [],
                "precautions": [
                    "用无菌生理盐水或温水棉签清洁眼部",
                    "从内眼角向外眼角轻轻擦拭",
                    "注意手部卫生，避免交叉感染",
                    "脓性分泌物持续不退或伴眼红应就医",
                ],
            },
            "红肿": {
                "description": "眼睑或眼白部分发红、肿胀",
                "possible_causes": [
                    "结膜炎",
                    "过敏",
                    "异物进入眼睛",
                ],
                "severity": "moderate",
                "related_knowledge": [],
                "precautions": [
                    "不要让婴儿揉眼睛",
                    "检查是否有异物进入眼睛",
                    "用清洁的湿棉巾轻轻擦拭眼周",
                    "红肿伴发热或视力异常应就医",
                ],
            },
            "流泪": {
                "description": "单眼或双眼流泪增多，无明显的刺激因素",
                "possible_causes": [
                    "鼻泪管阻塞或狭窄",
                    "倒睫",
                    "结膜炎",
                ],
                "severity": "mild",
                "related_knowledge": [],
                "precautions": [
                    "新生儿流泪多为鼻泪管发育不完善",
                    "可轻柔按摩内眼角下方鼻泪管区域",
                    "注意观察是否伴有眼红或分泌物",
                    "持续流泪超过6个月应就医评估",
                ],
            },
        },
    },
    "ear": {
        "name": "耳部",
        "name_en": "Ear",
        "description": "婴幼儿耳部问题常见于感冒后或出牙期间",
        "symptoms": {
            "抓耳朵": {
                "description": "婴儿频繁抓挠耳朵或拉扯耳垂",
                "possible_causes": [
                    "中耳炎（常继发于感冒）",
                    "出牙疼痛放射至耳部",
                    "外耳道湿疹",
                ],
                "severity": "moderate",
                "related_knowledge": [],
                "precautions": [
                    "注意是否伴有发热或感冒症状",
                    "观察是否有耳部分泌物或异味",
                    "不要自行用棉签清理耳道",
                    "抓耳朵伴发热、烦躁或睡眠不安应就医",
                ],
            },
            "耳部分泌物": {
                "description": "耳道有异常分泌物排出，可为液体、脓液或血性",
                "possible_causes": [
                    "中耳炎伴鼓膜穿孔",
                    "外耳道炎",
                    "耳道进水后感染",
                ],
                "severity": "moderate",
                "related_knowledge": [],
                "precautions": [
                    "不要自行清理耳道或滴入任何液体",
                    "保持耳道干燥，洗澡时注意防水",
                    "观察分泌物的颜色、气味和量",
                    "耳部分泌物伴发热或疼痛应就医",
                ],
            },
        },
    },
}

# Build a flat symptom name to category mapping for quick lookup
_SYMPTOM_TO_CATEGORY: Dict[str, tuple] = {}
for _cat_key, _cat_data in SYMPTOM_CATEGORIES.items():
    for _symptom_name, _symptom_data in _cat_data["symptoms"].items():
        _SYMPTOM_TO_CATEGORY[_symptom_name] = (_cat_key, _cat_data["name"], _symptom_data)


class SymptomChecker:
    """Symptom checker service for infant and toddler symptom classification.

    This service provides symptom categorization and related knowledge retrieval.
    It does NOT provide medical advice. Only classifies symptoms and retrieves
    relevant knowledge base entries.
    """

    def __init__(self):
        self.knowledge_base = knowledge_service
        self._analysis_cache: Dict[str, dict] = {}

    def get_all_categories(self) -> List[Dict]:
        """Get all symptom categories with their symptoms list."""
        categories = []
        for cat_key, cat_data in SYMPTOM_CATEGORIES.items():
            symptoms_list = []
            for sym_name, sym_data in cat_data["symptoms"].items():
                symptoms_list.append({
                    "name": sym_name,
                    "severity": sym_data["severity"],
                    "description": sym_data["description"],
                })
            categories.append({
                "key": cat_key,
                "name": cat_data["name"],
                "name_en": cat_data["name_en"],
                "description": cat_data["description"],
                "symptoms": symptoms_list,
            })
        return categories

    def _match_symptoms(self, input_symptoms: List[str]) -> List[Dict]:
        """Match input symptom names to the built-in classification tree.

        Supports exact match and fuzzy match (substring containment).
        Returns a list of matched symptom details.
        """
        matched = []
        seen_keys = set()

        for symptom in input_symptoms:
            symptom_stripped = symptom.strip()
            if not symptom_stripped:
                continue

            # Exact match first
            if symptom_stripped in _SYMPTOM_TO_CATEGORY:
                cat_key, cat_name, sym_data = _SYMPTOM_TO_CATEGORY[symptom_stripped]
                match_key = f"{cat_key}:{symptom_stripped}"
                if match_key not in seen_keys:
                    seen_keys.add(match_key)
                    matched.append({
                        "input": symptom_stripped,
                        "matched_name": symptom_stripped,
                        "category_key": cat_key,
                        "category_name": cat_name,
                        "data": sym_data,
                    })
                continue

            # Fuzzy match: check if input contains or is contained in any known symptom
            best_match = None
            for known_name, (cat_key, cat_name, sym_data) in _SYMPTOM_TO_CATEGORY.items():
                if symptom_stripped in known_name or known_name in symptom_stripped:
                    best_match = (known_name, cat_key, cat_name, sym_data)
                    break

            if best_match:
                known_name, cat_key, cat_name, sym_data = best_match
                match_key = f"{cat_key}:{known_name}"
                if match_key not in seen_keys:
                    seen_keys.add(match_key)
                    matched.append({
                        "input": symptom_stripped,
                        "matched_name": known_name,
                        "category_key": cat_key,
                        "category_name": cat_name,
                        "data": sym_data,
                    })

        return matched

    async def analyze_symptoms(
        self,
        symptoms: List[str],
        age_months: int,
        duration_days: Optional[int] = None,
        severity: Optional[int] = None,
    ) -> Dict:
        """Analyze symptoms and return classification results with related knowledge.

        This method does NOT provide medical advice. It only performs symptom
        classification and retrieves related knowledge base entries.

        Args:
            symptoms: List of symptom names provided by the user.
            age_months: Baby's age in months (0-144).
            duration_days: Optional duration of symptoms in days.
            severity: Optional user-reported severity (1-5).

        Returns:
            A dictionary containing matched categories, related knowledge,
            and general precautions.
        """
        # Build cache key
        cache_data = json.dumps(
            {"symptoms": sorted(symptoms), "age_months": age_months,
             "duration_days": duration_days, "severity": severity},
            sort_keys=True, ensure_ascii=False,
        )
        cache_key = hashlib.md5(cache_data.encode()).hexdigest()

        # Check cache
        cached = self._analysis_cache.get(cache_key)
        if cached:
            return cached

        matched_results = self._match_symptoms(symptoms)

        # Build analysis results grouped by category
        categories = []
        all_knowledge_ids = set()
        all_precautions = []
        overall_severity = "mild"

        severity_rank = {"mild": 1, "moderate": 2, "severe": 3}

        for match in matched_results:
            data = match["data"]

            # Collect knowledge IDs
            for kid in data.get("related_knowledge", []):
                all_knowledge_ids.add(kid)

            # Collect unique precautions
            for prec in data.get("precautions", []):
                if prec not in all_precautions:
                    all_precautions.append(prec)

            # Track overall severity
            sym_severity = data.get("severity", "mild")
            if severity_rank.get(sym_severity, 1) > severity_rank.get(overall_severity, 1):
                overall_severity = sym_severity

            # Adjust severity based on duration
            if duration_days and duration_days >= 7:
                if severity_rank.get(sym_severity, 1) < 3:
                    sym_severity = "moderate"
                    if severity_rank.get(sym_severity, 2) > severity_rank.get(overall_severity, 1):
                        overall_severity = sym_severity

            # Adjust severity based on user-reported severity
            if severity and severity >= 4:
                if severity_rank.get(sym_severity, 1) < 3:
                    sym_severity = "severe"
                    overall_severity = "severe"

            # Age-specific notes
            age_notes = self._get_age_specific_notes(age_months, match["category_key"], match["matched_name"])

            categories.append({
                "category": match["category_name"],
                "category_key": match["category_key"],
                "symptom": match["matched_name"],
                "description": data["description"],
                "possible_causes": data["possible_causes"],
                "severity": sym_severity,
                "related_knowledge": data.get("related_knowledge", []),
                "precautions": data.get("precautions", []),
                "age_notes": age_notes,
            })

        # Retrieve related knowledge from knowledge base
        knowledge_results = []
        if all_knowledge_ids:
            for entry in self.knowledge_base.entries:
                if entry["id"] in all_knowledge_ids:
                    knowledge_results.append({
                        "id": entry["id"],
                        "title": entry["title"],
                        "source": entry["source"],
                        "content": entry["content"][:300] + "...",
                    })

        # If no symptoms matched, try searching knowledge base with the input
        if not matched_results and symptoms:
            query = " ".join(symptoms)
            kb_search = self.knowledge_base.search(query, n_results=3)
            knowledge_results = kb_search.get("results", [])

        result = {
            "success": True,
            "disclaimer": "本服务仅提供症状分类和相关知识检索，不构成任何就医建议。如有疑虑请及时就医。",
            "age_months": age_months,
            "duration_days": duration_days,
            "matched_count": len(matched_results),
            "total_input": len(symptoms),
            "overall_severity": overall_severity,
            "categories": categories,
            "related_knowledge": knowledge_results,
            "general_precautions": all_precautions,
            "unmatched_symptoms": [
                s for s in symptoms
                if s.strip() and not any(
                    m["input"] == s.strip() for m in matched_results
                )
            ],
        }

        # Store in cache (limit size)
        if len(self._analysis_cache) > 100:
            keys = list(self._analysis_cache.keys())[:50]
            for k in keys:
                del self._analysis_cache[k]
        self._analysis_cache[cache_key] = result

        return result

    def _get_age_specific_notes(
        self, age_months: int, category_key: str, symptom_name: str
    ) -> List[str]:
        """Get age-specific precaution notes for a symptom."""
        notes = []

        if age_months < 3:
            notes.append("3个月以下婴儿免疫系统尚未发育成熟，任何异常症状都应引起重视")
            if category_key == "fever":
                notes.append("3个月以下婴儿发热(>=38°C)应立即就医，不可自行用药")

        if age_months < 6:
            if category_key == "digestive" and symptom_name == "腹泻":
                notes.append("6个月以下婴儿腹泻容易导致脱水，应密切观察尿量和精神状态")
            if category_key == "respiratory" and symptom_name == "喘息":
                notes.append("小婴儿喘息可能提示毛细支气管炎，需要医生评估")

        if age_months >= 6 and age_months <= 24:
            if category_key == "oral" and symptom_name == "出牙不适":
                notes.append("6-24个月是乳牙萌出高峰期，出牙不适在此年龄段较为常见")

        if age_months >= 12:
            if category_key == "sleep" and symptom_name == "入睡困难":
                notes.append("1岁以上幼儿入睡困难可能与作息不规律或白天活动量不足有关")

        return notes


# Module-level instance
symptom_checker = SymptomChecker()
