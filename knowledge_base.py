"""
知识库服务 - 国家卫健委婴幼儿照护指南
BM25 + 向量 + 关键词混合检索
"""
import json
import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from threading import Lock

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

from config import KNOWLEDGE_DATA_DIR

logger = logging.getLogger(__name__)

KNOWLEDGE_ENTRIES = [
    # 1. 母乳喂养频率与时长
    {
        "id": "kb_feed_001",
        "title": "母乳喂养频率与时长",
        "source": "国家卫健委《3岁以下婴幼儿健康养育照护指南（试行）》",
        "content": "母乳是婴儿最理想的天然食物。出生后应尽早开奶（半小时内），按需哺乳，每日不少于8次。新生儿期每次哺乳时间约15-20分钟，两侧乳房交替喂养。纯母乳喂养应持续至6个月，之后在添加辅食的基础上继续母乳喂养至24个月及以上。母亲应保持良好的营养状态和心情，保证充足的休息和水分摄入。哺乳前应洗手并清洁乳头，哺乳后可挤出少量乳汁涂抹在乳头上有助于预防乳头皲裂。",
        "keywords": ["母乳", "喂养", "哺乳", "开奶", "按需哺乳", "母乳喂养"]
    },
    # 2. 配方奶喂养量
    {
        "id": "kb_feed_002",
        "title": "配方奶喂养量",
        "source": "国家卫健委《3岁以下婴幼儿健康养育照护指南（试行）》",
        "content": "当无法进行母乳喂养时，应选择适合婴儿年龄段的配方奶粉。新生儿期（0-1月）每次喂奶量约60-90ml，每日7-8次；1-2月每次约90-120ml，每日6-7次；2-4月每次约120-150ml，每日5-6次；4-6月每次约150-210ml，每日4-5次。冲泡奶粉应严格按照说明书的比例，水温控制在40-50°C，先加水后加奶粉。喂奶后应竖抱拍嗝，防止溢奶。奶瓶、奶嘴每次使用后应彻底清洗消毒。",
        "keywords": ["配方奶", "奶粉", "奶量", "冲泡", "奶瓶", "人工喂养"]
    },
    # 3. 辅食添加时间与顺序
    {
        "id": "kb_feed_003",
        "title": "辅食添加时间与顺序",
        "source": "国家卫健委《3岁以下婴幼儿健康养育照护指南（试行）》",
        "content": "婴儿满6个月（180天）起应开始添加辅食，不能早于4个月或晚于8个月。添加顺序建议：首先添加富含铁的泥糊状食物，如强化铁米粉；然后依次添加蔬菜泥（如胡萝卜、南瓜、菠菜）、水果泥（如苹果、香蕉、梨）；7-8个月可添加肉泥、肝泥、鱼泥；9-10个月可添加碎末状食物；11-12个月可添加软饭、碎肉、碎菜。每种新食物应单独添加，观察3-5天确认无过敏反应后再添加下一种。",
        "keywords": ["辅食", "添加", "米粉", "断奶", "泥糊", "铁"]
    },
    # 4. 辅食添加原则
    {
        "id": "kb_feed_004",
        "title": "辅食添加原则",
        "source": "国家卫健委《3岁以下婴幼儿健康养育照护指南（试行）》",
        "content": "辅食添加应遵循以下原则：①由少到多：从1勺开始，逐渐增加量；②由稀到稠：从流质到半流质再到固体；③由细到粗：从泥糊状到碎末状再到小块状；④由单一到多样：每次只添加一种新食物；⑤不强迫进食：尊重婴儿的食欲和饱腹感信号；⑥不加调味品：1岁以内不加盐、糖、酱油等调味品；⑦注意过敏：观察是否有皮疹、腹泻、呕吐等过敏反应。辅食添加期间应继续母乳或配方奶喂养，奶量不应减少。",
        "keywords": ["辅食", "原则", "过敏", "调味品", "加盐", "添加原则"]
    },
    # 5. 每日睡眠时长推荐
    {
        "id": "kb_sleep_001",
        "title": "每日睡眠时长推荐",
        "source": "国家卫健委《3岁以下婴幼儿健康养育照护指南（试行）》",
        "content": "充足的睡眠对婴幼儿生长发育至关重要。各年龄段推荐睡眠时长：新生儿（0-1月）每日14-17小时，包括夜间和白天小睡；1-3月每日12-16小时；4-11月每日12-15小时，白天小睡2-3次；1-2岁每日11-14小时，白天小睡1-2次；2-3岁每日10-13小时，白天小睡1次。每个婴儿的睡眠需求存在个体差异，只要精神状态好、生长发育正常，略少于推荐时长也是正常的。应建立规律的作息时间，培养自主入睡能力。",
        "keywords": ["睡眠", "时长", "睡觉", "小睡", "作息", "睡眠时间"]
    },
    # 6. 睡眠环境安全
    {
        "id": "kb_sleep_002",
        "title": "睡眠环境安全",
        "source": "国家卫健委《3岁以下婴幼儿健康养育照护指南（试行）》",
        "content": "安全的睡眠环境是预防婴儿猝死综合征（SIDS）的关键。婴儿应单独睡在自己的婴儿床中，不应与成人同床。婴儿床应使用硬质床垫，床单应紧贴床垫。1岁以内婴儿仰卧位睡眠最为安全，侧卧位也有一定风险，俯卧位（趴睡）显著增加SIDS风险。婴儿床上不应放置枕头、毛绒玩具、宽松的毯子或防撞垫等软物。室温保持在20-22°C为宜，避免过度包裹。使用睡袋代替被子可降低窒息风险。",
        "keywords": ["睡眠安全", "婴儿床", "仰卧", "猝死", "SIDS", "趴睡"]
    },
    # 7. 夜间喂养与睡眠
    {
        "id": "kb_sleep_003",
        "title": "夜间喂养与睡眠",
        "source": "国家卫健委《3岁以下婴幼儿健康养育照护指南（试行）》",
        "content": "新生儿期夜间需要喂养2-3次，通常每3-4小时喂一次。随着婴儿月龄增加，夜间喂养次数逐渐减少。3-4个月时多数婴儿可以连续睡眠5-6小时，6个月后部分婴儿可以整夜不喂。培养良好的夜间睡眠习惯：建立固定的睡前仪式（如洗澡、抚触、讲故事），区分白天和夜晚的环境（白天明亮、夜晚安静昏暗），夜间喂奶保持安静、不开大灯。如果婴儿夜间醒来，先观察几分钟再决定是否需要干预，避免过度响应影响自主入睡能力。",
        "keywords": ["夜奶", "夜间", "夜醒", "入睡", "睡前", "哄睡"]
    },
    # 8. 尿布更换频率
    {
        "id": "kb_diaper_001",
        "title": "尿布更换频率",
        "source": "国家卫健委《3岁以下婴幼儿健康养育照护指南（试行）》",
        "content": "新生儿期每日排尿6-8次以上，排便次数不定（母乳喂养婴儿可能每次喂奶后都排便）。建议每2-3小时检查一次尿布，及时更换。大便后应立即更换，即使尿布只是湿了也应尽快更换以预防尿布疹。更换尿布时用温水清洗臀部，用柔软的毛巾轻轻擦干，可涂抹护臀膏预防尿布疹。选择透气性好、吸水性强的尿布，避免过紧。观察尿量和尿色，正常尿液应为淡黄色，如尿量明显减少或尿液颜色深黄，需注意是否摄入水分不足。",
        "keywords": ["尿布", "换尿布", "纸尿裤", "尿量", "尿不湿", "换纸尿裤"]
    },
    # 9. 便便颜色与质地
    {
        "id": "kb_diaper_002",
        "title": "便便颜色与质地",
        "source": "国家卫健委《3岁以下婴幼儿健康养育照护指南（试行）》",
        "content": "婴儿便便的颜色和质地随喂养方式和月龄变化。胎便为墨绿色粘稠状，出生后2-3天内排出。母乳喂养婴儿的便便通常为金黄色或芥末黄，糊状或软膏状，有酸臭味。配方奶喂养婴儿的便便为淡黄色或黄褐色，质地较成形，有轻微臭味。添加辅食后便便颜色和质地会随食物种类变化，如吃胡萝卜后偏橙色，吃绿叶蔬菜后偏绿色。绿色便便可能是前奶摄入过多或食物消化过快，通常无需担心。出现红色、黑色（非胎便期）或白色便便应及时就医。",
        "keywords": ["便便", "大便", "屎", "颜色", "绿色", "黄色"]
    },
    # 10. 便便异常判断
    {
        "id": "kb_diaper_003",
        "title": "便便异常判断",
        "source": "国家卫健委《3岁以下婴幼儿健康养育照护指南（试行）》",
        "content": "需要警惕的异常便便包括：①血便：红色便便或便中带血丝，可能为肛裂、肠道过敏或感染；②黑色柏油样便：可能为上消化道出血；③白色或灰白色便：可能提示胆道闭锁，需紧急就医；④水样便：腹泻的表现，需注意补充水分防止脱水；⑤蛋花汤样便：病毒性肠炎的典型表现；⑥黏液脓血便：细菌性肠炎的表现。腹泻时需注意观察精神状态、尿量和口唇是否干燥等脱水表现。如出现持续腹泻超过24小时、高热、血便、精神萎靡等情况应立即就医。",
        "keywords": ["腹泻", "异常", "血便", "便秘", "拉肚子", "消化不良"]
    },
    # 11. 哭声类型与原因
    {
        "id": "kb_cry_001",
        "title": "哭声类型与原因",
        "source": "国家卫健委《3岁以下婴幼儿健康养育照护指南（试行）》",
        "content": "哭是婴儿表达需求的主要方式。常见哭声原因及特征：①饥饿哭声：有节奏的哭闹，伴有觅食反射（转头找奶头、吸吮手指）；②困倦哭声：低沉的哼唧声，揉眼睛、打哈欠；③不适哭声：尖锐持续的哭声，可能为尿布湿了、太热或太冷、衣物不适；④疼痛哭声：突然的尖声哭叫，难以安抚，可能为肠绞痛、中耳炎等；⑤过度刺激哭声：在嘈杂环境中烦躁哭闹，需要安静的环境。婴儿肠绞痛通常在2-3周开始，6-8周达高峰，3-4个月自行缓解，表现为每天固定时段（傍晚）持续哭闹超过3小时。",
        "keywords": ["哭", "哭声", "哭闹", "肠绞痛", "安抚", "烦躁"]
    },
    # 12. 哭闹安抚方法
    {
        "id": "kb_cry_002",
        "title": "哭闹安抚方法",
        "source": "国家卫健委《3岁以下婴幼儿健康养育照护指南（试行）》",
        "content": "有效的安抚方法（5S法）：①包裹（Swaddle）：用襁褓适度包裹婴儿，模拟子宫环境；②侧卧/俯卧（Stomach/Side）：在清醒看护下让婴儿侧卧或趴在大人胸前；③嘘声（Shush）：在婴儿耳边发出嘘嘘声或白噪音；④摇晃（Swing）：轻柔地左右摇晃或使用婴儿摇椅；⑤吸吮（Suck）：提供安抚奶嘴或帮助婴儿吸吮手指。其他方法：肌肤接触（袋鼠护理）、温水洗澡、按摩、播放轻柔音乐。注意：安抚时大人应保持情绪稳定，如果感到沮丧应将婴儿放在安全的地方，离开几分钟调整情绪后再回来。",
        "keywords": ["安抚", "哄", "止哭", "5S", "包裹", "白噪音"]
    },
    # 13. 体重增长标准
    {
        "id": "kb_growth_001",
        "title": "体重增长标准",
        "source": "国家卫健委《7岁以下儿童生长发育参照标准》",
        "content": "体重是反映婴幼儿营养状况最敏感的指标。新生儿出生体重平均为3.2-3.3kg（男）/3.1-3.2kg（女）。前3个月每月增长约700-1000g，4-6个月每月增长约500-600g，7-12个月每月增长约250-300g。1岁时体重约为出生时的3倍（约9-10kg），2岁时约为出生时的4倍（约12kg），3岁时约为14-15kg。体重增长低于正常范围需警惕营养不良或疾病因素，增长过快也需注意过度喂养。建议定期测量体重并绘制生长曲线图。",
        "keywords": ["体重", "称重", "偏瘦", "偏胖", "超重", "生长曲线"]
    },
    # 14. 身长发育标准
    {
        "id": "kb_growth_002",
        "title": "身长发育标准",
        "source": "国家卫健委《7岁以下儿童生长发育参照标准》",
        "content": "身长（身高）反映骨骼发育状况。新生儿出生身长平均约50cm。第一年增长最快，前3个月每月增长约3.5cm，4-6个月每月增长约2cm，7-12个月每月增长约1.2cm。1岁时身长约75cm，2岁时约87cm，3岁时约96cm。2岁后每年增长约5-7cm。测量身长应让婴儿仰卧位测量（3岁以下），3岁以上可站立测量。身长低于同龄同性别儿童第3百分位或高于第97百分位需就医评估。遗传因素对身长有重要影响，应结合父母身高综合评估。",
        "keywords": ["身高", "身长", "长高", "矮小", "发育", "骨骼"]
    },
    # 15. 头围发育标准
    {
        "id": "kb_growth_003",
        "title": "头围发育标准",
        "source": "国家卫健委《7岁以下儿童生长发育参照标准》",
        "content": "头围反映脑和颅骨的发育状况。新生儿头围平均约34cm。前3个月每月增长约2cm，4-6个月每月增长约1cm，7-12个月每月增长约0.5cm。1岁时头围约46cm，2岁时约48cm，3岁时约49-50cm。头围测量应使用软尺，经眉弓上方和枕后最突出处水平绕头一周。头围过小需警惕小头畸形或脑发育不良，头围过大需警惕脑积水。前囟门出生时约2.5×2.5cm，6个月后逐渐缩小，12-18个月闭合。前囟门早闭需警惕狭颅症，晚闭需警惕佝偻病或甲状腺功能低下。",
        "keywords": ["头围", "囟门", "前囟", "大脑", "脑发育", "头大"]
    },
    # 16. 疫苗接种计划
    {
        "id": "kb_vaccine_001",
        "title": "疫苗接种计划",
        "source": "国家卫健委《国家免疫规划疫苗儿童免疫程序》",
        "content": "国家免疫规划疫苗（免费）接种时间表：出生时：乙肝疫苗第1剂、卡介苗；1月龄：乙肝疫苗第2剂；2月龄：脊灰灭活疫苗第1剂；3月龄：脊灰减毒活疫苗第2剂、百白破第1剂；4月龄：脊灰减毒活疫苗第3剂、百白破第2剂；5月龄：百白破第3剂；6月龄：乙肝疫苗第3剂、A群流脑多糖疫苗第1剂；8月龄：麻腮风疫苗第1剂、乙脑减毒活疫苗第1剂、A群流脑多糖疫苗第2剂；9月龄：A群C群流脑多糖疫苗第1剂；18月龄：百白破第4剂、麻腮风疫苗第2剂；2岁：乙脑减毒活疫苗第2剂、甲肝疫苗；3岁：A群C群流脑多糖疫苗第2剂；6岁：白破疫苗。",
        "keywords": ["疫苗", "接种", "打针", "免疫", "预防针", "卡介苗"]
    },
    # 17. 疫苗接种注意事项
    {
        "id": "kb_vaccine_002",
        "title": "疫苗接种注意事项",
        "source": "国家卫健委《国家免疫规划疫苗儿童免疫程序》",
        "content": "疫苗接种注意事项：①接种前：确认婴儿身体健康，无发热、腹泻等急性疾病症状；正在使用免疫抑制剂或患有免疫缺陷病的婴儿不应接种减毒活疫苗；对疫苗成分过敏者禁忌接种。②接种后：在接种场所观察30分钟，确认无严重过敏反应后方可离开；接种当天不要洗澡，保持接种部位清洁干燥；可能出现低热（<38.5°C）、局部红肿、轻微烦躁等反应，一般1-2天自行缓解，可用物理降温。③禁忌情况：高热、严重慢性病、急性疾病发作期、免疫缺陷等应暂缓接种。④补种原则：未按程序接种的应及时补种，无需从头开始。",
        "keywords": ["疫苗", "接种注意", "发烧", "不良反应", "禁忌", "补种"]
    },
    # 18. 体温测量方法
    {
        "id": "kb_health_001",
        "title": "体温测量方法",
        "source": "国家卫健委《3岁以下婴幼儿健康养育照护指南（试行）》",
        "content": "准确的体温测量对判断婴儿健康状况非常重要。推荐测量方法：①腋下测温：最常用且安全的方法，将体温计水银端放在腋窝深处，夹紧上臂，测量5-10分钟。正常腋温为36.0-37.2°C。②耳温枪：快速方便，适合3个月以上婴儿，但小婴儿耳道窄可能影响准确性。③额温枪：快速无接触，但易受环境温度影响，准确性相对较低。④肛温：最准确，但不建议家长自行操作。不推荐口腔测温（婴儿不配合）和腋下电子体温计（准确性待验证）。测量前应擦干腋下汗液，运动或进食后应休息30分钟再测量。同一时间连续测量2次取高值。",
        "keywords": ["体温", "发烧", "温度", "测温", "体温计", "发热"]
    },
    # 19. 发热处理
    {
        "id": "kb_health_002",
        "title": "发热处理",
        "source": "国家卫健委《3岁以下婴幼儿健康养育照护指南（试行）》",
        "content": "婴儿发热（腋温≥37.5°C）的处理原则：①3个月以下婴儿发热（≥38°C）应立即就医，不可自行用药。②3个月以上婴儿，体温38.5°C以下且精神状态好，可先物理降温（温水擦浴、减少衣物、保持室温适宜），多喂水或奶。③体温≥38.5°C或明显不适时，可使用退热药：对乙酰氨基酚（泰诺林）适用于2个月以上，布洛芬（美林）适用于6个月以上。严格按体重计算剂量，不可同时使用两种退热药，两次用药间隔不少于4-6小时。④需立即就医的情况：3月以下发热、发热超过3天、体温超过40°C、抽搐、精神萎靡、呼吸急促、皮疹、持续呕吐、前囟隆起等。",
        "keywords": ["发烧", "退烧", "退热药", "布洛芬", "对乙酰氨基酚", "高热"]
    },
    # 20. 腹泻处理
    {
        "id": "kb_health_003",
        "title": "腹泻处理",
        "source": "国家卫健委《3岁以下婴幼儿健康养育照护指南（试行）》",
        "content": "婴儿腹泻的处理原则：①预防脱水是关键：口服补液盐III（ORS）是预防和治疗脱水的首选方法。每次腹泻后补充一定量的ORS，少量多次喂服。②继续喂养：腹泻期间不应禁食，母乳喂养婴儿继续母乳，配方奶喂养婴儿可暂时换成无乳糖配方奶粉。添加辅食的婴儿可继续吃易消化的食物（如米粥、面条）。③药物使用：益生菌（如布拉氏酵母菌、鼠李糖乳杆菌）有助于缩短病程；蒙脱石散可保护肠黏膜；锌制剂可缩短腹泻持续时间。④避免使用：抗生素（除非医生确诊细菌感染）、止泻药（如洛哌丁胺，2岁以下禁用）。⑤需就医的情况：月龄<6月、大便带血、高热不退、频繁呕吐不能口服补液、精神萎靡、尿量明显减少、口唇干燥等脱水表现。",
        "keywords": ["腹泻", "拉肚子", "脱水", "补液盐", "益生菌", "拉稀"]
    },
    # 21. 湿疹护理
    {
        "id": "kb_health_004",
        "title": "湿疹护理",
        "source": "国家卫健委《3岁以下婴幼儿健康养育照护指南（试行）》",
        "content": "婴儿湿疹（特应性皮炎）是婴幼儿常见的皮肤问题。护理要点：①保湿是基础：每日多次（至少3-4次）涂抹无香料、无刺激的保湿润肤剂，沐浴后3分钟内涂抹效果最佳。推荐使用含神经酰胺的润肤霜或凡士林。②正确沐浴：水温32-37°C，时间5-10分钟，使用温和无皂基的沐浴露，轻轻拍干而非擦拭。③避免诱因：过热、出汗、摩擦、羊毛衣物、刺激性洗涤剂、食物过敏原（如鸡蛋、牛奶、花生）等。④衣物选择：穿纯棉宽松衣物，避免羊毛和化纤直接接触皮肤。⑤药物治疗：急性期可外用弱效糖皮质激素（如氢化可的松），非激素类药物（如他克莫司软膏）适用于2岁以上。严重湿疹应就医，在医生指导下使用药物。",
        "keywords": ["湿疹", "皮肤", "痒", "过敏", "润肤", "皮炎"]
    },
    # 22. 安全防护
    {
        "id": "kb_safety_001",
        "title": "安全防护",
        "source": "国家卫健委《3岁以下婴幼儿健康养育照护指南（试行）》",
        "content": "婴幼儿安全防护要点：①防窒息：避免给3岁以下婴儿整粒坚果、葡萄、果冻、硬糖等食物；小零件玩具远离婴儿；婴儿床上不放软物。②防跌落：不要将婴儿单独留在高出地面30cm以上的地方（如床、沙发、换尿布台）；窗户安装安全锁或防护网。③防烫伤：热水温度控制在50°C以下；热饮远离婴儿可触及范围；不要抱着婴儿喝热饮。④防溺水：给婴儿洗澡时不可离开；水桶、浴缸中的水用后及时倒空。⑤防中毒：药品、清洁剂、化妆品等锁好存放；不要用食品容器装非食品物质。⑥交通安全：使用合格的汽车安全座椅，1岁以下反向安装。⑦防误吞：硬币、纽扣电池、磁铁等小物品需妥善收纳。",
        "keywords": ["安全", "防护", "窒息", "跌落", "烫伤", "安全座椅"]
    },
    # 23. 亲子互动
    {
        "id": "kb_interact_001",
        "title": "亲子互动",
        "source": "国家卫健委《3岁以下婴幼儿健康养育照护指南（试行）》",
        "content": "高质量的亲子互动对婴幼儿早期发展至关重要。各年龄段互动建议：0-3月：多与婴儿进行眼神交流，对婴儿的咿呀声做出回应，进行肌肤接触（袋鼠护理），给婴儿看高对比度的黑白图片，轻柔地说话和唱歌。4-6月：提供安全的玩具让婴儿抓握，练习俯卧抬头（tummy time），和婴儿玩躲猫猫游戏，描述日常活动。7-12月：鼓励爬行和探索，一起看绘本并指认物品，教简单的手势（如挥手再见），玩叠叠乐等游戏。1-3岁：一起进行角色扮演游戏，唱儿歌做动作，玩简单的拼图和积木，鼓励用语言表达需求。每天保证至少15-30分钟的专注亲子互动时间，放下手机，全身心陪伴。",
        "keywords": ["亲子", "互动", "玩耍", "游戏", "早教", "陪伴"]
    },
    # 24. 口腔护理
    {
        "id": "kb_oral_001",
        "title": "口腔护理",
        "source": "国家卫健委《3岁以下婴幼儿健康养育照护指南（试行）》",
        "content": "婴幼儿口腔护理应从出生开始。0-6月（出牙前）：每次喂奶后喂少量温水清洁口腔，用干净的纱布蘸温水轻轻擦拭牙龈。6-12月（出牙期）：第一颗乳牙萌出后开始使用婴儿牙刷，每天早晚各刷一次，使用米粒大小的含氟牙膏（氟浓度1000ppm以下）。出牙期间婴儿可能出现流涎增多、烦躁、低热、喜欢咬东西等症状，可提供安全的牙胶缓解不适。1-3岁：使用豌豆大小的含氟牙膏，家长帮助刷牙直到孩子能独立刷干净（约6-7岁）。避免含奶瓶入睡（预防奶瓶龋），1岁后逐渐戒除奶瓶改用杯子。定期口腔检查，建议第一颗牙萌出后6个月内或1岁前进行首次口腔检查。",
        "keywords": ["口腔", "刷牙", "出牙", "牙齿", "乳牙", "氟"]
    },
    # ==================== 疫苗接种（新增 6 条）====================
    # 25. 乙肝疫苗
    {
        "id": "kb_vaccine_003",
        "title": "乙肝疫苗",
        "source": "国家卫健委《国家免疫规划疫苗儿童免疫程序》",
        "content": "乙肝疫苗是预防乙型病毒性肝炎的有效手段。接种程序为3剂次：出生后24小时内接种第1剂（越早越好），1月龄接种第2剂，6月龄接种第3剂。如果母亲为乙肝病毒携带者或乙肝患者，新生儿出生后除接种乙肝疫苗外，还应在12小时内注射乙肝免疫球蛋白（HBIG）进行被动免疫。接种部位为上臂三角肌肌内注射。注意事项：早产儿和低体重儿同样需要接种，不应延迟；接种后少数婴儿可能出现注射部位红肿、低热，一般1-2天自行缓解；完成3剂接种后建议在12月龄时检测乙肝表面抗体，如抗体阴性需加强接种。乙肝疫苗是安全的，不会导致乙肝感染。",
        "keywords": ["乙肝", "乙肝疫苗", "肝炎", "HBIG", "免疫球蛋白", "母婴阻断"]
    },
    # 26. 卡介苗
    {
        "id": "kb_vaccine_004",
        "title": "卡介苗",
        "source": "国家卫健委《国家免疫规划疫苗儿童免疫程序》",
        "content": "卡介苗（BCG）是预防结核病的疫苗，属于减毒活疫苗。接种时间为出生后尽快接种，通常在出生时于产房内完成。接种部位为左上臂三角肌外缘皮内注射。接种后2-3周局部会出现红肿、硬结，随后可能形成小脓疱或溃疡，约2-3个月结痂愈合，留下小疤痕，这是正常的接种反应，无需特殊处理。注意事项：保持接种部位清洁干燥，不要挤压或挑破脓疱；如局部溃疡直径超过1cm或腋下淋巴结肿大超过1cm需就医；化脓可用无菌棉签擦拭，不可涂抹药膏；接种部位应避免摩擦和浸泡。早产儿体重达到2500g后方可接种。免疫缺陷病患儿禁忌接种。",
        "keywords": ["卡介苗", "BCG", "结核", "化脓", "疤痕", "接种反应"]
    },
    # 27. 脊髓灰质炎疫苗
    {
        "id": "kb_vaccine_005",
        "title": "脊髓灰质炎疫苗",
        "source": "国家卫健委《国家免疫规划疫苗儿童免疫程序》",
        "content": "脊髓灰质炎疫苗（小儿麻痹糖丸/针）用于预防脊髓灰质炎（小儿麻痹症）。我国现行接种程序采用序贯免疫方案：2月龄接种1剂脊灰灭活疫苗（IPV，注射），3月龄和4月龄各接种1剂脊灰减毒活疫苗（OPV，口服）。注意事项：口服减毒活疫苗前后30分钟内不可喂热食或热饮，以免影响疫苗活性；接种OPV后大便中排出疫苗病毒，需注意手卫生，免疫缺陷患儿及密切接触者应使用IPV；极少数接种后可能出现疫苗相关麻痹型脊髓灰质炎（VAPP），发生率极低；如孩子有免疫缺陷、正在使用免疫抑制剂或家中有人为免疫缺陷者，应告知医生选择全部IPV方案。",
        "keywords": ["脊髓灰质炎", "小儿麻痹", "糖丸", "IPV", "OPV", "口服疫苗"]
    },
    # 28. 百白破疫苗
    {
        "id": "kb_vaccine_006",
        "title": "百白破疫苗",
        "source": "国家卫健委《国家免疫规划疫苗儿童免疫程序》",
        "content": "百白破疫苗是三联疫苗，同时预防百日咳、白喉和破伤风三种疾病。接种程序：3月龄、4月龄、5月龄各接种1剂基础免疫，18月龄接种1剂加强免疫。接种部位为大腿前外侧或上臂三角肌肌内注射。常见不良反应：接种后6-12小时局部可出现红肿、疼痛和硬结，部分婴儿可能出现低热（37.5-38.5°C）、烦躁、食欲下降，一般1-2天自行缓解。少数婴儿局部硬结持续1-2个月可逐渐吸收。注意事项：接种后如体温超过38.5°C或持续发热超过48小时需就医；局部红肿直径超过5cm需就医评估；有惊厥史或神经系统疾病的婴儿需在医生指导下接种；接种后注意休息，避免剧烈活动。",
        "keywords": ["百白破", "百日咳", "白喉", "破伤风", "不良反应", "红肿"]
    },
    # 29. 麻疹疫苗
    {
        "id": "kb_vaccine_007",
        "title": "麻疹疫苗",
        "source": "国家卫健委《国家免疫规划疫苗儿童免疫程序》",
        "content": "我国现行使用麻腮风联合疫苗（MMR），同时预防麻疹、流行性腮腺炎和风疹。接种程序：8月龄接种第1剂，18月龄接种第2剂。接种部位为上臂外侧皮下注射。接种禁忌：对疫苗成分（如明胶、新霉素）过敏者禁忌接种；免疫缺陷病患儿、正在使用免疫抑制剂者禁忌接种减毒活疫苗；妊娠期妇女禁忌接种；高热或急性疾病期应暂缓接种。注意事项：接种后7-12天可能出现低热（约5%-15%）、皮疹（约2%-5%），一般2-3天自行消退，无需特殊处理；少数婴儿接种后5-12天可能出现一过性腮腺肿大；接种后1个月内避免接触孕妇和免疫缺陷者。",
        "keywords": ["麻疹", "麻腮风", "MMR", "风疹", "腮腺炎", "出疹"]
    },
    # 30. 流感疫苗
    {
        "id": "kb_vaccine_008",
        "title": "流感疫苗",
        "source": "中国疾病预防控制中心《流感疫苗预防接种技术指南》",
        "content": "流感疫苗是预防流行性感冒及其严重并发症的重要手段。建议接种年龄：6个月以上婴幼儿及儿童均可接种。接种程序：6月龄-8岁首次接种者需接种2剂（间隔4周以上），之后每年接种1剂；9岁以上每年接种1剂。最佳接种时间为每年9-11月（流感高发季前）。注意事项：6月龄-3岁使用儿童剂型（0.25ml），3岁以上使用成人剂型（0.5ml）；对鸡蛋蛋白严重过敏者需在医生指导下接种；接种后可能出现注射部位疼痛、低热，一般1-2天缓解；流感病毒变异快，每年疫苗成分会更新，因此需要每年接种；不建议使用减毒活疫苗鼻喷剂（LAIV）于2岁以下儿童。",
        "keywords": ["流感", "流感疫苗", "感冒", "鼻喷疫苗", "季节性", "鸡蛋过敏"]
    },
    # ==================== 常见疾病（新增 8 条）====================
    # 31. 幼儿急疹
    {
        "id": "kb_illness_001",
        "title": "幼儿急疹",
        "source": "国家卫健委《3岁以下婴幼儿健康养育照护指南（试行）》",
        "content": "幼儿急疹（玫瑰疹）是由人类疱疹病毒6型（HHV-6）引起的常见婴幼儿急性传染病。典型表现：高热3-5天（体温可达39-40°C），婴儿精神状态相对较好，热退后躯干开始出现红色斑丘疹，逐渐蔓延至面部和四肢，皮疹1-3天消退，不留痕迹。好发年龄为6月龄-2岁。护理要点：发热期以物理降温为主，体温超过38.5°C可使用退热药（对乙酰氨基酚或布洛芬）；多喂水或奶，注意补充水分；饮食以清淡易消化为主；皮疹期无需特殊处理，避免使用药膏或护肤品刺激皮肤；一般无需抗病毒治疗，可自愈。需就医的情况：热退后精神萎靡、持续高热超过5天、出现抽搐、皮疹伴有出血点或按压不褪色。",
        "keywords": ["幼儿急疹", "玫瑰疹", "高热", "出疹", "HHV-6", "热退疹出"]
    },
    # 32. 手足口病
    {
        "id": "kb_illness_002",
        "title": "手足口病",
        "source": "国家卫健委《手足口病诊疗指南》",
        "content": "手足口病是由肠道病毒（主要为柯萨奇病毒A16型和EV71型）引起的常见传染病。症状：口腔黏膜出现疱疹或溃疡，手掌、足底和臀部出现红色斑丘疹或疱疹，可伴有低至中度发热、食欲下降、流涎。好发年龄为5岁以下，多见于夏秋季。预防措施：勤洗手，尤其是饭前便后；避免与患病儿童密切接触；不共用毛巾、餐具等物品；玩具和常接触的表面定期消毒。护理要点：注意口腔卫生，饭后用温水漱口；进食软凉食物，避免酸、辣、咸等刺激性食物；多饮水；发热时适当使用退热药。需立即就医的情况：持续高热不退、频繁呕吐、精神萎靡或烦躁不安、肢体抖动、呼吸心率增快、面色苍白，提示可能发展为重症。",
        "keywords": ["手足口病", "口腔疱疹", "手足疱疹", "EV71", "柯萨奇", "肠道病毒"]
    },
    # 33. 水痘
    {
        "id": "kb_illness_003",
        "title": "水痘",
        "source": "国家卫健委《3岁以下婴幼儿健康养育照护指南（试行）》",
        "content": "水痘是由水痘-带状疱疹病毒（VZV）引起的高度传染性疾病。症状：发热1-2天后出现皮疹，皮疹按斑疹、丘疹、疱疹、结痂的顺序演变，同一时期可见各阶段皮疹（\"四世同堂\"），皮疹先出现在躯干和头部，后蔓延至面部和四肢，伴有明显瘙痒。护理要点：剪短婴儿指甲，防止抓破疱疹继发感染；可使用炉甘石洗剂涂抹止痒；发热时使用退热药（避免使用阿司匹林，以防瑞氏综合征）；保持皮肤清洁，勤换衣物；疱疹破溃处可涂抹抗生素软膏预防感染。隔离至全部疱疹结痂（约7-10天）。注意：水痘疫苗为自费疫苗，建议12-18月龄接种第1剂，4-6岁接种第2剂。如孕妇、新生儿接触水痘患者需紧急就医评估。",
        "keywords": ["水痘", "疱疹", "瘙痒", "炉甘石", "VZV", "传染"]
    },
    # 34. 轮状病毒腹泻
    {
        "id": "kb_illness_004",
        "title": "轮状病毒腹泻",
        "source": "世界卫生组织《轮状病毒疫苗立场文件》",
        "content": "轮状病毒是婴幼儿秋冬季腹泻最常见的病原体。症状：突然发病，呕吐为先兆症状，随后出现大量水样或蛋花汤样便，每日可达10余次，可伴有低至中度发热，病程3-7天。护理要点：预防和纠正脱水是关键，每次腹泻后补充口服补液盐III（ORS）；继续母乳或配方奶喂养，无需禁食；可使用益生菌（如布拉氏酵母菌）辅助治疗；注意臀部护理，预防尿布疹。预防措施：接种轮状病毒疫苗（口服减毒活疫苗），目前国内有Lanzhou lamb和LLR两种，分别在2月龄、4月龄（或6月龄）口服；注意饮食卫生，奶瓶餐具消毒；看护人勤洗手。需就医的情况：频繁呕吐无法口服补液、尿量明显减少、精神萎靡、便中带血、高热不退。",
        "keywords": ["轮状病毒", "秋季腹泻", "水样便", "蛋花汤", "ORS", "脱水"]
    },
    # 35. 中耳炎
    {
        "id": "kb_illness_005",
        "title": "中耳炎",
        "source": "美国儿科学会《中耳炎诊断与治疗指南》",
        "content": "中耳炎是婴幼儿常见的感染性疾病，多继发于上呼吸道感染。识别症状：婴儿突然频繁抓扯耳朵、摇头；不明原因的持续哭闹，尤其是平躺时加重；发热（部分可无发热）；耳朵流出黄色或白色液体（鼓膜穿孔时）；听力下降的表现（对声音反应迟钝）。危险因素：奶瓶平躺喂养、长期二手烟暴露、上呼吸道感染后、过敏体质、日托机构儿童。护理要点：及时就医，医生会根据情况决定是否使用抗生素（急性中耳炎通常需要抗生素治疗）；按医嘱完成全程用药，不要自行停药；发热时使用退热药缓解不适；喂奶时抬高头部，避免平躺喂养。预防：母乳喂养可降低风险；避免二手烟；按时接种流感疫苗和肺炎球菌疫苗；上呼吸道感染后注意观察耳部症状。",
        "keywords": ["中耳炎", "耳朵", "抓耳朵", "耳痛", "流脓", "听力"]
    },
    # 36. 鹅口疮
    {
        "id": "kb_illness_006",
        "title": "鹅口疮",
        "source": "国家卫健委《3岁以下婴幼儿健康养育照护指南（试行）》",
        "content": "鹅口疮是由白色念珠菌（真菌）感染引起的口腔黏膜疾病，在婴幼儿中较为常见。症状：口腔黏膜（颊黏膜、舌面、上颚）出现白色或乳黄色的斑膜，形似奶块但不易擦去，强行擦去后可见红色糜烂面。婴儿可能无明显不适，也可能因疼痛影响进食。危险因素：免疫力低下、长期使用抗生素、奶嘴和奶瓶消毒不彻底、母亲阴道有真菌感染。治疗：使用抗真菌药物（如制霉菌素混悬液）涂抹患处，每日3-4次，通常连续使用7-14天；症状消失后仍需继续用药2-3天以防复发。护理：喂奶后用温水清洁口腔；奶嘴、奶瓶、安抚奶嘴每日煮沸消毒15分钟；母亲哺乳前后清洁乳头；避免过度使用抗生素。需就医的情况：治疗3天后无改善、影响进食、出现发热或腹泻。",
        "keywords": ["鹅口疮", "白色念珠菌", "口腔白斑", "真菌", "制霉菌素", "奶瓶消毒"]
    },
    # 37. 湿疹
    {
        "id": "kb_illness_007",
        "title": "湿疹",
        "source": "国家卫健委《3岁以下婴幼儿健康养育照护指南（试行）》",
        "content": "婴儿湿疹（特应性皮炎）是婴幼儿最常见的慢性皮肤问题，与遗传过敏体质密切相关。常见触发因素（triggers）：食物过敏原（牛奶蛋白、鸡蛋、花生、大豆、小麦等）、环境因素（尘螨、花粉、宠物皮屑、霉菌）、物理刺激（过热、出汗、干燥空气、羊毛衣物）、化学刺激（肥皂、沐浴露、洗衣液、柔顺剂）、情绪因素（哭闹、压力）。护理要点：保湿是核心，每日涂抹保湿剂至少3-4次，沐浴后3分钟内涂抹效果最佳；选择无香料、无色素的润肤产品（如凡士林、含神经酰胺的润肤霜）；避免已知触发因素；急性发作期在医生指导下使用外用药物（弱效激素或钙调磷酸酶抑制剂）；剪短指甲防止抓挠，睡眠时戴纯棉手套。约60%的婴儿湿疹在2岁前缓解，但部分可能持续至儿童期。",
        "keywords": ["湿疹", "特应性皮炎", "过敏", "保湿", "triggers", "瘙痒"]
    },
    # 38. 过敏性鼻炎
    {
        "id": "kb_illness_008",
        "title": "过敏性鼻炎",
        "source": "中国儿童过敏性鼻炎诊疗指南",
        "content": "过敏性鼻炎是儿童常见的过敏性疾病，近年发病率呈上升趋势。症状：反复发作的鼻塞、流清水样鼻涕、打喷嚏（阵发性连打数个）、鼻痒（婴儿表现为揉鼻子、皱眉）；可伴有眼痒、流泪（过敏性结膜炎）；长期可导致张口呼吸、打鼾、睡眠质量下降。常见过敏原：尘螨（最常见的室内过敏原）、花粉（春秋季节性发作）、宠物皮屑、霉菌。管理措施：避免接触过敏原（使用防螨床品、定期清洗床上用品、控制室内湿度在50%以下、花粉季减少户外活动或佩戴口罩）；鼻腔盐水冲洗（每日1-2次，有助于清除过敏原和分泌物）；在医生指导下使用抗组胺药（如西替利嗪）或鼻用糖皮质激素（如糠酸莫米松鼻喷剂）；免疫治疗（脱敏治疗）适用于5岁以上、症状持续、药物控制不佳的患儿。",
        "keywords": ["过敏性鼻炎", "鼻塞", "打喷嚏", "流鼻涕", "尘螨", "抗组胺"]
    },
    # ==================== 营养补充（新增 4 条）====================
    # 39. 维生素D补充
    {
        "id": "kb_nutrition_001",
        "title": "维生素D补充",
        "source": "国家卫健委《3岁以下婴幼儿健康养育照护指南（试行）》",
        "content": "维生素D对婴幼儿钙磷代谢和骨骼发育至关重要。补充建议：足月儿出生后数天内开始每日补充400IU维生素D；早产儿、低出生体重儿、双胎儿出生后即开始每日补充800IU，3个月后改为400IU。补充时间：应持续补充至2岁甚至更久，尤其在日照不足的季节和地区。注意：母乳中维生素D含量极低，纯母乳喂养婴儿必须额外补充；配方奶喂养婴儿需计算配方奶中维生素D含量，不足400IU的部分需额外补充。安全性：每日400IU是安全剂量，一般不会导致维生素D中毒；不要自行增加大剂量。来源：适量户外活动（阳光照射皮肤合成维生素D），但6个月以下婴儿应避免阳光直射，需通过补充剂获取。",
        "keywords": ["维生素D", "VD", "补钙", "佝偻病", "晒太阳", "400IU"]
    },
    # 40. 铁剂补充
    {
        "id": "kb_nutrition_002",
        "title": "铁剂补充",
        "source": "国家卫健委《3岁以下婴幼儿健康养育照护指南（试行）》",
        "content": "铁是婴幼儿生长发育和脑发育的重要微量元素。何时需要补充：早产儿和低出生体重儿：出生后2-4周开始补充元素铁2mg/(kg·d)，持续至1岁；足月儿：4-6个月开始，母乳中铁含量低，应及时添加富含铁的辅食（如强化铁米粉、肉泥、肝泥）；1-3岁：每日铁需求量为9mg，通过饮食摄入（红肉、动物肝脏、蛋黄、深绿色蔬菜）。铁缺乏的表现：面色苍白、食欲下降、易疲劳、注意力不集中、异食癖（吃土、纸等）。如确诊缺铁性贫血，需在医生指导下补充铁剂，常用为硫酸亚铁或葡萄糖酸亚铁，剂量为元素铁2-6mg/(kg·d)，分2-3次口服，两餐之间服用吸收更好，与维生素C同服可促进吸收。铁剂补充后网织红细胞1周升高，血红蛋白2-4周改善，需持续补充2-3个月恢复铁储备。",
        "keywords": ["铁", "铁剂", "贫血", "缺铁", "血红蛋白", "补铁"]
    },
    # 41. 钙补充
    {
        "id": "kb_nutrition_003",
        "title": "钙补充",
        "source": "国家卫健委《3岁以下婴幼儿健康养育照护指南（试行）》",
        "content": "钙是骨骼和牙齿发育的重要元素。各年龄段每日钙推荐摄入量：0-6月龄200mg，7-12月龄260mg，1-3岁600mg。母乳喂养婴儿每日奶量充足（约600-800ml）即可满足钙需求，无需额外补钙。配方奶喂养婴儿，配方奶中已添加足量钙。饮食来源：奶及奶制品（母乳、配方奶、酸奶、奶酪）是最佳钙源；豆制品（豆腐、豆干）；深绿色蔬菜（菠菜、西兰花，注意草酸含量高的需焯水）；小鱼干、虾皮。补充剂：一般不需要额外补充钙剂，除非经医生评估确认钙摄入不足或存在钙缺乏。注意：钙的吸收需要维生素D的参与，保证维生素D充足比单纯补钙更重要；不要盲目补钙，过量补钙可能导致便秘、影响铁和锌的吸收；骨头汤补钙效果极差，不推荐。",
        "keywords": ["钙", "补钙", "骨头汤", "奶制品", "维生素D", "豆腐"]
    },
    # 42. DHA/ARA
    {
        "id": "kb_nutrition_004",
        "title": "DHA/ARA补充",
        "source": "中国营养学会《中国居民膳食营养素参考摄入量》",
        "content": "DHA（二十二碳六烯酸）和ARA（花生四烯酸）是对婴幼儿大脑和视觉发育重要的长链多不饱和脂肪酸。来源：母乳中天然含有DHA和ARA，是婴儿获取的最佳途径；配方奶中通常会添加DHA和ARA；食物来源包括深海鱼（三文鱼、沙丁鱼、鳕鱼）、海藻、蛋黄、核桃油等。补充建议：母乳喂养婴儿，母亲应保证每周摄入2-3次富含DHA的鱼类（注意选择低汞鱼类），或适当补充DHA补充剂（每日200-300mg）；配方奶喂养婴儿，如配方奶中已添加足量DHA，一般无需额外补充；6个月以上添加辅食后，可适量引入富含DHA的食物。注意事项：DHA补充并非越多越好，适量即可；选择正规品牌的产品，注意DHA的纯度和来源；ARA通常可通过体内自行合成，一般不需要额外补充。",
        "keywords": ["DHA", "ARA", "脑发育", "视力", "深海鱼", "核桃"]
    },
    # ==================== 安全急救（新增 4 条）====================
    # 43. 窒息急救
    {
        "id": "kb_safety_002",
        "title": "窒息急救",
        "source": "美国心脏协会《婴幼儿心肺复苏指南》",
        "content": "婴幼儿窒息是危及生命的紧急情况，必须立即处理。海姆立克急救法（婴儿版）：适用于1岁以下婴儿。步骤：①将婴儿面朝下放在前臂上，用手托住婴儿下颌（注意不要捏住脖子），使婴儿头部低于胸部；②用手掌根部在婴儿两肩胛骨之间拍击5次；③如异物未排出，将婴儿翻转面朝上，用两指在胸骨下半部（乳头连线下一横指处）按压5次；④交替进行拍背和按压，直到异物排出或婴儿开始哭泣。注意事项：不要盲目用手指在婴儿口中掏取异物，以免将异物推入更深；如果婴儿意识丧失，立即开始心肺复苏（CPR）；即使异物已排出，也应就医检查是否有内脏损伤。预防：避免给3岁以下婴儿整粒坚果、葡萄（需切四瓣）、果冻、硬糖、爆米花等食物；小零件玩具远离婴儿；进食时保持安静，不要边跑边吃。",
        "keywords": ["窒息", "海姆立克", "急救", "异物", "拍背", "心肺复苏"]
    },
    # 44. 烫伤处理
    {
        "id": "kb_safety_003",
        "title": "烫伤处理",
        "source": "中华医学会烧伤外科学分会《烧伤早期救治指南》",
        "content": "婴幼儿烫伤是常见的家庭意外伤害。急救五字诀：冲、脱、泡、盖、送。①冲：立即用流动冷水冲洗烫伤部位15-20分钟，降低皮肤温度，减轻疼痛和组织损伤。注意水流不要太大，避免冲破水泡。②脱：在冷水中小心脱去或剪开烫伤部位的衣物，如衣物与皮肤粘连，切勿强行撕扯，应保留粘连部分的衣物。③泡：如果疼痛明显，可将烫伤部位继续浸泡在冷水中10-30分钟。注意水温不能过低（不要用冰块直接接触），以免冻伤。④盖：用清洁的湿纱布或保鲜膜轻轻覆盖创面，保护伤口。不要涂抹牙膏、酱油、面粉等民间偏方，不要使用有色药水。⑤送：面积大于婴儿手掌面积的烫伤、面部、手部、会阴部烫伤或出现水泡的烫伤应立即送医。注意事项：不要挑破水泡；一度烫伤（皮肤发红、疼痛）可在家护理；二度以上烫伤需就医。",
        "keywords": ["烫伤", "烧伤", "冲脱泡盖送", "水泡", "冷水", "急救"]
    },
    # 45. 跌落处理
    {
        "id": "kb_safety_004",
        "title": "跌落处理",
        "source": "国家卫健委《3岁以下婴幼儿健康养育照护指南（试行）》",
        "content": "婴幼儿跌落是常见的意外伤害，多发生在从床、沙发、换尿布台、学步车等高处跌落。处理步骤：①保持冷静，先观察婴儿的反应，不要立即抱起（如有脊柱损伤可能，移动会加重损伤）。②观察婴儿是否哭泣：如果跌落后立即大哭，通常意识清醒，可抱起安抚并检查受伤部位。③检查是否有明显外伤：头部有无血肿、伤口、出血；四肢有无肿胀、畸形、活动受限。④观察有无严重表现：意识丧失或嗜睡、持续呕吐、抽搐、耳鼻流出血液或清亮液体、瞳孔不等大。何时就医：跌落高度超过1米；头部着地且出现上述严重表现；出现意识改变、持续呕吐、异常哭闹或异常安静；肢体活动受限或畸形；伤口较深或出血不止。48小时内需密切观察：即使初期无异常，也需观察48小时，注意是否出现迟发性颅内出血的表现（嗜睡、呕吐、拒食、烦躁）。",
        "keywords": ["跌落", "摔伤", "头部着地", "脑震荡", "血肿", "坠落"]
    },
    # 46. 误食处理
    {
        "id": "kb_safety_005",
        "title": "误食处理",
        "source": "国家卫健委《3岁以下婴幼儿健康养育照护指南（试行）》",
        "content": "婴幼儿误食是常见的家庭急症，常见误食物品包括药物、清洁剂、化妆品、纽扣电池、小玩具零件、植物等。处理原则：①立即清除口腔残留物：用手指或纱布清除口腔内可见的残留物。②判断误食物品及量：保留误食物品的包装或残余，以便告知医生。③不要自行催吐：腐蚀性物质（强酸、强碱、漂白剂）催吐会造成二次灼伤；意识不清的婴儿催吐可能导致误吸。④立即就医的情况：腐蚀性物质、纽扣电池（可在食道内数小时内造成严重灼伤）、药物过量、尖锐物品、含汞物品、不明物质；出现呼吸困难、口唇发紫、流涎、呕吐、腹痛、意识改变。⑤可先观察的情况：误食少量无毒物质（如少量粉笔、蜡笔），婴儿精神好、无不适。就医时携带误食物品包装，告知误食时间、估计量及婴儿当前状态。预防：药品锁好存放、使用安全锁扣；使用儿童安全包装的产品；不要用食品容器装非食品物质。",
        "keywords": ["误食", "误吞", "中毒", "催吐", "纽扣电池", "清洁剂"]
    },
]

KB_PKL_PATH = KNOWLEDGE_DATA_DIR / "kb_bm25_index.pkl"


class KnowledgeBaseService:
    def __init__(self):
        self.entries = list(KNOWLEDGE_ENTRIES)
        self.bm25_index = None
        self._entry_embeddings: Dict[str, List[float]] = {}  # Cache for vector embeddings
        self._embeddings_lock = Lock()  # Thread-safe lock for embeddings
        self._entries_lock = Lock()  # Thread-safe lock for entries modification
        self._load_or_build_index()
        self._precompute_embeddings()  # Pre-compute embeddings for all entries

    def _load_or_build_index(self):
        if KB_PKL_PATH.exists():
            try:
                with open(KB_PKL_PATH, "rb") as f:
                    cached = pickle.load(f)
                if cached.get("version") == 1 and cached.get("count") == len(self.entries):
                    self.bm25_index = cached["index"]
                    logger.info(f"BM25 索引已从缓存加载 ({len(self.entries)} 条)")
                    return
            except Exception as e:
                logger.warning(f"加载 BM25 缓存失败: {e}")
        self._build_index()

    def _build_index(self):
        try:
            from rank_bm25 import BM25Okapi
            import jieba
            tokenized = []
            for entry in self.entries:
                text = f"{entry['title']} {entry['content']}"
                tokens = list(jieba.cut(text))
                tokenized.append(tokens)
            self.bm25_index = BM25Okapi(tokenized)
            KNOWLEDGE_DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(KB_PKL_PATH, "wb") as f:
                pickle.dump({"version": 1, "count": len(self.entries), "index": self.bm25_index}, f)
            logger.info(f"BM25 索引已构建并缓存 ({len(self.entries)} 条)")
        except ImportError:
            logger.warning("rank_bm25 或 jieba 未安装，BM25 搜索不可用")
            self.bm25_index = None

    def _rebuild_index(self):
        """Rebuild BM25 index after entries change and invalidate cache."""
        if KB_PKL_PATH.exists():
            try:
                KB_PKL_PATH.unlink()
            except Exception:
                pass
        self._build_index()
        # Also rebuild embeddings cache
        self._precompute_embeddings()

    def _precompute_embeddings(self):
        """Pre-compute embeddings for all entries to avoid real-time computation."""
        try:
            from vector_db import vector_db_service

            with self._embeddings_lock:
                self._entry_embeddings = {}
                for entry in self.entries:
                    text = f"{entry['title']} {entry['content']}"
                    embedding = vector_db_service.generate_embedding(text)
                    if embedding and not all(v == 0.0 for v in embedding):
                        self._entry_embeddings[entry['id']] = embedding

            logger.info(f"Pre-computed embeddings for {len(self._entry_embeddings)} entries")
        except Exception as e:
            logger.warning(f"Failed to pre-compute embeddings: {e}")
            self._entry_embeddings = {}

    def search(self, query: str, n_results: int = 3) -> Dict:
        results = []
        keyword_matched = set()

        # Keyword matching
        query_lower = query.lower()
        for i, entry in enumerate(self.entries):
            for kw in entry.get("keywords", []):
                if kw.lower() in query_lower or query_lower in kw.lower():
                    keyword_matched.add(i)
                    break

        for idx in keyword_matched:
            entry = self.entries[idx]
            results.append({
                "id": entry["id"],
                "title": entry["title"],
                "source": entry["source"],
                "content": entry["content"][:300] + "...",
                "keywords": entry["keywords"],
                "score": 1.0,
                "match_type": "keyword",
            })

        # BM25 matching
        if self.bm25_index:
            try:
                import jieba
                tokens = list(jieba.cut(query))
                scores = self.bm25_index.get_scores(tokens)
                scored = [(i, s) for i, s in enumerate(scores) if s > 0 and i not in keyword_matched]
                scored.sort(key=lambda x: x[1], reverse=True)
                for i, score in scored[:n_results]:
                    entry = self.entries[i]
                    results.append({
                        "id": entry["id"],
                        "title": entry["title"],
                        "source": entry["source"],
                        "content": entry["content"][:300] + "...",
                        "keywords": entry["keywords"],
                        "score": float(score),
                        "match_type": "bm25",
                    })
            except Exception as e:
                logger.error(f"BM25 搜索失败: {e}")

        results.sort(key=lambda x: x["score"], reverse=True)
        return {
            "success": True,
            "query": query,
            "results": results[:n_results],
            "total": len(results),
        }

    def vector_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Semantic search using pre-computed embeddings.

        Uses cached embeddings for fast similarity computation.
        Falls back to BM25 search if vector database is not available.
        """
        try:
            from vector_db import vector_db_service

            # Generate query embedding (only once per search)
            query_embedding = vector_db_service.generate_embedding(query)
            if not query_embedding or all(v == 0.0 for v in query_embedding):
                logger.warning("Vector embedding is zero, falling back to BM25")
                return self._bm25_fallback(query, top_k)

            if not HAS_NUMPY:
                logger.warning("numpy not available, falling back to BM25")
                return self._bm25_fallback(query, top_k)

            # Use pre-computed embeddings for fast similarity computation
            with self._embeddings_lock:
                cached_embeddings = dict(self._entry_embeddings)

            if not cached_embeddings:
                logger.warning("No cached embeddings available, falling back to BM25")
                return self._bm25_fallback(query, top_k)

            # Compute cosine similarity using numpy
            vec_a = np.array(query_embedding)
            norm_a = np.linalg.norm(vec_a)

            results = []
            for entry in self.entries:
                entry_id = entry["id"]
                if entry_id not in cached_embeddings:
                    continue

                entry_embedding = cached_embeddings[entry_id]
                vec_b = np.array(entry_embedding)
                norm_b = np.linalg.norm(vec_b)

                if norm_a > 0 and norm_b > 0:
                    similarity = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
                else:
                    similarity = 0.0

                results.append({
                    "id": entry["id"],
                    "title": entry["title"],
                    "source": entry["source"],
                    "content": entry["content"][:300] + "...",
                    "keywords": entry.get("keywords", []),
                    "score": similarity,
                    "match_type": "vector",
                })

            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.warning(f"Vector search failed, falling back to BM25: {e}")
            return self._bm25_fallback(query, top_k)

    def _bm25_fallback(self, query: str, top_k: int) -> List[Dict]:
        """Fallback to BM25 search when vector search is unavailable."""
        search_result = self.search(query, n_results=top_k)
        return search_result.get("results", [])

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Hybrid search: combine BM25 + vector search results with dedup.

        Merges results from both retrieval methods, normalizes scores,
        and returns deduplicated entries ranked by combined score.
        """
        # Collect BM25 results (including keyword matches)
        bm25_results = []
        try:
            search_result = self.search(query, n_results=top_k * 2)
            bm25_results = search_result.get("results", [])
        except Exception as e:
            logger.error(f"BM25 search failed in hybrid mode: {e}")

        # Collect vector results
        vector_results = []
        try:
            vector_results = self.vector_search(query, top_k=top_k * 2)
        except Exception as e:
            logger.error(f"Vector search failed in hybrid mode: {e}")

        # Merge and deduplicate by entry id
        merged = {}
        for r in bm25_results:
            eid = r["id"]
            if eid not in merged:
                merged[eid] = r
                merged[eid]["bm25_score"] = r.get("score", 0)
                merged[eid]["vector_score"] = 0
            else:
                merged[eid]["bm25_score"] = max(merged[eid].get("bm25_score", 0), r.get("score", 0))

        for r in vector_results:
            eid = r["id"]
            if eid not in merged:
                merged[eid] = r
                merged[eid]["bm25_score"] = 0
                merged[eid]["vector_score"] = r.get("score", 0)
            else:
                merged[eid]["vector_score"] = max(merged[eid].get("vector_score", 0), r.get("score", 0))

        # Compute combined score (weighted average: 0.4 BM25 + 0.6 vector)
        # Normalize scores using max score in current batch (min-max normalization)
        max_bm25 = max([merged[eid].get("bm25_score", 0) for eid in merged] or [1.0])
        max_vector = max([merged[eid].get("vector_score", 0) for eid in merged] or [1.0])

        for eid in merged:
            bm25 = merged[eid].get("bm25_score", 0)
            vec = merged[eid].get("vector_score", 0)
            # Min-max normalization to 0-1 range
            bm25_norm = float(bm25) / max_bm25 if max_bm25 > 0 else 0
            vec_norm = float(vec) / max_vector if max_vector > 0 else 0
            combined = 0.4 * bm25_norm + 0.6 * vec_norm
            merged[eid]["score"] = combined
            merged[eid]["match_type"] = "hybrid"

        # Sort by combined score and return top_k
        results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    # ==================== Dynamic management methods ====================

    def _get_category_from_id(self, entry_id: str) -> str:
        """Infer category from entry ID prefix."""
        prefix_map = {
            "kb_feed": "feeding",
            "kb_sleep": "sleep",
            "kb_diaper": "diaper",
            "kb_cry": "cry",
            "kb_health": "health",
            "kb_vaccine": "vaccine",
            "kb_illness": "illness",
            "kb_nutrition": "nutrition",
            "kb_safety": "safety",
        }
        for prefix, cat in prefix_map.items():
            if entry_id.startswith(prefix):
                return cat
        return "other"

    def list_entries(self, category: Optional[str] = None) -> List[Dict]:
        """List all knowledge entries, optionally filtered by category."""
        entries = self.entries
        if category:
            entries = [e for e in entries if self._get_category_from_id(e["id"]) == category]
        return [
            {
                "id": e["id"],
                "title": e["title"],
                "source": e.get("source", ""),
                "keywords": e.get("keywords", []),
                "category": self._get_category_from_id(e["id"]),
            }
            for e in entries
        ]

    def get_entry(self, entry_id: str) -> Optional[Dict]:
        """Get a single knowledge entry by id."""
        for entry in self.entries:
            if entry["id"] == entry_id:
                return dict(entry)
        return None

    def add_entry(self, entry: Dict) -> Dict:
        """Add a new knowledge entry dynamically."""
        # Validate required fields
        if not entry.get("id") or not entry.get("title") or not entry.get("content"):
            raise ValueError("id, title, and content are required")
        # Check for duplicate id
        if any(e["id"] == entry["id"] for e in self.entries):
            raise ValueError(f"Entry with id '{entry['id']}' already exists")
        # Set defaults
        entry.setdefault("source", "")
        entry.setdefault("keywords", [])
        entry.setdefault("category", "")
        self.entries.append(entry)
        self._rebuild_index()
        logger.info(f"Added knowledge entry: {entry['id']} - {entry['title']}")
        return entry

    def delete_entry(self, entry_id: str) -> bool:
        """Delete a knowledge entry by id."""
        original_len = len(self.entries)
        self.entries = [e for e in self.entries if e["id"] != entry_id]
        if len(self.entries) < original_len:
            self._rebuild_index()
            logger.info(f"Deleted knowledge entry: {entry_id}")
            return True
        return False

    def get_status(self) -> Dict:
        return {
            "success": True,
            "ready": self.bm25_index is not None,
            "total_entries": len(self.entries),
            "source": "国家卫健委婴幼儿照护指南（3部）",
            "retrieval_method": "BM25 + 关键词混合检索 + 向量语义检索",
        }


knowledge_service = KnowledgeBaseService()
