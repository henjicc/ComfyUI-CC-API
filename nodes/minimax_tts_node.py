import os
import json
import base64
import tempfile
import requests
import numpy as np
import torch
from scipy.io import wavfile
import server
from aiohttp import web
from .cc_utils import CCConfig

class MiniMaxTTS:
    """MiniMax TTS节点"""
    
    # 定义音色名称到音色ID的映射表
    VOICE_NAME_TO_ID = {
        # 中文 (普通话) 音色
        "中文 - 青涩青年音色": "male-qn-qingse",
        "中文 - 精英青年音色": "male-qn-jingying",
        "中文 - 霸道青年音色": "male-qn-badao",
        "中文 - 青年大学生音色": "male-qn-daxuesheng",
        "中文 - 少女音色": "female-shaonv",
        "中文 - 御姐音色": "female-yujie",
        "中文 - 成熟女性音色": "female-chengshu",
        "中文 - 甜美女性音色": "female-tianmei",
        "中文 - 男性主持人": "presenter_male",
        "中文 - 女性主持人": "presenter_female",
        "中文 - 男性有声书1": "audiobook_male_1",
        "中文 - 男性有声书2": "audiobook_male_2",
        "中文 - 女性有声书1": "audiobook_female_1",
        "中文 - 女性有声书2": "audiobook_female_2",
        "中文 - 青涩青年音色-beta": "male-qn-qingse-jingpin",
        "中文 - 精英青年音色-beta": "male-qn-jingying-jingpin",
        "中文 - 霸道青年音色-beta": "male-qn-badao-jingpin",
        "中文 - 青年大学生音色-beta": "male-qn-daxuesheng-jingpin",
        "中文 - 少女音色-beta": "female-shaonv-jingpin",
        "中文 - 御姐音色-beta": "female-yujie-jingpin",
        "中文 - 成熟女性音色-beta": "female-chengshu-jingpin",
        "中文 - 甜美女性音色-beta": "female-tianmei-jingpin",
        "中文 - 聪明男童": "clever_boy",
        "中文 - 可爱男童": "cute_boy",
        "中文 - 萌萌女童": "lovely_girl",
        "中文 - 卡通猪小琪": "cartoon_pig",
        "中文 - 病娇弟弟": "bingjiao_didi",
        "中文 - 俊朗男友": "junlang_nanyou",
        "中文 - 纯真学弟": "chunzhen_xuedi",
        "中文 - 冷淡学长": "lengdan_xiongzhang",
        "中文 - 霸道少爷": "badao_shaoye",
        "中文 - 甜心小玲": "tianxin_xiaoling",
        "中文 - 俏皮萌妹": "qiaopi_mengmei",
        "中文 - 妩媚御姐": "wumei_yujie",
        "中文 - 嗲嗲学妹": "diadia_xuemei",
        "中文 - 淡雅学姐": "danya_xuejie",
        "中文 - 沉稳高管": "Chinese (Mandarin)_Reliable_Executive",
        "中文 - 新闻女声": "Chinese (Mandarin)_News_Anchor",
        "中文 - 傲娇御姐": "Chinese (Mandarin)_Mature_Woman",
        "中文 - 不羁青年": "Chinese (Mandarin)_Unrestrained_Young_Man",
        "中文 - 嚣张小姐": "Arrogant_Miss",
        "中文 - 机械战甲": "Robot_Armor",
        "中文 - 热心大婶": "Chinese (Mandarin)_Kind-hearted_Antie",
        "中文 - 港普空姐": "Chinese (Mandarin)_HK_Flight_Attendant",
        "中文 - 搞笑大爷": "Chinese (Mandarin)_Humorous_Elder",
        "中文 - 温润男声": "Chinese (Mandarin)_Gentleman",
        "中文 - 温暖闺蜜": "Chinese (Mandarin)_Warm_Bestie",
        "中文 - 播报男声": "Chinese (Mandarin)_Male_Announcer",
        "中文 - 甜美女声": "Chinese (Mandarin)_Sweet_Lady",
        "中文 - 南方小哥": "Chinese (Mandarin)_Southern_Young_Man",
        "中文 - 阅历姐姐": "Chinese (Mandarin)_Wise_Women",
        "中文 - 温润青年": "Chinese (Mandarin)_Gentle_Youth",
        "中文 - 温暖少女": "Chinese (Mandarin)_Warm_Girl",
        "中文 - 花甲奶奶": "Chinese (Mandarin)_Kind-hearted_Elder",
        "中文 - 憨憨萌兽": "Chinese (Mandarin)_Cute_Spirit",
        "中文 - 电台男主播": "Chinese (Mandarin)_Radio_Host",
        "中文 - 抒情男声": "Chinese (Mandarin)_Lyrical_Voice",
        "中文 - 率真弟弟": "Chinese (Mandarin)_Straightforward_Boy",
        "中文 - 真诚青年": "Chinese (Mandarin)_Sincere_Adult",
        "中文 - 温柔学姐": "Chinese (Mandarin)_Gentle_Senior",
        "中文 - 嘴硬竹马": "Chinese (Mandarin)_Stubborn_Friend",
        "中文 - 清脆少女": "Chinese (Mandarin)_Crisp_Girl",
        "中文 - 清澈邻家弟弟": "Chinese (Mandarin)_Pure-hearted_Boy",
        "中文 - 软软女孩": "Chinese (Mandarin)_Soft_Girl",
        # 中文 (粤语) 音色
        "中文 (粤语) - 专业女主持": "Cantonese_ProfessionalHost(F)",
        "中文 (粤语) - 温柔女声": "Cantonese_GentleLady",
        "中文 (粤语) - 专业男主持": "Cantonese_ProfessionalHost(M)",
        "中文 (粤语) - 活泼男声": "Cantonese_PlayfulMan",
        "中文 (粤语) - 可爱女孩": "Cantonese_CuteGirl",
        "中文 (粤语) - 善良女声": "Cantonese_KindWoman",
        # 英文音色
        "英文 - Santa Claus": "Santa_Claus",
        "英文 - Grinch": "Grinch",
        "英文 - Rudolph": "Rudolph",
        "英文 - Arnold": "Arnold",
        "英文 - Charming Santa": "Charming_Santa",
        "英文 - Charming Lady": "Charming_Lady",
        "英文 - Sweet Girl": "Sweet_Girl",
        "英文 - Cute Elf": "Cute_Elf",
        "英文 - Attractive Girl": "Attractive_Girl",
        "英文 - Serene Woman": "Serene_Woman",
        "英文 - Trustworthy Man": "English_Trustworthy_Man",
        "英文 - Graceful Lady": "English_Graceful_Lady",
        "英文 - Aussie Bloke": "English_Aussie_Bloke",
        "英文 - Whispering girl": "English_Whispering_girl",
        "英文 - Diligent Man": "English_Diligent_Man",
        "英文 - Gentle-voiced man": "English_Gentle-voiced_man",
        # 日文音色
        "日文 - Intellectual Senior": "Japanese_IntellectualSenior",
        "日文 - Decisive Princess": "Japanese_DecisivePrincess",
        "日文 - Loyal Knight": "Japanese_LoyalKnight",
        "日文 - Dominant Man": "Japanese_DominantMan",
        "日文 - Serious Commander": "Japanese_SeriousCommander",
        "日文 - Cold Queen": "Japanese_ColdQueen",
        "日文 - Dependable Woman": "Japanese_DependableWoman",
        "日文 - Gentle Butler": "Japanese_GentleButler",
        "日文 - Kind Lady": "Japanese_KindLady",
        "日文 - Calm Lady": "Japanese_CalmLady",
        "日文 - Optimistic Youth": "Japanese_OptimisticYouth",
        "日文 - Generous Izakaya Owner": "Japanese_GenerousIzakayaOwner",
        "日文 - Sporty Student": "Japanese_SportyStudent",
        "日文 - Innocent Boy": "Japanese_InnocentBoy",
        "日文 - Graceful Maiden": "Japanese_GracefulMaiden",
        # 韩文音色
        "韩文 - Sweet Girl": "Korean_SweetGirl",
        "韩文 - Cheerful Boyfriend": "Korean_CheerfulBoyfriend",
        "韩文 - Enchanting Sister": "Korean_EnchantingSister",
        "韩文 - Shy Girl": "Korean_ShyGirl",
        "韩文 - Reliable Sister": "Korean_ReliableSister",
        "韩文 - Strict Boss": "Korean_StrictBoss",
        "韩文 - Sassy Girl": "Korean_SassyGirl",
        "韩文 - Childhood Friend Girl": "Korean_ChildhoodFriendGirl",
        "韩文 - Playboy Charmer": "Korean_PlayboyCharmer",
        "韩文 - Elegant Princess": "Korean_ElegantPrincess",
        "韩文 - Brave Female Warrior": "Korean_BraveFemaleWarrior",
        "韩文 - Brave Youth": "Korean_BraveYouth",
        "韩文 - Calm Lady": "Korean_CalmLady",
        "韩文 - Enthusiastic Teen": "Korean_EnthusiasticTeen",
        "韩文 - Soothing Lady": "Korean_SoothingLady",
        "韩文 - Intellectual Senior": "Korean_IntellectualSenior",
        "韩文 - Lonely Warrior": "Korean_LonelyWarrior",
        "韩文 - Mature Lady": "Korean_MatureLady",
        "韩文 - Innocent Boy": "Korean_InnocentBoy",
        "韩文 - Charming Sister": "Korean_CharmingSister",
        "韩文 - Athletic Student": "Korean_AthleticStudent",
        "韩文 - Brave Adventurer": "Korean_BraveAdventurer",
        "韩文 - Calm Gentleman": "Korean_CalmGentleman",
        "韩文 - Wise Elf": "Korean_WiseElf",
        "韩文 - Cheerful Cool Junior": "Korean_CheerfulCoolJunior",
        "韩文 - Decisive Queen": "Korean_DecisiveQueen",
        "韩文 - Cold Young Man": "Korean_ColdYoungMan",
        "韩文 - Mysterious Girl": "Korean_MysteriousGirl",
        "韩文 - Quirky Girl": "Korean_QuirkyGirl",
        "韩文 - Considerate Senior": "Korean_ConsiderateSenior",
        "韩文 - Cheerful Little Sister": "Korean_CheerfulLittleSister",
        "韩文 - Dominant Man": "Korean_DominantMan",
        "韩文 - Airheaded Girl": "Korean_AirheadedGirl",
        "韩文 - Reliable Youth": "Korean_ReliableYouth",
        "韩文 - Friendly Big Sister": "Korean_FriendlyBigSister",
        "韩文 - Gentle Boss": "Korean_GentleBoss",
        "韩文 - Cold Girl": "Korean_ColdGirl",
        "韩文 - Haughty Lady": "Korean_HaughtyLady",
        "韩文 - Charming Elder Sister": "Korean_CharmingElderSister",
        "韩文 - Intellectual Man": "Korean_IntellectualMan",
        "韩文 - Caring Woman": "Korean_CaringWoman",
        "韩文 - Wise Teacher": "Korean_WiseTeacher",
        "韩文 - Confident Boss": "Korean_ConfidentBoss",
        "韩文 - Athletic Girl": "Korean_AthleticGirl",
        "韩文 - Possessive Man": "Korean_PossessiveMan",
        "韩文 - Gentle Woman": "Korean_GentleWoman",
        "韩文 - Cocky Guy": "Korean_CockyGuy",
        "韩文 - Thoughtful Woman": "Korean_ThoughtfulWoman",
        "韩文 - Optimistic Youth": "Korean_OptimisticYouth",
        # 西班牙文音色
        "西班牙文 - Serene Woman": "Spanish_SereneWoman",
        "西班牙文 - Mature Partner": "Spanish_MaturePartner",
        "西班牙文 - Captivating Storyteller": "Spanish_CaptivatingStoryteller",
        "西班牙文 - Narrator": "Spanish_Narrator",
        "西班牙文 - Wise Scholar": "Spanish_WiseScholar",
        "西班牙文 - Kind-hearted Girl": "Spanish_Kind-heartedGirl",
        "西班牙文 - Determined Manager": "Spanish_DeterminedManager",
        "西班牙文 - Bossy Leader": "Spanish_BossyLeader",
        "西班牙文 - Reserved Young Man": "Spanish_ReservedYoungMan",
        "西班牙文 - Confident Woman": "Spanish_ConfidentWoman",
        "西班牙文 - Thoughtful Man": "Spanish_ThoughtfulMan",
        "西班牙文 - Strong-willed Boy": "Spanish_Strong-WilledBoy",
        "西班牙文 - Sophisticated Lady": "Spanish_SophisticatedLady",
        "西班牙文 - Rational Man": "Spanish_RationalMan",
        "西班牙文 - Anime Character": "Spanish_AnimeCharacter",
        "西班牙文 - Deep-toned Man": "Spanish_Deep-tonedMan",
        "西班牙文 - Fussy hostess": "Spanish_Fussyhostess",
        "西班牙文 - Sincere Teen": "Spanish_SincereTeen",
        "西班牙文 - Frank Lady": "Spanish_FrankLady",
        "西班牙文 - Comedian": "Spanish_Comedian",
        "西班牙文 - Debator": "Spanish_Debator",
        "西班牙文 - Tough Boss": "Spanish_ToughBoss",
        "西班牙文 - Wise Lady": "Spanish_Wiselady",
        "西班牙文 - Steady Mentor": "Spanish_Steadymentor",
        "西班牙文 - Jovial Man": "Spanish_Jovialman",
        "西班牙文 - Santa Claus": "Spanish_SantaClaus",
        "西班牙文 - Rudolph": "Spanish_Rudolph",
        "西班牙文 - Intonate Girl": "Spanish_Intonategirl",
        "西班牙文 - Arnold": "Spanish_Arnold",
        "西班牙文 - Ghost": "Spanish_Ghost",
        "西班牙文 - Humorous Elder": "Spanish_HumorousElder",
        "西班牙文 - Energetic Boy": "Spanish_EnergeticBoy",
        "西班牙文 - Whimsical Girl": "Spanish_WhimsicalGirl",
        "西班牙文 - Strict Boss": "Spanish_StrictBoss",
        "西班牙文 - Reliable Man": "Spanish_ReliableMan",
        "西班牙文 - Serene Elder": "Spanish_SereneElder",
        "西班牙文 - Angry Man": "Spanish_AngryMan",
        "西班牙文 - Assertive Queen": "Spanish_AssertiveQueen",
        "西班牙文 - Caring Girlfriend": "Spanish_CaringGirlfriend",
        "西班牙文 - Powerful Soldier": "Spanish_PowerfulSoldier",
        "西班牙文 - Passionate Warrior": "Spanish_PassionateWarrior",
        "西班牙文 - Chatty Girl": "Spanish_ChattyGirl",
        "西班牙文 - Romantic Husband": "Spanish_RomanticHusband",
        "西班牙文 - Compelling Girl": "Spanish_CompellingGirl",
        # 葡萄牙文音色
        "葡萄牙文 - Sentimental Lady": "Portuguese_SentimentalLady",
        "葡萄牙文 - Bossy Leader": "Portuguese_BossyLeader",
        "葡萄牙文 - Wise Lady": "Portuguese_Wiselady",
        "葡萄牙文 - Strong-willed Boy": "Portuguese_Strong-WilledBoy",
        "葡萄牙文 - Deep-voiced Gentleman": "Portuguese_Deep-VoicedGentleman",
        "葡萄牙文 - Upset Girl": "Portuguese_UpsetGirl",
        "葡萄牙文 - Passionate Warrior": "Portuguese_PassionateWarrior",
        "葡萄牙文 - Anime Character": "Portuguese_AnimeCharacter",
        "葡萄牙文 - Confident Woman": "Portuguese_ConfidentWoman",
        "葡萄牙文 - Angry Man": "Portuguese_AngryMan",
        "葡萄牙文 - Captivating Storyteller": "Portuguese_CaptivatingStoryteller",
        "葡萄牙文 - Godfather": "Portuguese_Godfather",
        "葡萄牙文 - Reserved Young Man": "Portuguese_ReservedYoungMan",
        "葡萄牙文 - Smart Young Girl": "Portuguese_SmartYoungGirl",
        "葡萄牙文 - Kind-hearted Girl": "Portuguese_Kind-heartedGirl",
        "葡萄牙文 - Pompous Lady": "Portuguese_Pompouslady",
        "葡萄牙文 - Grinch": "Portuguese_Grinch",
        "葡萄牙文 - Debator": "Portuguese_Debator",
        "葡萄牙文 - Sweet Girl": "Portuguese_SweetGirl",
        "葡萄牙文 - Attractive Girl": "Portuguese_AttractiveGirl",
        "葡萄牙文 - Thoughtful Man": "Portuguese_ThoughtfulMan",
        "葡萄牙文 - Playful Girl": "Portuguese_PlayfulGirl",
        "葡萄牙文 - Gorgeous Lady": "Portuguese_GorgeousLady",
        "葡萄牙文 - Lovely Lady": "Portuguese_LovelyLady",
        "葡萄牙文 - Serene Woman": "Portuguese_SereneWoman",
        "葡萄牙文 - Sad Teen": "Portuguese_SadTeen",
        "葡萄牙文 - Mature Partner": "Portuguese_MaturePartner",
        "葡萄牙文 - Comedian": "Portuguese_Comedian",
        "葡萄牙文 - Naughty Schoolgirl": "Portuguese_NaughtySchoolgirl",
        "葡萄牙文 - Narrator": "Portuguese_Narrator",
        "葡萄牙文 - Tough Boss": "Portuguese_ToughBoss",
        "葡萄牙文 - Fussy Hostess": "Portuguese_Fussyhostess",
        "葡萄牙文 - Dramatist": "Portuguese_Dramatist",
        "葡萄牙文 - Steady Mentor": "Portuguese_Steadymentor",
        "葡萄牙文 - Jovial Man": "Portuguese_Jovialman",
        "葡萄牙文 - Charming Queen": "Portuguese_CharmingQueen",
        "葡萄牙文 - Santa Claus": "Portuguese_SantaClaus",
        "葡萄牙文 - Rudolph": "Portuguese_Rudolph",
        "葡萄牙文 - Arnold": "Portuguese_Arnold",
        "葡萄牙文 - Charming Santa": "Portuguese_CharmingSanta",
        "葡萄牙文 - Charming Lady": "Portuguese_CharmingLady",
        "葡萄牙文 - Ghost": "Portuguese_Ghost",
        "葡萄牙文 - Humorous Elder": "Portuguese_HumorousElder",
        "葡萄牙文 - Calm Leader": "Portuguese_CalmLeader",
        "葡萄牙文 - Gentle Teacher": "Portuguese_GentleTeacher",
        "葡萄牙文 - Energetic Boy": "Portuguese_EnergeticBoy",
        "葡萄牙文 - Reliable Man": "Portuguese_ReliableMan",
        "葡萄牙文 - Serene Elder": "Portuguese_SereneElder",
        "葡萄牙文 - Grim Reaper": "Portuguese_GrimReaper",
        "葡萄牙文 - Assertive Queen": "Portuguese_AssertiveQueen",
        "葡萄牙文 - Whimsical Girl": "Portuguese_WhimsicalGirl",
        "葡萄牙文 - Stressed Lady": "Portuguese_StressedLady",
        "葡萄牙文 - Friendly Neighbor": "Portuguese_FriendlyNeighbor",
        "葡萄牙文 - Caring Girlfriend": "Portuguese_CaringGirlfriend",
        "葡萄牙文 - Powerful Soldier": "Portuguese_PowerfulSoldier",
        "葡萄牙文 - Fascinating Boy": "Portuguese_FascinatingBoy",
        "葡萄牙文 - Romantic Husband": "Portuguese_RomanticHusband",
        "葡萄牙文 - Strict Boss": "Portuguese_StrictBoss",
        "葡萄牙文 - Inspiring Lady": "Portuguese_InspiringLady",
        "葡萄牙文 - Playful Spirit": "Portuguese_PlayfulSpirit",
        "葡萄牙文 - Elegant Girl": "Portuguese_ElegantGirl",
        "葡萄牙文 - Compelling Girl": "Portuguese_CompellingGirl",
        "葡萄牙文 - Powerful Veteran": "Portuguese_PowerfulVeteran",
        "葡萄牙文 - Sensible Manager": "Portuguese_SensibleManager",
        "葡萄牙文 - Thoughtful Lady": "Portuguese_ThoughtfulLady",
        "葡萄牙文 - Theatrical Actor": "Portuguese_TheatricalActor",
        "葡萄牙文 - Fragile Boy": "Portuguese_FragileBoy",
        "葡萄牙文 - Chatty Girl": "Portuguese_ChattyGirl",
        "葡萄牙文 - Conscientious Instructor": "Portuguese_Conscientiousinstructor",
        "葡萄牙文 - Rational Man": "Portuguese_RationalMan",
        "葡萄牙文 - Wise Scholar": "Portuguese_WiseScholar",
        "葡萄牙文 - Frank Lady": "Portuguese_FrankLady",
        "葡萄牙文 - Determined Manager": "Portuguese_DeterminedManager",
        # 法文音色
        "法文 - Level-Headed Man": "French_Male_Speech_New",
        "法文 - Patient Female Presenter": "French_Female_News Anchor",
        "法文 - Casual Man": "French_CasualMan",
        "法文 - Movie Lead Female": "French_MovieLeadFemale",
        "法文 - Female Anchor": "French_FemaleAnchor",
        "法文 - Male Narrator": "French_MaleNarrator",
        # 印尼文音色
        "印尼文 - Sweet Girl": "Indonesian_SweetGirl",
        "印尼文 - Reserved Young Man": "Indonesian_ReservedYoungMan",
        "印尼文 - Charming Girl": "Indonesian_CharmingGirl",
        "印尼文 - Calm Woman": "Indonesian_CalmWoman",
        "印尼文 - Confident Woman": "Indonesian_ConfidentWoman",
        "印尼文 - Caring Man": "Indonesian_CaringMan",
        "印尼文 - Bossy Leader": "Indonesian_BossyLeader",
        "印尼文 - Determined Boy": "Indonesian_DeterminedBoy",
        "印尼文 - Gentle Girl": "Indonesian_GentleGirl",
        # 德文音色
        "德文 - Friendly Man": "German_FriendlyMan",
        "德文 - Sweet Lady": "German_SweetLady",
        "德文 - Playful Man": "German_PlayfulMan",
        # 俄文音色
        "俄文 - Handsome Childhood Friend": "Russian_HandsomeChildhoodFriend",
        "俄文 - Bright Queen": "Russian_BrightHeroine",
        "俄文 - Ambitious Woman": "Russian_AmbitiousWoman",
        "俄文 - Reliable Man": "Russian_ReliableMan",
        "俄文 - Crazy Girl": "Russian_CrazyQueen",
        "俄文 - Pessimistic Girl": "Russian_PessimisticGirl",
        "俄文 - Attractive Guy": "Russian_AttractiveGuy",
        "俄文 - Bad-tempered Boy": "Russian_Bad-temperedBoy",
        # 意大利文音色
        "意大利文 - Brave Heroine": "Italian_BraveHeroine",
        "意大利文 - Narrator": "Italian_Narrator",
        "意大利文 - Wandering Sorcerer": "Italian_WanderingSorcerer",
        "意大利文 - Diligent Leader": "Italian_DiligentLeader",
        # 阿拉伯文音色
        "阿拉伯文 - Calm Woman": "Arabic_CalmWoman",
        "阿拉伯文 - Friendly Guy": "Arabic_FriendlyGuy",
        # 土耳其文音色
        "土耳其文 - Calm Woman": "Turkish_CalmWoman",
        "土耳其文 - Trustworthy Man": "Turkish_Trustworthyman",
        # 乌克兰文音色
        "乌克兰文 - Calm Woman": "Ukrainian_CalmWoman",
        "乌克兰文 - Wise Scholar": "Ukrainian_WiseScholar",
        # 荷兰文音色
        "荷兰文 - Kind-hearted Girl": "Dutch_kindhearted_girl",
        "荷兰文 - Bossy Leader": "Dutch_bossy_leader",
        # 越南文音色
        "越南文 - Kind-hearted Girl": "Vietnamese_kindhearted_girl"
    }
    
    # 定义可用的音色列表
    VOICE_LIST = list(VOICE_NAME_TO_ID.keys())
    
    # 定义可用的模型列表
    MODEL_LIST = [
        "speech-2.5-hd-preview",
        "speech-2.5-turbo-preview",
        "speech-02-hd",
        "speech-02-turbo",
        "speech-01-hd",
        "speech-01-turbo"
    ]
    
    # 定义可用的情绪列表
    EMOTION_LIST = [
        "happy", "sad", "angry", "fearful", "disgusted", "surprised", "calm"
    ]
    
    # 定义可用的音频格式列表
    FORMAT_LIST = [
        "mp3", "pcm", "flac", "wav"
    ]
    
    # 定义可用的采样率列表
    SAMPLE_RATE_LIST = [
        8000, 16000, 22050, 24000, 32000, 44100
    ]
    
    # 定义可用的比特率列表
    BITRATE_LIST = [
        32000, 64000, 128000, 256000
    ]
    
    # 定义可用的声道数列表
    CHANNEL_LIST = [
        1, 2
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        """定义节点输入类型"""
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "请输入要合成的文本"}),
                "voice": (cls.VOICE_LIST, {"default": "中文 - 青涩青年音色"}),
                "model": (cls.MODEL_LIST, {"default": "speech-2.5-turbo-preview"}),
            },
            "optional": {
                "voice_id": ("STRING", {"forceInput": True, "tooltip": "音色ID输入端口。当连接此端口时，将忽略'音色'选择器的选择。"}),  # 添加音色ID输入端口
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "vol": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "pitch": ("INT", {"default": 0, "min": -12, "max": 12, "step": 1}),
                "emotion": (cls.EMOTION_LIST, {"default": "calm"}),
                "text_normalization": ("BOOLEAN", {"default": False}),
                "format": (cls.FORMAT_LIST, {"default": "wav"}),
                "sample_rate": (cls.SAMPLE_RATE_LIST, {"default": 24000}),
                "bitrate": (cls.BITRATE_LIST, {"default": 128000}),
                "channel": (cls.CHANNEL_LIST, {"default": 1}),
                "api_key": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate_speech"
    CATEGORY = "CC-API/Audio"
    
    def generate_speech(
        self,
        text,
        voice,
        model,
        voice_id="",  # 添加音色ID参数
        speed=1.0,
        vol=1.0,
        pitch=0,
        emotion="calm",
        text_normalization=False,
        format="wav",
        sample_rate=24000,
        bitrate=128000,
        channel=1,
        api_key=""
    ):
        """生成语音"""
        
        # 检查API密钥
        if not api_key:
            # 尝试从配置文件获取API密钥
            api_key = CCConfig().get_minimax_key()
            if not api_key:
                print("Error: No MiniMax API key provided")
                return self._create_blank_audio(sample_rate)
        
        # 检查文本长度
        if len(text) > 10000:
            print("Warning: Text exceeds maximum length of 10000 characters. Truncating...")
            text = text[:10000]
        
        # 如果提供了voice_id，则使用它；否则使用voice参数转换的音色ID
        if voice_id:
            # 使用直接提供的音色ID
            selected_voice_id = voice_id
        else:
            # 将中文显示名称转换为API所需的音色ID
            selected_voice_id = self.VOICE_NAME_TO_ID.get(voice, voice)
        
        # 准备请求数据
        request_data = {
            "model": model,
            "text": text,
            "stream": False,  # 使用非流式同步语音合成
            "voice_setting": {
                "voice_id": selected_voice_id,
                "speed": speed,
                "vol": vol,
                "pitch": pitch,
                "emotion": emotion
            },
            "audio_setting": {
                "sample_rate": sample_rate,
                "format": format,
                "channel": channel
            },
            "text_normalization": text_normalization
        }
        
        # 如果是mp3格式，添加比特率设置
        if format == "mp3":
            request_data["audio_setting"]["bitrate"] = bitrate
        
        try:
            # 发送请求
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                "https://api.minimaxi.com/v1/t2a_v2",
                headers=headers,
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # 检查响应中是否包含音频数据
                if "data" in result and "audio" in result["data"]:
                    audio_hex = result["data"]["audio"]
                    
                    # 将十六进制数据转换为二进制
                    audio_binary = bytes.fromhex(audio_hex)
                    
                    # 将音频数据保存到临时文件
                    with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as temp_file:
                        temp_file.write(audio_binary)
                        temp_file_path = temp_file.name
                    
                    # 如果不是wav格式，需要转换为wav
                    if format != "wav":
                        # 这里应该添加格式转换逻辑，但为了简化，我们假设返回的是wav格式
                        # 实际应用中可能需要使用ffmpeg或其他工具进行格式转换
                        pass
                    
                    # 读取音频文件并转换为波形数据
                    audio_sample_rate, waveform = wavfile.read(temp_file_path)
                    
                    # 确保波形数据是float32格式
                    if waveform.dtype != np.float32:
                        # 如果是整数类型，转换为float32并归一化到[-1, 1]范围
                        if np.issubdtype(waveform.dtype, np.integer):
                            max_val = np.max(np.abs(waveform))
                            if max_val > 0:
                                waveform = waveform.astype(np.float32) / max_val
                            else:
                                waveform = waveform.astype(np.float32)
                        else:
                            waveform = waveform.astype(np.float32)
                    
                    # 确保波形数据形状正确 [B, C, T]
                    if waveform.ndim == 1:
                        # 单声道，添加批次和声道维度 [1, 1, T]
                        waveform = waveform.reshape(1, 1, -1)
                    elif waveform.ndim == 2:
                        # 已经是二维数组，可能是 [C, T] 或 [T, C]
                        if waveform.shape[0] <= 2 and waveform.shape[0] < waveform.shape[1]:
                            # 可能是 [C, T]，添加批次维度 [1, C, T]
                            waveform = waveform.reshape(1, waveform.shape[0], -1)
                        else:
                            # 可能是 [T, C]，需要转置并添加批次维度 [1, C, T]
                            waveform = waveform.T.reshape(1, -1, waveform.shape[0])
                    
                    # 转换为PyTorch张量
                    waveform_tensor = torch.from_numpy(waveform)
                    
                    # 删除临时文件
                    os.unlink(temp_file_path)
                    
                    # 返回ComfyUI期望的音频格式
                    return ({
                        "waveform": waveform_tensor,
                        "sample_rate": audio_sample_rate
                    },)
                else:
                    print("Error: No audio data in response")
                    return self._create_blank_audio(sample_rate)
            else:
                print(f"API request failed with status {response.status_code}: {response.text}")
                return self._create_blank_audio(sample_rate)
        
        except Exception as e:
            print(f"Error generating speech: {str(e)}")
            return self._create_blank_audio(sample_rate)

    def _create_blank_audio(self, sample_rate=24000):
        """创建一个空白音频文件"""
        # 创建一个短暂的静音音频文件
        duration = 0.1  # 0.1秒的静音
        silence = np.zeros(int(sample_rate * duration), dtype=np.float32)
        
        # 确保波形数据形状正确 [B, C, T]
        # 单声道，添加批次和声道维度 [1, 1, T]
        silence = silence.reshape(1, 1, -1)
        
        # 转换为PyTorch张量
        silence_tensor = torch.from_numpy(silence)
        
        # 返回ComfyUI期望的音频格式
        return ({
            "waveform": silence_tensor,
            "sample_rate": sample_rate
        },)


class MiniMaxVoiceSelector:
    """MiniMax音色选择器节点"""
    
    # 音色数据文件路径
    VOICE_DATA_FILE = os.path.join(os.path.dirname(__file__), "minimax_voices.json")
    
    # 类变量存储音色数据
    voice_mapping = {}
    voice_names = []
    
    @classmethod
    def _load_voice_data(cls):
        """从本地JSON文件加载音色数据"""
        try:
            if os.path.exists(cls.VOICE_DATA_FILE):
                with open(cls.VOICE_DATA_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    cls.voice_mapping = data.get('voice_mapping', {})
                    cls.voice_names = list(cls.voice_mapping.keys()) if cls.voice_mapping else []
            else:
                # 如果文件不存在，初始化为空数据
                cls.voice_mapping = {}
                cls.voice_names = []
        except Exception as e:
            print(f"Error loading voice data: {e}")
            cls.voice_mapping = {}
            cls.voice_names = []
    
    @classmethod
    def _save_voice_data(cls, voice_data):
        """保存音色数据到本地JSON文件"""
        try:
            # 重新组织数据结构以便保存
            data = {
                'voice_mapping': voice_data,
                'updated_at': np.datetime_as_string(np.datetime64('now'))
            }
            
            with open(cls.VOICE_DATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            # 重新加载数据
            cls._load_voice_data()
        except Exception as e:
            print(f"Error saving voice data: {e}")
    
    @classmethod
    def _fetch_voice_data(cls, api_key, voice_type="all"):
        """从API获取音色数据"""
        if not api_key:
            # 尝试从配置文件获取API密钥
            api_key = CCConfig().get_minimax_key()
            if not api_key:
                print("Error: No MiniMax API key provided")
                return {}
        
        try:
            # 准备请求数据
            request_data = {
                "voice_type": voice_type
            }
            
            # 发送请求
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                "https://api.minimaxi.com/v1/get_voice",
                headers=headers,
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # 检查响应状态
                if result.get("base_resp", {}).get("status_code", -1) == 0:
                    # 解析音色数据
                    voice_mapping = {}
                    
                    # 处理系统音色
                    system_voices = result.get("system_voice", [])
                    for voice in system_voices:
                        voice_name = voice.get("voice_name", "")
                        voice_id = voice.get("voice_id", "")
                        if voice_name and voice_id:
                            voice_mapping[voice_name] = voice_id
                    
                    # 处理快速复刻音色
                    cloning_voices = result.get("voice_cloning", [])
                    for voice in cloning_voices:
                        voice_id = voice.get("voice_id", "")
                        # 使用voice_id作为名称，因为可能没有voice_name
                        if voice_id:
                            voice_mapping[f"快速复刻 - {voice_id}"] = voice_id
                    
                    # 处理文生音色
                    generation_voices = result.get("voice_generation", [])
                    for voice in generation_voices:
                        voice_id = voice.get("voice_id", "")
                        # 使用voice_id作为名称，因为可能没有voice_name
                        if voice_id:
                            voice_mapping[f"文生音色 - {voice_id}"] = voice_id
                    
                    # 保存数据到本地
                    cls._save_voice_data(voice_mapping)
                    return voice_mapping
                else:
                    status_msg = result.get("base_resp", {}).get("status_msg", "Unknown error")
                    print(f"API request failed: {status_msg}")
                    return {}
            else:
                print(f"API request failed with status {response.status_code}: {response.text}")
                return {}
        except Exception as e:
            print(f"Error fetching voice data: {str(e)}")
            return {}
    
    @classmethod
    def INPUT_TYPES(cls):
        """定义节点输入类型"""
        # 加载音色数据
        cls._load_voice_data()
        
        return {
            "required": {
                "voice_name": (cls.voice_names if cls.voice_names else ["无可用音色"], {"default": cls.voice_names[0] if cls.voice_names else "无可用音色"}),
            },
            "optional": {
                "voice_type": (["all", "system", "voice_cloning", "voice_generation"], {"default": "all"}),
                "api_key": ("STRING", {"default": ""}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("voice_id",)
    FUNCTION = "select_voice"
    CATEGORY = "CC-API/Audio"
    
    def select_voice(self, voice_name, api_key="", voice_type="all", prompt=None, extra_pnginfo=None, unique_id=None):
        """选择音色并返回音色ID"""
        # 获取选中音色的ID
        voice_id = MiniMaxVoiceSelector.voice_mapping.get(voice_name, "")
        
        if not voice_id:
            print(f"Warning: Voice ID not found for '{voice_name}'")
            # 如果找不到对应的ID，返回空字符串
            return ("",)
        
        # 返回音色ID
        return (voice_id,)


# 添加API端点用于刷新音色列表
@server.PromptServer.instance.routes.post("/minimax_refresh_voices")
async def refresh_minimax_voices(request):
    """处理刷新MiniMax音色列表的请求"""
    try:
        # 获取请求数据
        post_data = await request.json()
        api_key = post_data.get("api_key", "")
        voice_type = post_data.get("voice_type", "all")
        
        if not api_key:
            # 返回错误响应
            return web.json_response({
                "status": "error", 
                "error": "API密钥不能为空"
            })
        
        # 调用MiniMaxVoiceSelector类方法获取音色数据
        voice_mapping = MiniMaxVoiceSelector._fetch_voice_data(api_key, voice_type)
        
        if not voice_mapping:
            # 返回错误响应
            return web.json_response({
                "status": "error", 
                "error": "无法获取音色数据"
            })
        
        # 返回成功响应和音色名称列表
        voice_names = list(voice_mapping.keys())
        return web.json_response({
            "status": "success",
            "voice_names": voice_names,
            "count": len(voice_names)
        })
        
    except Exception as e:
        print(f"Error refreshing voices: {e}")
        # 返回错误响应
        return web.json_response({
            "status": "error", 
            "error": str(e)
        })


# 注册节点
NODE_CLASS_MAPPINGS = {
    "MiniMaxTTS": MiniMaxTTS,
    "MiniMaxVoiceSelector": MiniMaxVoiceSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxTTS": "MiniMax TTS",
    "MiniMaxVoiceSelector": "MiniMax Voice Selector",
}
