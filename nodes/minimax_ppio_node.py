import os
import json
import base64
import tempfile
import requests
import numpy as np
import torch
from scipy.io import wavfile
from .cc_utils import CCConfig
from .audio_utils import process_audio_for_minimax

# 尝试导入音频处理库
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

class MiniMaxPPIOTTS:
    """MiniMax TTS节点 (派欧云)"""
    
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
        "韩문 - Cheerful Little Sister": "Korean_CheerfulLittleSister",
        "韩문 - Dominant Man": "Korean_DominantMan",
        "韩문 - Airheaded Girl": "Korean_AirheadedGirl",
        "韩문 - Reliable Youth": "Korean_ReliableYouth",
        "韩문 - Friendly Big Sister": "Korean_FriendlyBigSister",
        "韩문 - Gentle Boss": "Korean_GentleBoss",
        "韩문 - Cold Girl": "Korean_ColdGirl",
        "韩문 - Haughty Lady": "Korean_HaughtyLady",
        "韩문 - Charming Elder Sister": "Korean_CharmingElderSister",
        "韩문 - Intellectual Man": "Korean_IntellectualMan",
        "韩문 - Caring Woman": "Korean_CaringWoman",
        "韩문 - Wise Teacher": "Korean_WiseTeacher",
        "韩문 - Confident Boss": "Korean_ConfidentBoss",
        "韩문 - Athletic Girl": "Korean_AthleticGirl",
        "韩문 - Possessive Man": "Korean_PossessiveMan",
        "韩문 - Gentle Woman": "Korean_GentleWoman",
        "韩문 - Cocky Guy": "Korean_CockyGuy",
        "韩문 - Thoughtful Woman": "Korean_ThoughtfulWoman",
        "韩문 - Optimistic Youth": "Korean_OptimisticYouth",
        # 西班牙文音色
        "西班牙文 - Serene Woman": "Spanish_SereneWoman",
        "西班牙文 - Mature Partner": "Spanish_MaturePartner",
        "西班牙文 - Captivating Storyteller": "Spanish_CaptivatingStoryteller",
        "西班牙文 - Narrator": "Spanish_Narrator",
        "西班牙文 - Wise Scholar": "Spanish_WiseScholar",
        "西班牙文 - Kind-hearted Girl": "Spanish_Kind-heartedGirl",
        "西班牙文 - Determined Manager": "Spanish_DeterminedManager",
        "Spanish文 - Bossy Leader": "Spanish_BossyLeader",
        "Spanish文 - Reserved Young Man": "Spanish_ReservedYoungMan",
        "Spanish文 - Confident Woman": "Spanish_ConfidentWoman",
        "Spanish文 - Thoughtful Man": "Spanish_ThoughtfulMan",
        "Spanish文 - Strong-willed Boy": "Spanish_Strong-WilledBoy",
        "Spanish文 - Sophisticated Lady": "Spanish_SophisticatedLady",
        "Spanish文 - Rational Man": "Spanish_RationalMan",
        "Spanish文 - Anime Character": "Spanish_AnimeCharacter",
        "Spanish文 - Deep-toned Man": "Spanish_Deep-tonedMan",
        "Spanish文 - Fussy hostess": "Spanish_Fussyhostess",
        "Spanish文 - Sincere Teen": "Spanish_SincereTeen",
        "Spanish文 - Frank Lady": "Spanish_FrankLady",
        "Spanish文 - Comedian": "Spanish_Comedian",
        "Spanish文 - Debator": "Spanish_Debator",
        "Spanish文 - Tough Boss": "Spanish_ToughBoss",
        "Spanish文 - Wise Lady": "Spanish_Wiselady",
        "Spanish文 - Steady Mentor": "Spanish_Steadymentor",
        "Spanish文 - Jovial Man": "Spanish_Jovialman",
        "Spanish文 - Santa Claus": "Spanish_SantaClaus",
        "Spanish文 - Rudolph": "Spanish_Rudolph",
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        """定义节点输入类型"""
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "你好，这是MiniMax TTS的示例文本。", "tooltip": "要合成的文本内容，支持多语言混合输入。最长输入为10000个字符。"}),
                "voice": (list(cls.VOICE_NAME_TO_ID.keys()), {"default": "中文 - 精英青年音色", "tooltip": "选择语音合成的音色"}),
                "model": (["speech-02-hd", "speech-02-turbo", "speech-2.5-hd-preview", "speech-2.5-turbo-preview"], {"default": "speech-2.5-hd-preview", "tooltip": "选择语音合成模型，不同模型在音质和速度上有所差异。"}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1, "tooltip": "调整语音播放速度，范围0.5-2.0，默认值为1.0。"}),
                "vol": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": "调整语音音量，范围0.1-10.0，默认值为1.0。"}),
                "pitch": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 1.0, "tooltip": "调整语音音调，范围-12到12，默认值为0。"}),
                "format": (["mp3", "pcm", "flac", "wav"], {"default": "mp3", "tooltip": "选择输出音频的格式。"}),
                "sample_rate": ([8000, 16000, 22050, 24000, 32000, 44100, 48000], {"default": 24000, "tooltip": "选择音频采样率，影响音频质量和文件大小。"}),
                "bitrate": ([32000, 64000, 128000, 192000, 256000, 320000], {"default": 128000, "tooltip": "选择音频比特率，仅对MP3格式有效，影响音频质量和文件大小。"}),
                "channel": ([1, 2], {"default": 1, "tooltip": "选择音频声道数，1为单声道，2为立体声。"}),
                "api_key": ("STRING", {"default": "", "tooltip": "派欧云API的访问密钥。如果未提供，将使用配置文件中的密钥。"}),
            },
            "optional": {
                "emotion": (["happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral"], {"default": "neutral", "tooltip": "选择语音情绪，影响语音的情感表达。"}),
                "text_normalization": ("BOOLEAN", {"default": True, "tooltip": "是否对文本进行规范化处理，如数字、日期等的转换。"}),
                "voice_id": ("STRING", {"default": "", "tooltip": "音色ID输入端口。当连接此端口时，将忽略'音色'选择器的选择。"}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "CC-API/Audio"
    
    def get_api_key(self, provided_key=""):
        """获取API密钥，优先级：参数 > 环境变量 > 配置文件"""
        # 首先检查传入的参数
        if provided_key:
            return provided_key
        
        # 然后检查环境变量
        env_var_name = "PPIO_API_KEY"
        if os.environ.get(env_var_name) is not None:
            return os.environ[env_var_name]
        
        # 最后检查配置文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        config_path = os.path.join(parent_dir, "config.ini")
        
        if os.path.exists(config_path):
            import configparser
            config = configparser.ConfigParser()
            # 在Windows系统上明确指定使用UTF-8编码读取配置文件
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config.read_file(f)
            except UnicodeDecodeError:
                # 如果UTF-8失败，尝试使用系统默认编码
                config.read(config_path)
            
            try:
                if "ppio" in config and "API_KEY" in config["ppio"]:
                    return config["ppio"]["API_KEY"]
            except KeyError:
                pass
        
        return None
    
    def generate_speech(
        self,
        text,
        voice,
        model,
        speed,
        vol,
        pitch,
        format,
        sample_rate,
        bitrate,
        channel,
        api_key="",
        emotion="calm",
        text_normalization=True,
        voice_id=""
    ):
        """生成语音"""
        
        # 检查API密钥
        api_key = self.get_api_key(api_key)
        if not api_key:
            raise ValueError("Error: No PPIO API key provided")
        
        # 检查文本长度
        if len(text) > 10000:
            raise ValueError("Error: Text length exceeds 10000 characters limit")
        
        # 确定使用的音色ID
        if voice_id and voice_id.strip():
            # 如果提供了voice_id，使用它
            selected_voice_id = voice_id.strip()
        else:
            # 否则使用选择的音色名称映射到ID
            selected_voice_id = self.VOICE_NAME_TO_ID.get(voice, "male-qn-jingying")
        
        try:
            # 调用MiniMax TTS API
            audio_data = self._call_tts_api(
                text,
                selected_voice_id,
                model,
                speed,
                vol,
                pitch,
                format,
                sample_rate,
                bitrate,
                channel,
                api_key,
                emotion,
                text_normalization
            )
            
            return (audio_data,)
            
        except Exception as e:
            raise ValueError(f"Error generating speech: {str(e)}")
    
    def _call_tts_api(
        self,
        text,
        voice_id,
        model,
        speed,
        vol,
        pitch,
        format,
        sample_rate,
        bitrate,
        channel,
        api_key,
        emotion,
        text_normalization
    ):
        """调用MiniMax TTS API"""
        try:
            # 确定API端点
            if model == "speech-02-hd":
                url = "https://api.ppinfra.com/v3/minimax-speech-02-hd"
            elif model == "speech-02-turbo":
                url = "https://api.ppinfra.com/v3/minimax-speech-02-turbo"
            elif model == "speech-2.5-hd-preview":
                url = "https://api.ppinfra.com/v3/minimax-speech-2.5-hd-preview"
            elif model == "speech-2.5-turbo-preview":
                url = "https://api.ppinfra.com/v3/minimax-speech-2.5-turbo-preview"
            else:
                url = "https://api.ppinfra.com/v3/minimax-speech-2.5-hd-preview"  # 默认值
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # 构建请求数据
            payload = {
                "text": text,
                "output_format": "hex",  # 使用hex格式返回
                "voice_setting": {
                    "speed": speed,
                    "vol": vol,
                    "pitch": pitch,
                    "voice_id": voice_id,
                    "emotion": emotion,
                    "text_normalization": text_normalization
                },
                "audio_setting": {
                    "sample_rate": sample_rate,
                    "bitrate": bitrate,
                    "format": format,
                    "channel": channel
                }
            }
            
            # 发送请求
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                if "audio" in result:
                    audio_hex = result["audio"]
                    # 将十六进制数据转换为二进制
                    audio_binary = bytes.fromhex(audio_hex)
                    # 直接处理音频数据而不是下载URL
                    audio_data = self._process_audio_binary(audio_binary, format)
                    return audio_data
                else:
                    raise ValueError(f"API response missing audio data: {result}")
            else:
                raise ValueError(f"API request failed with status {response.status_code}: {response.text}")
                
        except Exception as e:
            raise ValueError(f"Error calling MiniMax TTS API: {str(e)}")

    def _process_audio_binary(self, audio_binary, format):
        """处理二进制音频数据并返回AUDIO类型数据"""
        try:
            # 保存到临时文件
            with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as temp_file:
                temp_file.write(audio_binary)
                temp_filename = temp_file.name
            
            # 尝试读取音频文件
            waveform, sample_rate = None, None
            
            # 首先尝试使用soundfile读取
            if HAS_SOUNDFILE:
                try:
                    import soundfile as sf
                    waveform, sample_rate = sf.read(temp_filename, dtype='float32')
                except Exception as e:
                    print(f"Soundfile read failed: {e}")
            
            # 如果soundfile不可用或失败，尝试使用pydub
            if waveform is None and HAS_PYDUB:
                try:
                    # 直接使用已导入的AudioSegment
                    from pydub import AudioSegment
                    audio = AudioSegment.from_file(temp_filename)
                    # 转换为numpy数组
                    waveform = np.array(audio.get_array_of_samples(), dtype=np.float32)
                    # 归一化到[-1, 1]范围
                    waveform = waveform / (2 ** (audio.sample_width * 8 - 1))
                    sample_rate = audio.frame_rate
                    
                    # 如果是立体声，转换为单声道
                    if audio.channels > 1:
                        waveform = waveform.reshape((-1, audio.channels)).mean(axis=1)
                except Exception as e:
                    print(f"Pydub read failed: {e}")
            
            # 如果以上方法都失败，尝试使用scipy.io.wavfile
            if waveform is None:
                try:
                    sample_rate, waveform = wavfile.read(temp_filename)
                    # 转换为float32格式并归一化到[-1, 1]范围
                    if waveform.dtype != np.float32:
                        if np.issubdtype(waveform.dtype, np.integer):
                            max_val = np.iinfo(waveform.dtype).max
                            waveform = waveform.astype(np.float32) / max_val
                        else:
                            waveform = waveform.astype(np.float32)
                except Exception as e:
                    print(f"Wavfile read failed: {e}")
            
            # 删除临时文件
            os.unlink(temp_filename)
            
            # 如果仍然无法读取音频
            if waveform is None or sample_rate is None:
                raise ValueError("Failed to read audio file with any method")
            
            # 确保波形数据形状正确 [B, C, T]
            if waveform.ndim == 1:
                # 单声道，添加批次和声道维度 [1, 1, T]
                waveform = waveform.reshape(1, 1, -1)
            elif waveform.ndim == 2:
                # 已经是二维数组，可能是 [T, C]
                # 转置并添加批次维度 [1, C, T]
                waveform = waveform.T.reshape(1, waveform.shape[1], -1)
            
            # 转换为PyTorch张量
            waveform_tensor = torch.from_numpy(waveform)
            
            # 返回ComfyUI期望的音频格式
            return {
                "waveform": waveform_tensor,
                "sample_rate": sample_rate
            }
            
        except Exception as e:
            raise ValueError(f"Error processing audio binary: {str(e)}")


class MiniMaxPPIOVoiceClone:
    """MiniMax 声音克隆节点 (派欧云)"""
    
    @classmethod
    def INPUT_TYPES(cls):
        """定义节点输入类型"""
        return {
            "required": {
                "clone_audio": ("AUDIO", {"tooltip": "待克隆的音频文件"}),
                "voice_id": ("STRING", {"default": "cloned_voice_001", "tooltip": "自定义克隆音色ID，长度8-256，首字符必须为英文字母，允许数字、字母、-、_，末位不可为-、_"}),
            },
            "optional": {
                "prompt_audio": ("AUDIO", {"tooltip": "可选的示例音频文件，有助于增强音色相似度和稳定性"}),
                "prompt_text": ("STRING", {"multiline": True, "default": "", "tooltip": "示例音频对应的文本，需确保和音频内容一致，句末需有标点符号做结尾"}),
                "test_text": ("STRING", {"multiline": True, "default": "您好，这是克隆音色的试听音频。", "tooltip": "复刻试听参数，模型将使用复刻后的音色朗读本段文本内容"}),
                "model": (["speech-2.5-hd-preview", "speech-2.5-turbo-preview", "speech-02-hd", "speech-02-turbo"], {"default": "speech-2.5-hd-preview", "tooltip": "指定合成试听音频使用的语音模型"}),
                "need_noise_reduction": ("BOOLEAN", {"default": False, "tooltip": "音频复刻参数，表示是否开启降噪"}),
                "need_volume_normalization": ("BOOLEAN", {"default": False, "tooltip": "音频复刻参数，是否开启音量归一化"}),
                "api_key": ("STRING", {"default": "", "tooltip": "派欧云API的访问密钥。如果未提供，将使用配置文件中的密钥。"}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("demo_audio", "cloned_voice_id")
    FUNCTION = "clone_voice"
    CATEGORY = "CC-API/Audio"
    
    def get_api_key(self, provided_key=""):
        """获取API密钥，优先级：参数 > 环境变量 > 配置文件"""
        # 首先检查传入的参数
        if provided_key:
            return provided_key
        
        # 然后检查环境变量
        env_var_name = "PPIO_API_KEY"
        if os.environ.get(env_var_name) is not None:
            return os.environ[env_var_name]
        
        # 最后检查配置文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        config_path = os.path.join(parent_dir, "config.ini")
        
        if os.path.exists(config_path):
            import configparser
            config = configparser.ConfigParser()
            # 在Windows系统上明确指定使用UTF-8编码读取配置文件
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config.read_file(f)
            except UnicodeDecodeError:
                # 如果UTF-8失败，尝试使用系统默认编码
                config.read(config_path)
            
            try:
                if "ppio" in config and "API_KEY" in config["ppio"]:
                    return config["ppio"]["API_KEY"]
            except KeyError:
                pass
        
        return None
    
    def clone_voice(
        self,
        clone_audio,
        voice_id,
        prompt_audio=None,
        prompt_text="",
        test_text="您好，这是克隆音色的试听音频。",
        model="speech-2.5-hd-preview",
        api_key="",
        need_noise_reduction=False,
        need_volume_normalization=False
    ):
        """执行音色克隆"""
        
        # 检查API密钥
        api_key = self.get_api_key(api_key)
        if not api_key:
            raise ValueError("Error: No PPIO API key provided")
        
        try:
            # 1. 上传待克隆音频文件
            clone_audio_url = self._upload_audio_file(clone_audio, api_key, "voice_clone")
            if not clone_audio_url:
                raise ValueError("Error: Failed to upload clone audio file")
            
            # 2. 如果提供了示例音频，上传示例音频文件
            prompt_audio_url = None
            if prompt_audio is not None:
                prompt_audio_url = self._upload_audio_file(prompt_audio, api_key, "prompt_audio")
                if not prompt_audio_url:
                    # print("Warning: Failed to upload prompt audio file, continuing without it")
                    pass
            
            # 3. 调用音色克隆接口
            cloned_voice_id, demo_audio_url = self._call_voice_clone_api(
                clone_audio_url,
                voice_id,
                prompt_audio_url,
                prompt_text,
                test_text,
                model,
                api_key,
                need_noise_reduction,
                need_volume_normalization
            )
            
            if not cloned_voice_id:
                raise ValueError("Error: Failed to clone voice")
            
            # 4. 如果有试听音频URL，下载并返回音频
            demo_audio = self._create_blank_audio()
            if demo_audio_url:
                demo_audio = self._download_audio(demo_audio_url)
            
            # 返回试听音频和克隆的音色ID
            return (demo_audio, cloned_voice_id)
            
        except Exception as e:
            raise ValueError(f"Error cloning voice: {str(e)}")
    
    def _upload_audio_file(self, audio_data, api_key, purpose):
        """上传音频文件到派欧云"""
        temp_filename = None
        try:
            # 使用新的音频处理工具处理音频
            temp_filename, file_size = process_audio_for_minimax(audio_data)
            
            # 上传文件
            url = "https://api.ppinfra.com/v3/files/upload"
            headers = {
                "Authorization": f"Bearer {api_key}"
            }
            data = {
                "purpose": purpose
            }
            
            # 使用上下文管理器确保文件正确关闭
            with open(temp_filename, "rb") as f:
                files = {"file": f}
                response = requests.post(url, headers=headers, data=data, files=files)
            
            # 删除临时文件
            os.unlink(temp_filename)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("base_resp", {}).get("status_code", -1) == 0:
                    file_url = result.get("file", {}).get("url")
                    return file_url
                else:
                    status_msg = result.get("base_resp", {}).get("status_msg", "Unknown error")
                    raise ValueError(f"File upload failed: {status_msg}")
            else:
                raise ValueError(f"File upload failed with status {response.status_code}: {response.text}")
                
        except Exception as e:
            # 确保即使出错也删除临时文件
            if temp_filename and os.path.exists(temp_filename):
                os.unlink(temp_filename)
            raise ValueError(f"Error uploading audio file: {str(e)}")
    
    def _call_voice_clone_api(
        self,
        audio_url,
        voice_id,
        prompt_audio_url=None,
        prompt_text="",
        test_text="",
        model="speech-2.5-hd-preview",
        api_key="",
        need_noise_reduction=False,
        need_volume_normalization=False
    ):
        """调用音色克隆接口"""
        try:
            url = "https://api.ppinfra.com/v3/minimax-voice-cloning"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # 构建请求数据
            payload = {
                "audio_url": audio_url,
                "voice_id": voice_id,
                "need_noise_reduction": need_noise_reduction,
                "need_volume_normalization": need_volume_normalization
            }
            
            # 如果提供了示例音频和文本
            if prompt_audio_url and prompt_text:
                payload["clone_prompt"] = {
                    "prompt_audio": prompt_audio_url,
                    "prompt_text": prompt_text
                }
            
            # 如果提供了试听文本和模型
            if test_text and model:
                payload["text"] = test_text
                payload["model"] = model
            
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                demo_audio_url = result.get("demo_audio_url", "")
                cloned_voice_id = result.get("voice_id", "")
                return cloned_voice_id, demo_audio_url
            else:
                raise ValueError(f"Voice clone API request failed with status {response.status_code}: {response.text}")
                
        except Exception as e:
            raise ValueError(f"Error calling voice clone API: {str(e)}")
    
    def _download_audio(self, audio_url):
        """下载音频文件并返回AUDIO类型数据"""
        try:
            # 下载音频文件
            response = requests.get(audio_url)
            if response.status_code != 200:
                raise ValueError(f"Failed to download audio file: {response.status_code}")
            
            # 保存到临时文件
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(response.content)
                temp_filename = temp_file.name
            
            # 尝试使用soundfile读取音频文件（支持更多格式）
            if HAS_SOUNDFILE:
                try:
                    waveform, audio_sample_rate = sf.read(temp_filename)
                except Exception as e:
                    # 如果soundfile失败，回退到wavfile
                    audio_sample_rate, waveform = wavfile.read(temp_filename)
            else:
                # 使用wavfile读取音频文件
                audio_sample_rate, waveform = wavfile.read(temp_filename)
            
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
            os.unlink(temp_filename)
            
            # 返回ComfyUI期望的音频格式
            return {
                "waveform": waveform_tensor,
                "sample_rate": audio_sample_rate
            }
            
        except Exception as e:
            raise ValueError(f"Error downloading audio: {str(e)}")
    
    def _create_blank_audio(self):
        """创建空白音频数据"""
        # 创建一个1秒的空白音频
        sample_rate = 22050
        duration = 1  # 1秒
        samples = int(sample_rate * duration)
        audio_array = np.zeros(samples, dtype=np.float32)
        
        # 创建AUDIO类型数据
        audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)  # 添加通道维度
        audio_data = {
            "waveform": audio_tensor,
            "sample_rate": sample_rate
        }
        
        return audio_data


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "MiniMaxPPIOTTS": MiniMaxPPIOTTS,
    "MiniMaxPPIOVoiceClone": MiniMaxPPIOVoiceClone
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxPPIOTTS": "MiniMax TTS (派欧云)",
    "MiniMaxPPIOVoiceClone": "MiniMax 声音克隆 (派欧云)"
}