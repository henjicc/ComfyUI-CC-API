import os
import json
import base64
import tempfile
import requests
import numpy as np
import torch
import re
from scipy.io import wavfile
from .cc_utils import CCConfig

class DoubaoTTS:
    """豆包语音合成节点"""
    
    # 定义音色映射表：音色名称 -> (voice_type, resource_id)
    VOICE_MAP = {
        # 豆包语音合成模型2.0音色
        "vivi": ("zh_female_vv_uranus_bigtts", "seed-tts-2.0"),
        "小何": ("zh_female_xiaohe_jupiter_bigtts", "seed-tts-2.0"),
        "云舟": ("zh_male_yunzhou_jupiter_bigtts", "seed-tts-2.0"),
        "小天": ("zh_male_xiaotian_jupiter_bigtts", "seed-tts-2.0"),
        
        # 豆包语音合成模型1.0音色（多情感）
        "冷酷哥哥（多情感）": ("zh_male_lengkugege_emo_v2_mars_bigtts", "seed-tts-1.0"),
        "甜心小美（多情感）": ("zh_female_tianxinxiaomei_emo_v2_mars_bigtts", "seed-tts-1.0"),
        "高冷御姐（多情感）": ("zh_female_gaolengyujie_emo_v2_mars_bigtts", "seed-tts-1.0"),
        "傲娇霸总（多情感）": ("zh_male_aojiaobazong_emo_v2_mars_bigtts", "seed-tts-1.0"),
        "广州德哥（多情感）": ("zh_male_guangzhoudege_emo_mars_bigtts", "seed-tts-1.0"),
        "京腔侃爷（多情感）": ("zh_male_jingqiangkanye_emo_mars_bigtts", "seed-tts-1.0"),
        "邻居阿姨（多情感）": ("zh_female_linjuayi_emo_v2_mars_bigtts", "seed-tts-1.0"),
        "优柔公子（多情感）": ("zh_male_yourougongzi_emo_v2_mars_bigtts", "seed-tts-1.0"),
        "儒雅男友（多情感）": ("zh_male_ruyayichen_emo_v2_mars_bigtts", "seed-tts-1.0"),
        "俊朗男友（多情感）": ("zh_male_junlangnanyou_emo_v2_mars_bigtts", "seed-tts-1.0"),
        "北京小爷（多情感）": ("zh_male_beijingxiaoye_emo_v2_mars_bigtts", "seed-tts-1.0"),
        "柔美女友（多情感）": ("zh_female_roumeinvyou_emo_v2_mars_bigtts", "seed-tts-1.0"),
        "阳光青年（多情感）": ("zh_male_yangguangqingnian_emo_v2_mars_bigtts", "seed-tts-1.0"),
        "魅力女友（多情感）": ("zh_female_meilinvyou_emo_v2_mars_bigtts", "seed-tts-1.0"),
        "爽快思思（多情感）": ("zh_female_shuangkuaisisi_emo_v2_mars_bigtts", "seed-tts-1.0"),
        "Candice": ("en_female_candice_emo_v2_mars_bigtts", "seed-tts-1.0"),
        "Serena": ("en_female_skye_emo_v2_mars_bigtts", "seed-tts-1.0"),
        "Glen": ("en_male_glen_emo_v2_mars_bigtts", "seed-tts-1.0"),
        "Sylus": ("en_male_sylus_emo_v2_mars_bigtts", "seed-tts-1.0"),
        "Corey": ("en_male_corey_emo_v2_mars_bigtts", "seed-tts-1.0"),
        "Nadia": ("en_female_nadia_tips_emo_v2_mars_bigtts", "seed-tts-1.0"),
        "Tina老师": ("zh_female_yingyujiaoyu_mars_bigtts", "seed-tts-1.0"),
        
        # 豆包语音合成模型1.0音色
        "Vivi": ("zh_female_vv_mars_bigtts", "seed-tts-1.0"),
        "亲切女声": ("zh_female_qinqienvsheng_moon_bigtts", "seed-tts-1.0"),
        "阳光阿辰": ("zh_male_qingyiyuxuan_mars_bigtts", "seed-tts-1.0"),
        "快乐小东": ("zh_male_xudong_conversation_wvae_bigtts", "seed-tts-1.0"),
        "阳光青年": ("zh_male_yangguangqingnian_moon_bigtts", "seed-tts-1.0"),
        "甜美桃子": ("zh_female_tianmeitaozi_mars_bigtts", "seed-tts-1.0"),
        "清新女声": ("zh_female_qingxinnvsheng_mars_bigtts", "seed-tts-1.0"),
        "知性女声": ("zh_female_zhixingnvsheng_mars_bigtts", "seed-tts-1.0"),
        "清爽男大": ("zh_male_qingshuangnanda_mars_bigtts", "seed-tts-1.0"),
        "邻家女孩": ("zh_female_linjianvhai_moon_bigtts", "seed-tts-1.0"),
        "渊博小叔": ("zh_male_yuanboxiaoshu_moon_bigtts", "seed-tts-1.0"),
        "甜美小源": ("zh_female_tianmeixiaoyuan_moon_bigtts", "seed-tts-1.0"),
        "清澈梓梓": ("zh_female_qingchezizi_moon_bigtts", "seed-tts-1.0"),
        "解说小明": ("zh_male_jieshuoxiaoming_moon_bigtts", "seed-tts-1.0"),
        "开朗姐姐": ("zh_female_kailangjiejie_moon_bigtts", "seed-tts-1.0"),
        "邻家男孩": ("zh_male_linjiananhai_moon_bigtts", "seed-tts-1.0"),
        "甜美悦悦": ("zh_female_tianmeiyueyue_moon_bigtts", "seed-tts-1.0"),
        "心灵鸡汤": ("zh_female_xinlingjitang_moon_bigtts", "seed-tts-1.0"),
        "温柔小哥": ("zh_male_wenrouxiaoge_mars_bigtts", "seed-tts-1.0"),
        "灿灿/Shiny": ("zh_female_cancan_mars_bigtts", "seed-tts-1.0"),
        "爽快思思/Skye": ("zh_female_shuangkuaisisi_moon_bigtts", "seed-tts-1.0"),
        "温暖阿虎/Alvin": ("zh_male_wennuanahu_moon_bigtts", "seed-tts-1.0"),
        "少年梓辛/Brayan": ("zh_male_shaonianzixin_moon_bigtts", "seed-tts-1.0"),
        "沪普男": ("zh_male_hupunan_mars_bigtts", "seed-tts-1.0"),
        "鲁班七号": ("zh_male_lubanqihao_mars_bigtts", "seed-tts-1.0"),
        "林潇": ("zh_female_yangmi_mars_bigtts", "seed-tts-1.0"),
        "玲玲姐姐": ("zh_female_linzhiling_mars_bigtts", "seed-tts-1.0"),
        "春日部姐姐": ("zh_female_jiyejizi2_mars_bigtts", "seed-tts-1.0"),
        "唐僧": ("zh_male_tangseng_mars_bigtts", "seed-tts-1.0"),
        "庄周": ("zh_male_zhuangzhou_mars_bigtts", "seed-tts-1.0"),
        "猪八戒": ("zh_male_zhubajie_mars_bigtts", "seed-tts-1.0"),
        "感冒电音姐姐": ("zh_female_ganmaodianyin_mars_bigtts", "seed-tts-1.0"),
        "直率英子": ("zh_female_naying_mars_bigtts", "seed-tts-1.0"),
        "女雷神": ("zh_female_leidian_mars_bigtts", "seed-tts-1.0"),
        "豫州子轩": ("zh_male_yuzhouzixuan_moon_bigtts", "seed-tts-1.0"),
        "呆萌川妹": ("zh_female_daimengchuanmei_moon_bigtts", "seed-tts-1.0"),
        "广西远舟": ("zh_male_guangxiyuanzhou_moon_bigtts", "seed-tts-1.0"),
        "双节棍小哥": ("zh_male_zhoujielun_emo_v2_mars_bigtts", "seed-tts-1.0"),
        "湾湾小何": ("zh_female_wanwanxiaohe_moon_bigtts", "seed-tts-1.0"),
        "湾区大叔": ("zh_female_wanqudashu_moon_bigtts", "seed-tts-1.0"),
        "广州德哥": ("zh_male_guozhoudege_moon_bigtts", "seed-tts-1.0"),
        "浩宇小哥": ("zh_male_haoyuxiaoge_moon_bigtts", "seed-tts-1.0"),
        "北京小爷": ("zh_male_beijingxiaoye_moon_bigtts", "seed-tts-1.0"),
        "京腔侃爷/Harmony": ("zh_male_jingqiangkanye_moon_bigtts", "seed-tts-1.0"),
        "妹坨洁儿": ("zh_female_meituojieer_moon_bigtts", "seed-tts-1.0"),
        "高冷御姐": ("zh_female_gaolengyujie_moon_bigtts", "seed-tts-1.0"),
        "傲娇霸总": ("zh_male_aojiaobazong_moon_bigtts", "seed-tts-1.0"),
        "魅力女友": ("zh_female_meilinvyou_moon_bigtts", "seed-tts-1.0"),
        "深夜播客": ("zh_male_shenyeboke_moon_bigtts", "seed-tts-1.0"),
        "柔美女友": ("zh_female_sajiaonvyou_moon_bigtts", "seed-tts-1.0"),
        "撒娇学妹": ("zh_female_yuanqinvyou_moon_bigtts", "seed-tts-1.0"),
        "东方浩然": ("zh_male_dongfanghaoran_moon_bigtts", "seed-tts-1.0"),
        "悠悠君子": ("zh_male_M100_conversation_wvae_bigtts", "seed-tts-1.0"),
        "文静毛毛": ("zh_female_maomao_conversation_wvae_bigtts", "seed-tts-1.0"),
        "温柔小雅": ("zh_female_wenrouxiaoya_moon_bigtts", "seed-tts-1.0"),
        "天才童声": ("zh_male_tiancaitongsheng_mars_bigtts", "seed-tts-1.0"),
        "猴哥": ("zh_male_sunwukong_mars_bigtts", "seed-tts-1.0"),
        "熊二": ("zh_male_xionger_mars_bigtts", "seed-tts-1.0"),
        "佩奇猪": ("zh_female_peiqi_mars_bigtts", "seed-tts-1.0"),
        "武则天": ("zh_female_wuzetian_mars_bigtts", "seed-tts-1.0"),
        "顾姐": ("zh_female_gujie_mars_bigtts", "seed-tts-1.0"),
        "樱桃丸子": ("zh_female_yingtaowanzi_mars_bigtts", "seed-tts-1.0"),
        "广告解说": ("zh_male_chunhui_mars_bigtts", "seed-tts-1.0"),
        "少儿故事": ("zh_female_shaoergushi_mars_bigtts", "seed-tts-1.0"),
        "四郎": ("zh_male_silang_mars_bigtts", "seed-tts-1.0"),
        "俏皮女声": ("zh_female_qiaopinvsheng_mars_bigtts", "seed-tts-1.0"),
        "懒音绵宝": ("zh_male_lanxiaoyang_mars_bigtts", "seed-tts-1.0"),
        "亮嗓萌仔": ("zh_male_dongmanhaimian_mars_bigtts", "seed-tts-1.0"),
        "磁性解说男声/Morgan": ("zh_male_jieshuonansheng_mars_bigtts", "seed-tts-1.0"),
        "鸡汤妹妹/Hope": ("zh_female_jitangmeimei_mars_bigtts", "seed-tts-1.0"),
        "贴心女声/Candy": ("zh_female_tiexinnvsheng_mars_bigtts", "seed-tts-1.0"),
        "萌丫头/Cutey": ("zh_female_mengyatou_mars_bigtts", "seed-tts-1.0"),
        "儒雅青年": ("zh_male_ruyaqingnian_mars_bigtts", "seed-tts-1.0"),
        "霸气青叔": ("zh_male_baqiqingshu_mars_bigtts", "seed-tts-1.0"),
        "擎苍": ("zh_male_qingcang_mars_bigtts", "seed-tts-1.0"),
        "活力小哥": ("zh_male_yangguangqingnian_mars_bigtts", "seed-tts-1.0"),
        "古风少御": ("zh_female_gufengshaoyu_mars_bigtts", "seed-tts-1.0"),
        "温柔淑女": ("zh_female_wenroushunv_mars_bigtts", "seed-tts-1.0"),
        "反卷青年": ("zh_male_fanjuanqingnian_mars_bigtts", "seed-tts-1.0"),
        
        # 英文音色
        "Adam": ("en_male_adam_mars_bigtts", "seed-tts-1.0"),
        "Amanda": ("en_female_amanda_mars_bigtts", "seed-tts-1.0"),
        "Jackson": ("en_male_jackson_mars_bigtts", "seed-tts-1.0"),
        "Emily": ("en_female_emily_mars_bigtts", "seed-tts-1.0"),
        "Smith": ("en_male_smith_mars_bigtts", "seed-tts-1.0"),
        "Anna": ("en_female_anna_mars_bigtts", "seed-tts-1.0"),
        "Delicate Girl": ("en_female_daisy_moon_bigtts", "seed-tts-1.0"),
        "Dave": ("en_male_dave_moon_bigtts", "seed-tts-1.0"),
        "Hades": ("en_male_hades_moon_bigtts", "seed-tts-1.0"),
        "Onez": ("en_female_onez_moon_bigtts", "seed-tts-1.0"),
        "Sarah": ("en_female_sarah_mars_bigtts", "seed-tts-1.0"),
        "Dryw": ("en_male_dryw_mars_bigtts", "seed-tts-1.0"),
        
        # 日文音色
        "さとみ（智美）": ("multi_female_sophie_conversation_wvae_bigtts", "seed-tts-1.0"),
        "まさお（正男）": ("multi_male_xudong_conversation_wvae_bigtts", "seed-tts-1.0"),
        "つき（月）": ("multi_female_maomao_conversation_wvae_bigtts", "seed-tts-1.0"),
        "あけみ（朱美）": ("multi_female_gaolengyujie_moon_bigtts", "seed-tts-1.0"),
        
        # 韩文音色
        "Sweet Girl": ("ko_female_sweet_girl", "seed-tts-1.0"),
        
        # 西班牙文音色
        "Serene Woman": ("es_female_serene_woman", "seed-tts-1.0"),
        
        # 其他音色
        "Lucas": ("zh_male_M100_conversation_wvae_bigtts", "seed-tts-1.0"),
        "Sophie": ("zh_female_sophie_conversation_wvae_bigtts", "seed-tts-1.0"),
        "Daniel": ("zh_male_xudong_conversation_wvae_bigtts", "seed-tts-1.0"),
        "Diana": ("zh_female_maomao_conversation_wvae_bigtts", "seed-tts-1.0"),
        "Daisy": ("en_female_dacey_conversation_wvae_bigtts", "seed-tts-1.0"),
        "Owen": ("en_male_charlie_conversation_wvae_bigtts", "seed-tts-1.0"),
        "Luna": ("en_female_sarah_new_conversation_wvae_bigtts", "seed-tts-1.0"),
    }
    
    # 定义可用的音色名称列表（用于显示）
    VOICE_NAMES = list(VOICE_MAP.keys())
    
    # 定义可用的音频格式列表
    FORMAT_LIST = [
        "mp3", "pcm", "ogg_opus"
    ]
    
    # 定义可用的采样率列表
    SAMPLE_RATE_LIST = [
        8000, 16000, 22050, 24000, 32000, 44100, 48000
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
                "voice": (cls.VOICE_NAMES, {"default": "爽快思思（多情感）"}),
                "app_id": ("STRING", {"default": ""}),
                "access_key": ("STRING", {"default": ""}),
            },
            "optional": {
                "speed": ("FLOAT", {"default": 0.0, "min": -50.0, "max": 100.0, "step": 1.0}),
                "pitch": ("INT", {"default": 0, "min": -12, "max": 12, "step": 1}),
                "volume": ("FLOAT", {"default": 0.0, "min": -50.0, "max": 100.0, "step": 1.0}),
                "emotion": (["happy", "sad", "angry", "fear", "hate", "excited", "coldness", "neutral", "depressed", "lovey-dovey", "shy", "comfort", "tension", "tender", "storytelling", "radio", "magnetic", "advertising", "vocal-fry", "asmr", "news", "entertainment", "dialect", ""], {"default": ""}),
                "format": (cls.FORMAT_LIST, {"default": "pcm"}),
                "sample_rate": (cls.SAMPLE_RATE_LIST, {"default": 24000}),
                "channel": (cls.CHANNEL_LIST, {"default": 1}),
                "debug_output": ("BOOLEAN", {"default": False}),  # 添加调试输出选项
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")  # 添加STRING返回类型
    RETURN_NAMES = ("audio", "debug_info")  # 添加返回名称
    FUNCTION = "generate_speech"
    CATEGORY = "CC-API/Audio"
    
    def generate_speech(
        self,
        text,
        voice,
        app_id,
        access_key,
        speed=0.0,
        pitch=0,
        volume=0.0,
        emotion="",
        format="pcm",
        sample_rate=24000,
        channel=1,
        debug_output=False  # 添加调试输出参数
    ):
        """生成语音"""
        
        # 检查API密钥
        if not app_id:
            # 尝试从配置文件获取API密钥
            app_id = CCConfig().get_doubao_app_id()
            if not app_id:
                print("Error: No Doubao App ID provided")
                audio_data, debug_info = self._create_blank_audio(sample_rate)
                return (audio_data, debug_info)
        
        if not access_key:
            # 尝试从配置文件获取API密钥
            access_key = CCConfig().get_doubao_access_key()
            if not access_key:
                print("Error: No Doubao Access Key provided")
                audio_data, debug_info = self._create_blank_audio(sample_rate)
                return (audio_data, debug_info)
        
        # 根据选择的音色获取voice_type和resource_id
        if voice in self.VOICE_MAP:
            voice_type, resource_id = self.VOICE_MAP[voice]
        else:
            # 如果找不到映射，默认使用第一个音色
            voice_type, resource_id = list(self.VOICE_MAP.values())[0]
            print(f"Warning: Voice '{voice}' not found in voice map, using default voice")
        
        # 打印调试信息
        print(f"Using App ID: {app_id[:8]}... (truncated for security)")
        print(f"Using voice: {voice} (voice_type: {voice_type})")
        print(f"Using resource ID: {resource_id}")
        
        # 检查文本长度
        if len(text) > 10000:
            print("Warning: Text exceeds maximum length of 10000 characters. Truncating...")
            text = text[:10000]
        
        # 准备请求数据
        request_data = {
            "user": {
                "uid": "comfyui_user"
            },
            "req_params": {
                "text": text,
                "speaker": voice_type,  # 使用voice_type而不是显示名称
                "audio_params": {
                    "format": format,
                    "sample_rate": sample_rate,
                    "channel": channel
                }
            }
        }
        
        # 添加可选参数
        if speed != 0.0:
            request_data["req_params"]["audio_params"]["speech_rate"] = speed
            
        if pitch != 0:
            request_data["req_params"]["audio_params"]["pitch"] = pitch
            
        if volume != 0.0:
            request_data["req_params"]["audio_params"]["loudness_rate"] = volume
            
        if emotion:
            request_data["req_params"]["audio_params"]["emotion"] = emotion
        
        try:
            # 发送请求
            headers = {
                "X-Api-App-Id": app_id,
                "X-Api-Access-Key": access_key,
                "X-Api-Resource-Id": resource_id,  # 自动适配的资源ID
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                "https://openspeech.bytedance.com/api/v3/tts/unidirectional",
                headers=headers,
                json=request_data
            )
            
            # 如果启用了调试输出，直接返回原始响应内容
            if debug_output:
                debug_info = f"Status Code: {response.status_code}\n"
                debug_info += f"Response Headers: {dict(response.headers)}\n"
                debug_info += f"Response Content: {response.text}"
                audio_data, _ = self._create_blank_audio(sample_rate)
                return (audio_data, debug_info)
            
            if response.status_code == 200:
                # 根据官方示例代码，使用stream方式逐行处理响应
                try:
                    # 用于存储音频数据
                    audio_data = bytearray()
                    total_audio_size = 0
                    
                    print(f"Response headers: {response.headers}")
                    
                    # 逐行处理响应
                    for chunk in response.iter_lines(decode_unicode=True):
                        if not chunk:
                            continue
                        
                        # 解析每一行JSON数据
                        data = json.loads(chunk)
                        print(f"Received chunk: code={data.get('code', 'N/A')}, has_data={'data' in data}")
                        
                        # 检查是否有音频数据
                        if data.get("code", 0) == 0 and "data" in data and data["data"]:
                            chunk_audio = base64.b64decode(data["data"])
                            audio_size = len(chunk_audio)
                            total_audio_size += audio_size
                            audio_data.extend(chunk_audio)
                            print(f"Added audio chunk, size: {audio_size}, total: {total_audio_size}")
                            continue
                        
                        # 处理文本信息
                        if data.get("code", 0) == 0 and "sentence" in data and data["sentence"]:
                            print(f"Sentence data: {data['sentence']}")
                            continue
                        
                        # 结束标志
                        if data.get("code", 0) == 20000000:
                            print("Received end signal")
                            break
                        
                        # 错误处理
                        if data.get("code", 0) > 0:
                            print(f"Error response: {data}")
                            break
                    
                    print(f"Total audio data size: {len(audio_data)} bytes")
                    
                    # 如果没有收到音频数据，返回空白音频
                    if not audio_data:
                        print("No audio data received")
                        audio_data, debug_info = self._create_blank_audio(sample_rate)
                        return (audio_data, debug_info)
                    
                    # 将bytearray转换为bytes
                    audio_binary = bytes(audio_data)
                    print(f"Converted to bytes, length: {len(audio_binary)}")
                    
                    # 检查解码后的数据是否为空
                    if len(audio_binary) == 0:
                        print("Warning: Decoded audio binary is empty")
                        audio_data, debug_info = self._create_blank_audio(sample_rate)
                        return (audio_data, debug_info)
                    
                    # 根据请求的格式处理音频数据
                    if format == "pcm":
                        # 对于PCM格式，直接转换为WAV格式
                        try:
                            # 将PCM数据转换为numpy数组
                            # 首先尝试确定PCM数据的位深度
                            print(f"Audio binary length: {len(audio_binary)}")
                            
                            # 根据采样率和数据长度推测位深度
                            if sample_rate == 8000:
                                # 8位PCM
                                print("Processing as 8-bit PCM")
                                pcm_data = np.frombuffer(audio_binary, dtype=np.uint8)
                                # 转换为float32并归一化到[-1, 1]范围
                                pcm_data = pcm_data.astype(np.float32) / 128.0 - 1.0
                            else:
                                # 默认处理为16位PCM
                                print("Processing as 16-bit PCM")
                                # 确保数据长度是偶数（16位数据）
                                if len(audio_binary) % 2 != 0:
                                    print("Warning: Audio binary length is odd, trimming last byte")
                                    audio_binary = audio_binary[:-1]
                                
                                if len(audio_binary) >= 2:
                                    pcm_data = np.frombuffer(audio_binary, dtype=np.int16)
                                    # 转换为float32并归一化到[-1, 1]范围
                                    pcm_data = pcm_data.astype(np.float32) / 32768.0
                                else:
                                    print("Error: Audio binary data is too short for 16-bit PCM")
                                    audio_data, debug_info = self._create_blank_audio(sample_rate)
                                    return (audio_data, debug_info)
                            
                            print(f"PCM data shape: {pcm_data.shape}")
                            print(f"PCM data dtype: {pcm_data.dtype}")
                            print(f"PCM data min: {np.min(pcm_data)}, max: {np.max(pcm_data)}")
                            
                            # 验证PCM数据是否为空
                            if pcm_data.size == 0:
                                print("Warning: PCM data is empty")
                                audio_data, debug_info = self._create_blank_audio(sample_rate)
                                return (audio_data, debug_info)
                                
                            # 添加更多PCM数据信息
                            print(f"PCM data length in samples: {len(pcm_data)}")
                            duration = len(pcm_data) / sample_rate
                            print(f"Estimated audio duration: {duration:.2f} seconds")
                            
                            # 确保波形数据形状正确 [B, C, T]
                            # 根据请求参数确定声道数
                            print(f"Requested channel count: {channel}")
                            
                            if pcm_data.ndim == 1:
                                # 单声道数据
                                if channel == 1:
                                    # 请求单声道，直接使用
                                    waveform = pcm_data.reshape(1, 1, -1)
                                else:
                                    # 请求立体声，复制单声道数据到两个声道
                                    print("Converting mono to stereo by duplicating channel")
                                    stereo_data = np.tile(pcm_data, (2, 1))  # [2, T]
                                    waveform = stereo_data.reshape(1, 2, -1)  # [1, 2, T]
                            elif pcm_data.ndim == 2:
                                # 已经是二维数组，可能是 [C, T] 或 [T, C]
                                if pcm_data.shape[0] <= 2 and pcm_data.shape[0] < pcm_data.shape[1]:
                                    # 可能是 [C, T]，添加批次维度 [1, C, T]
                                    waveform = pcm_data.reshape(1, pcm_data.shape[0], -1)
                                else:
                                    # 可能是 [T, C]，需要转置并添加批次维度 [1, C, T]
                                    waveform = pcm_data.T.reshape(1, -1, pcm_data.shape[0])
                            else:
                                # 其他情况，直接添加批次和声道维度 [1, 1, T]
                                waveform = pcm_data.reshape(1, 1, -1)
                        
                            print(f"Final waveform shape: {waveform.shape}")
                            
                            # 验证波形数据
                            if waveform.size == 0:
                                print("Warning: Generated waveform is empty")
                                audio_data, debug_info = self._create_blank_audio(sample_rate)
                                return (audio_data, debug_info)
                            
                            # 验证波形数据是否包含有效音频信息
                            if np.max(np.abs(waveform)) < 1e-6:
                                print("Warning: Generated waveform contains mostly silence")
                                
                            # 添加波形数据的详细信息
                            print(f"Waveform duration: {waveform.shape[2] / sample_rate:.2f} seconds")
                            print(f"Waveform min: {np.min(waveform):.6f}, max: {np.max(waveform):.6f}")
                        
                            # 转换为PyTorch张量
                            waveform_tensor = torch.from_numpy(waveform)
                            
                            # 返回ComfyUI期望的音频格式
                            audio_data = {
                                "waveform": waveform_tensor,
                                "sample_rate": sample_rate
                            }
                            return (audio_data, "")
                        except Exception as e:
                            print(f"Error processing PCM audio data: {str(e)}")
                            audio_data, debug_info = self._create_blank_audio(sample_rate)
                            return (audio_data, debug_info)
                    else:
                        # 对于MP3或OGG格式，保存到临时文件并使用scipy读取
                        # 将音频数据保存到临时文件
                        with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as temp_file:
                            temp_file.write(audio_binary)
                            temp_file_path = temp_file.name
                        
                        print(f"Saved audio to temporary file: {temp_file_path}")
                        
                        # 读取音频文件并转换为波形数据
                        try:
                            audio_sample_rate, waveform = wavfile.read(temp_file_path)
                            print(f"Read audio file - Sample rate: {audio_sample_rate}, Waveform shape: {waveform.shape}")
                            
                            # 验证读取的波形数据是否为空
                            if waveform.size == 0:
                                print("Warning: Read waveform is empty")
                                os.unlink(temp_file_path)
                                audio_data, debug_info = self._create_blank_audio(sample_rate)
                                return (audio_data, debug_info)
                        except Exception as e:
                            print(f"Error reading audio file: {str(e)}")
                            os.unlink(temp_file_path)
                            audio_data, debug_info = self._create_blank_audio(sample_rate)
                            return (audio_data, debug_info)
                            
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
                        # 根据请求参数确定声道数
                        print(f"Requested channel count: {channel}")
                        
                        if waveform.ndim == 1:
                            # 单声道数据
                            if channel == 1:
                                # 请求单声道，直接使用
                                waveform = waveform.reshape(1, 1, -1)
                            else:
                                # 请求立体声，复制单声道数据到两个声道
                                print("Converting mono to stereo by duplicating channel")
                                stereo_data = np.tile(waveform, (2, 1))  # [2, T]
                                waveform = stereo_data.reshape(1, 2, -1)  # [1, 2, T]
                        elif waveform.ndim == 2:
                            # 已经是二维数组，可能是 [C, T] 或 [T, C]
                            if waveform.shape[0] <= 2 and waveform.shape[0] < waveform.shape[1]:
                                # 可能是 [C, T]，添加批次维度 [1, C, T]
                                waveform = waveform.reshape(1, waveform.shape[0], -1)
                            else:
                                # 可能是 [T, C]，需要转置并添加批次维度 [1, C, T]
                                waveform = waveform.T.reshape(1, -1, waveform.shape[0])
                        else:
                            # 其他情况，直接添加批次和声道维度 [1, 1, T]
                            waveform = waveform.reshape(1, 1, -1)
                        
                        print(f"Final waveform shape: {waveform.shape}")
                        
                        # 验证波形数据
                        if waveform.size == 0:
                            print("Warning: Generated waveform is empty")
                            os.unlink(temp_file_path)
                            audio_data, debug_info = self._create_blank_audio(sample_rate)
                            return (audio_data, debug_info)
                        
                        # 验证波形数据是否包含有效音频信息
                        if np.max(np.abs(waveform)) < 1e-6:
                            print("Warning: Generated waveform contains mostly silence")
                            
                        # 添加波形数据的详细信息
                        print(f"Waveform duration: {waveform.shape[2] / audio_sample_rate:.2f} seconds")
                        print(f"Waveform min: {np.min(waveform):.6f}, max: {np.max(waveform):.6f}")
                        
                        # 转换为PyTorch张量
                        waveform_tensor = torch.from_numpy(waveform)
                            
                        # 删除临时文件
                        os.unlink(temp_file_path)
                            
                        # 返回ComfyUI期望的音频格式
                        audio_data = {
                            "waveform": waveform_tensor,
                            "sample_rate": audio_sample_rate
                        }
                        return (audio_data, "")
                except Exception as e:
                    print(f"Error processing stream response: {str(e)}")
                    audio_data, debug_info = self._create_blank_audio(sample_rate)
                    return (audio_data, debug_info)
            else:
                print(f"API request failed with status {response.status_code}: {response.text}")
                # 如果响应包含JSON数据，尝试解析并显示详细错误信息
                try:
                    error_result = response.json()
                    if "header" in error_result:
                        header = error_result["header"]
                        print(f"Error code: {header.get('code', 'N/A')}")
                        print(f"Error message: {header.get('message', 'N/A')}")
                except:
                    # 如果不是JSON格式，直接显示响应内容
                    pass
                audio_data, debug_info = self._create_blank_audio(sample_rate)
                return (audio_data, debug_info)
        
        except Exception as e:
            print(f"Error generating speech: {str(e)}")
            audio_data, debug_info = self._create_blank_audio(sample_rate)
            return (audio_data, debug_info)

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
        audio_data = {
            "waveform": silence_tensor,
            "sample_rate": sample_rate
        }
        return (audio_data, "")

# 注册节点
NODE_CLASS_MAPPINGS = {
    "DoubaoTTS": DoubaoTTS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DoubaoTTS": "Doubao TTS",
}