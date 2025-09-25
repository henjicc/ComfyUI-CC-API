import os
import json
import base64
import tempfile
import requests
import numpy as np
import torch
from scipy.io import wavfile
from .cc_utils import CCConfig

class DoubaoTTS:
    """豆包语音合成节点"""
    
    # 定义可用的音色列表（基于火山引擎音色列表文档）
    VOICE_LIST = [
        # 豆包语音合成模型2.0
        "vivi",  # vivi
        "zh_female_vv_uranus_bigtts",  # 豆包语音合成模型2.0
        
        # 端到端实时语音大模型-O版本服务端
        "zh_female_xiaohe_jupiter_bigtts",  # 小何
        "zh_male_yunzhou_jupiter_bigtts",   # 云舟
        "zh_male_xiaotian_jupiter_bigtts",  # 小天
        
        # 豆包语音合成模型1.0音色
        "zh_male_lengkugege_emo_v2_mars_bigtts",  # 冷酷哥哥（多情感）
        "zh_female_tianxinxiaomei_emo_v2_mars_bigtts",  # 甜心小美（多情感）
        "zh_female_gaolengyujie_emo_v2_mars_bigtts",  # 高冷御姐（多情感）
        "zh_male_aojiaobazong_emo_v2_mars_bigtts",  # 傲娇霸总（多情感）
        "zh_male_guangzhoudege_emo_mars_bigtts",  # 广州德哥（多情感）
        "zh_male_jingqiangkanye_emo_mars_bigtts",  # 京腔侃爷（多情感）
        "zh_female_linjuayi_emo_v2_mars_bigtts",  # 邻居阿姨（多情感）
        "zh_male_yourougongzi_emo_v2_mars_bigtts",  # 优柔公子（多情感）
        "zh_male_ruyayichen_emo_v2_mars_bigtts",  # 儒雅男友（多情感）
        "zh_male_junlangnanyou_emo_v2_mars_bigtts",  # 俊朗男友（多情感）
        "zh_male_beijingxiaoye_emo_v2_mars_bigtts",  # 北京小爷（多情感）
        "zh_female_roumeinvyou_emo_v2_mars_bigtts",  # 柔美女友（多情感）
        "zh_male_yangguangqingnian_emo_v2_mars_bigtts",  # 阳光青年（多情感）
        "zh_female_meilinvyou_emo_v2_mars_bigtts",  # 魅力女友（多情感）
        "zh_female_shuangkuaisisi_emo_v2_mars_bigtts",  # 爽快思思（多情感）
        "en_female_candice_emo_v2_mars_bigtts",  # Candice
        "en_female_skye_emo_v2_mars_bigtts",  # Serena
        "en_male_glen_emo_v2_mars_bigtts",  # Glen
        "en_male_sylus_emo_v2_mars_bigtts",  # Sylus
        "en_male_corey_emo_v2_mars_bigtts",  # Corey
        "en_female_nadia_tips_emo_v2_mars_bigtts",  # Nadia
        "zh_female_yingyujiaoyu_mars_bigtts",  # Tina老师
        "zh_female_vv_mars_bigtts",  # Vivi
        "zh_female_qinqienvsheng_moon_bigtts",  # 亲切女声
        "zh_male_qingyiyuxuan_mars_bigtts",  # 阳光阿辰
        "zh_male_xudong_conversation_wvae_bigtts",  # 快乐小东
        "zh_male_yangguangqingnian_moon_bigtts",  # 阳光青年
        "zh_female_tianmeitaozi_mars_bigtts",  # 甜美桃子
        "zh_female_qingxinnvsheng_mars_bigtts",  # 清新女声
        "zh_female_zhixingnvsheng_mars_bigtts",  # 知性女声
        "zh_male_qingshuangnanda_mars_bigtts",  # 清爽男大
        "zh_female_linjianvhai_moon_bigtts",  # 邻家女孩
        "zh_male_yuanboxiaoshu_moon_bigtts",  # 渊博小叔
        "zh_female_tianmeixiaoyuan_moon_bigtts",  # 甜美小源
        "zh_female_qingchezizi_moon_bigtts",  # 清澈梓梓
        "zh_male_jieshuoxiaoming_moon_bigtts",  # 解说小明
        "zh_female_kailangjiejie_moon_bigtts",  # 开朗姐姐
        "zh_male_linjiananhai_moon_bigtts",  # 邻家男孩
        "zh_female_tianmeiyueyue_moon_bigtts",  # 甜美悦悦
        "zh_female_xinlingjitang_moon_bigtts",  # 心灵鸡汤
        "zh_male_wenrouxiaoge_mars_bigtts",  # 温柔小哥
        "zh_female_cancan_mars_bigtts",  # 灿灿/Shiny
        "zh_female_shuangkuaisisi_moon_bigtts",  # 爽快思思/Skye
        "zh_male_wennuanahu_moon_bigtts",  # 温暖阿虎/Alvin
        "zh_male_shaonianzixin_moon_bigtts",  # 少年梓辛/Brayan
        "zh_male_hupunan_mars_bigtts",  # 沪普男
        "zh_male_lubanqihao_mars_bigtts",  # 鲁班七号
        "zh_female_yangmi_mars_bigtts",  # 林潇
        "zh_female_linzhiling_mars_bigtts",  # 玲玲姐姐
        "zh_female_jiyejizi2_mars_bigtts",  # 春日部姐姐
        "zh_male_tangseng_mars_bigtts",  # 唐僧
        "zh_male_zhuangzhou_mars_bigtts",  # 庄周
        "zh_male_zhubajie_mars_bigtts",  # 猪八戒
        "zh_female_ganmaodianyin_mars_bigtts",  # 感冒电音姐姐
        "zh_female_naying_mars_bigtts",  # 直率英子
        "zh_female_leidian_mars_bigtts",  # 女雷神
        "zh_male_yuzhouzixuan_moon_bigtts",  # 豫州子轩
        "zh_female_daimengchuanmei_moon_bigtts",  # 呆萌川妹
        "zh_male_guangxiyuanzhou_moon_bigtts",  # 广西远舟
        "zh_male_zhoujielun_emo_v2_mars_bigtts",  # 双节棍小哥
        "zh_female_wanwanxiaohe_moon_bigtts",  # 湾湾小何
        "zh_female_wanqudashu_moon_bigtts",  # 湾区大叔
        "zh_male_guozhoudege_moon_bigtts",  # 广州德哥
        "zh_male_haoyuxiaoge_moon_bigtts",  # 浩宇小哥
        "zh_male_beijingxiaoye_moon_bigtts",  # 北京小爷
        "zh_male_jingqiangkanye_moon_bigtts",  # 京腔侃爷/Harmony
        "zh_female_meituojieer_moon_bigtts",  # 妹坨洁儿
        "zh_female_gaolengyujie_moon_bigtts",  # 高冷御姐
        "zh_male_aojiaobazong_moon_bigtts",  # 傲娇霸总
        "zh_female_meilinvyou_moon_bigtts",  # 魅力女友
        "zh_male_shenyeboke_moon_bigtts",  # 深夜播客
        "zh_female_sajiaonvyou_moon_bigtts",  # 柔美女友
        "zh_female_yuanqinvyou_moon_bigtts",  # 撒娇学妹
        "zh_male_dongfanghaoran_moon_bigtts",  # 东方浩然
        "zh_male_M100_conversation_wvae_bigtts",  # 悠悠君子
        "zh_female_maomao_conversation_wvae_bigtts",  # 文静毛毛
        "zh_female_wenrouxiaoya_moon_bigtts",  # 温柔小雅
        "zh_male_tiancaitongsheng_mars_bigtts",  # 天才童声
        "zh_male_sunwukong_mars_bigtts",  # 猴哥
        "zh_male_xionger_mars_bigtts",  # 熊二
        "zh_female_peiqi_mars_bigtts",  # 佩奇猪
        "zh_female_wuzetian_mars_bigtts",  # 武则天
        "zh_female_gujie_mars_bigtts",  # 顾姐
        "zh_female_yingtaowanzi_mars_bigtts",  # 樱桃丸子
        "zh_male_chunhui_mars_bigtts",  # 广告解说
        "zh_female_shaoergushi_mars_bigtts",  # 少儿故事
        "zh_male_silang_mars_bigtts",  # 四郎
        "zh_female_qiaopinvsheng_mars_bigtts",  # 俏皮女声
        "zh_male_lanxiaoyang_mars_bigtts",  # 懒音绵宝
        "zh_male_dongmanhaimian_mars_bigtts",  # 亮嗓萌仔
        "zh_male_jieshuonansheng_mars_bigtts",  # 磁性解说男声/Morgan
        "zh_female_jitangmeimei_mars_bigtts",  # 鸡汤妹妹/Hope
        "zh_female_tiexinnvsheng_mars_bigtts",  # 贴心女声/Candy
        "zh_female_mengyatou_mars_bigtts",  # 萌丫头/Cutey
        "zh_male_ruyaqingnian_mars_bigtts",  # 儒雅青年
        "zh_male_baqiqingshu_mars_bigtts",  # 霸气青叔
        "zh_male_qingcang_mars_bigtts",  # 擎苍
        "zh_male_yangguangqingnian_mars_bigtts",  # 活力小哥
        "zh_female_gufengshaoyu_mars_bigtts",  # 古风少御
        "zh_female_wenroushunv_mars_bigtts",  # 温柔淑女
        "zh_male_fanjuanqingnian_mars_bigtts",  # 反卷青年
        
        # 英文音色
        "en_male_adam_mars_bigtts",  # Adam
        "en_female_amanda_mars_bigtts",  # Amanda
        "en_male_jackson_mars_bigtts",  # Jackson
        "en_female_emily_mars_bigtts",  # Emily
        "en_male_smith_mars_bigtts",  # Smith
        "en_female_anna_mars_bigtts",  # Anna
        "en_female_daisy_moon_bigtts",  # Delicate Girl
        "en_male_dave_moon_bigtts",  # Dave
        "en_male_hades_moon_bigtts",  # Hades
        "en_female_onez_moon_bigtts",  # Onez
        "en_female_sarah_mars_bigtts",  # Sarah
        "en_male_dryw_mars_bigtts",  # Dryw
        
        # 日文音色
        "multi_female_sophie_conversation_wvae_bigtts",  # さとみ（智美）
        "multi_male_xudong_conversation_wvae_bigtts",  # まさお（正男）
        "multi_female_maomao_conversation_wvae_bigtts",  # つき（月）
        "multi_female_gaolengyujie_moon_bigtts",  # あけみ（朱美）
        
        # 韩文音色
        "ko_female_sweet_girl",  # Sweet Girl
        
        # 西班牙文音色
        "es_female_serene_woman",  # Serene Woman
        
        # 其他音色
        "zh_male_M100_conversation_wvae_bigtts",  # Lucas
        "zh_female_sophie_conversation_wvae_bigtts",  # Sophie
        "zh_male_xudong_conversation_wvae_bigtts",  # Daniel
        "zh_female_maomao_conversation_wvae_bigtts",  # Diana
        "en_female_dacey_conversation_wvae_bigtts",  # Daisy
        "en_male_charlie_conversation_wvae_bigtts",  # Owen
        "en_female_sarah_new_conversation_wvae_bigtts",  # Luna
    ]
    
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
                "voice": (cls.VOICE_LIST, {"default": "zh_female_shuangkuaisisi_emo_v2_mars_bigtts"}),
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
                "resource_id": (["seed-tts-2.0", "seed-tts-1.0", "seed-tts-1.0-concurr", "volc.megatts.default", "volc.megatts.concurr"], {"default": "seed-tts-2.0"}),
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
        resource_id="seed-tts-2.0",
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
        
        # 打印调试信息
        print(f"Using App ID: {app_id[:8]}... (truncated for security)")
        print(f"Using voice: {voice}")
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
                "speaker": voice,
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
                "X-Api-Resource-Id": resource_id,  # 使用传入的资源ID
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
                # 尝试解析响应
                try:
                    # 首先检查响应内容是否为空
                    if not response.text:
                        print("Error: Empty response from API")
                        audio_data, debug_info = self._create_blank_audio(sample_rate)
                        return (audio_data, debug_info)
                    
                    # 处理可能包含多个JSON对象的响应（火山引擎API特殊情况）
                    response_text = response.text.strip()
                    
                    # 尝试解析JSON，如果失败则打印原始响应内容
                    try:
                        result = response.json()
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON response: {str(e)}")
                        print(f"Response content (first 500 chars): {response_text[:500]}")
                        
                        # 特殊处理：火山引擎API可能返回多个JSON对象
                        # 尝试分割响应并解析最后一个包含音频数据的JSON对象
                        json_objects = []
                        start = 0
                        while start < len(response_text):
                            # 查找下一个 { 的位置
                            obj_start = response_text.find('{', start)
                            if obj_start == -1:
                                break
                            
                            # 尝试找到匹配的 }
                            brace_count = 0
                            obj_end = obj_start
                            for i in range(obj_start, len(response_text)):
                                if response_text[i] == '{':
                                    brace_count += 1
                                elif response_text[i] == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        obj_end = i + 1
                                        break
                            
                            if brace_count == 0:
                                # 提取JSON对象
                                json_str = response_text[obj_start:obj_end]
                                try:
                                    json_obj = json.loads(json_str)
                                    json_objects.append(json_obj)
                                    start = obj_end
                                except Exception as e:
                                    print(f"Error parsing individual JSON object: {str(e)}")
                                    print(f"JSON string: {json_str[:100]}...")
                                    start = obj_start + 1
                                    break
                            else:
                                break
                        
                        # 如果找到了多个JSON对象，收集所有包含音频数据的片段
                        if json_objects:
                            audio_fragments = []
                            text_fragments = []
                            
                            # 收集所有包含非空data字段的JSON对象（排除code为20000000的对象）
                            for obj in json_objects:
                                if "data" in obj and obj["data"] is not None and obj.get("code") != 20000000:
                                    audio_fragments.append(obj["data"])
                                elif "sentence" in obj and obj.get("code") != 20000000:
                                    text_fragments.append(obj)
                            
                            print(f"Parsed {len(json_objects)} JSON objects from response")
                            print(f"Found {len(audio_fragments)} audio fragments")
                            print(f"Found {len(text_fragments)} text fragments")
                            
                            # 合并所有音频片段
                            if audio_fragments:
                                # 将所有音频片段连接起来
                                combined_audio_data = "".join(audio_fragments)
                                print(f"Combined audio data length: {len(combined_audio_data)}")
                                
                                # 创建一个包含合并音频数据的结果对象
                                result = {
                                    "code": 0,
                                    "message": "",
                                    "data": combined_audio_data
                                }
                                print(f"Using combined audio data with {len(audio_fragments)} fragments")
                            else:
                                # 如果没有找到音频片段，使用第一个对象
                                result = json_objects[0] if json_objects else {}
                                print("No audio fragments found, using first JSON object")
                        else:
                            # 尝试手动解析JSON，处理可能的额外数据
                            try:
                                # 查找第一个 { 和最后一个 } 之间的内容
                                start = response_text.find('{')
                                end = response_text.rfind('}') + 1
                                if start != -1 and end > start:
                                    json_str = response_text[start:end]
                                    result = json.loads(json_str)
                                else:
                                    raise ValueError("No valid JSON object found in response")
                            except Exception as e2:
                                print(f"Failed to manually parse JSON: {str(e2)}")
                                audio_data, debug_info = self._create_blank_audio(sample_rate)
                                return (audio_data, debug_info)
                except Exception as e:
                    print(f"Error parsing JSON response: {str(e)}")
                    print(f"Response content: {response.text[:200]}...")  # 只打印前200个字符
                    audio_data, debug_info = self._create_blank_audio(sample_rate)
                    return (audio_data, debug_info)
                
                # 检查响应中是否包含音频数据
                print(f"Selected result object code: {result.get('code', 'N/A')}")
                if "data" in result and result["data"]:
                    audio_base64 = result["data"]
                    print(f"Audio data length: {len(audio_base64) if audio_base64 else 0}")
                    
                    # 将base64数据转换为二进制
                    try:
                        audio_binary = base64.b64decode(audio_base64)
                        print(f"Decoded audio binary length: {len(audio_binary)}")
                    except Exception as e:
                        print(f"Error decoding base64 audio data: {str(e)}")
                        audio_data, debug_info = self._create_blank_audio(sample_rate)
                        return (audio_data, debug_info)
                    
                    # 根据请求的格式处理音频数据
                    if format == "pcm":
                        # 对于PCM格式，直接转换为WAV格式
                        try:
                            # 创建WAV文件头
                            import wave
                            import struct
                            
                            # 将PCM数据转换为numpy数组
                            if sample_rate == 8000:
                                # 8位PCM
                                pcm_data = np.frombuffer(audio_binary, dtype=np.uint8)
                                # 转换为float32并归一化到[-1, 1]范围
                                pcm_data = pcm_data.astype(np.float32) / 128.0 - 1.0
                            else:
                                # 16位PCM
                                pcm_data = np.frombuffer(audio_binary, dtype=np.int16)
                                # 转换为float32并归一化到[-1, 1]范围
                                pcm_data = pcm_data.astype(np.float32) / 32768.0
                            
                            print(f"PCM data shape: {pcm_data.shape}")
                            print(f"PCM data dtype: {pcm_data.dtype}")
                            
                            # 确保波形数据形状正确 [B, C, T]
                            if pcm_data.ndim == 1:
                                # 单声道，添加批次和声道维度 [1, 1, T]
                                waveform = pcm_data.reshape(1, 1, -1)
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
                            
                            print(f"Waveform shape: {waveform.shape}")
                            
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
                            import traceback
                            traceback.print_exc()
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
                        
                        print(f"Final waveform shape: {waveform.shape}")
                        
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
                else:
                    print("Error: No audio data in response")
                    print(f"Response content: {result}")
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