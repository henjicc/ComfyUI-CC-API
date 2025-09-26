import os
import json
import base64
import tempfile
import requests
import numpy as np
import torch
from scipy.io import wavfile
from .cc_utils import CCConfig

class DoubaoTTS_Mix:
    """豆包语音合成MIX节点 - 支持多个音色混合"""
    
    def __init__(self):
        """初始化节点，从JSON文件加载音色映射"""
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建JSON文件路径
        json_file_path = os.path.join(current_dir, 'doubao_voices.json')
        
        # 从JSON文件加载音色映射
        self.VOICE_MAP = {}
        self.VOICE_CATEGORIES = {}
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                voices = data.get('voice_categories', {})
                
                # 将嵌套的音色分类结构扁平化为VOICE_MAP
                for category, voice_dict in voices.items():
                    self.VOICE_CATEGORIES[category] = list(voice_dict.keys())
                    for voice_name, voice_info in voice_dict.items():
                        self.VOICE_MAP[voice_name] = voice_info
                        
        except FileNotFoundError:
            print(f"Warning: Could not find voice mapping file at {json_file_path}")
        except Exception as e:
            print(f"Error loading voice mapping from JSON: {str(e)}")
    
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
        # 创建节点实例以获取音色列表
        instance = cls()
        
        # 只获取1.0非多情感音色
        voice_1_0 = instance.VOICE_CATEGORIES.get("1.0", [])
        # 添加分类前缀
        voice_1_0_categorized = [f"1.0/{voice}" for voice in voice_1_0]
        
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "请输入要合成的文本"}),
                # 第一个音色
                "voice_1": (voice_1_0_categorized, {"default": voice_1_0_categorized[0] if voice_1_0_categorized else "1.0/爽快思思/Skye"}),
                "mix_factor_1": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                # 第二个音色
                "voice_2": (voice_1_0_categorized, {"default": voice_1_0_categorized[0] if voice_1_0_categorized else "1.0/爽快思思/Skye"}),
                "mix_factor_2": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                # 第三个音色（可选，可以设置为"无"）
                "voice_3": (["无"] + voice_1_0_categorized, {"default": "无"}),
                "mix_factor_3": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "speed": ("FLOAT", {"default": 0.0, "min": -50.0, "max": 100.0, "step": 1.0}),
                "pitch": ("INT", {"default": 0, "min": -12, "max": 12, "step": 1}),
                "format": (cls.FORMAT_LIST, {"default": "pcm"}),
                "sample_rate": (cls.SAMPLE_RATE_LIST, {"default": 24000}),
                "channel": (cls.CHANNEL_LIST, {"default": 1}),
                "app_id": ("STRING", {"default": "", "display_name": "APP ID"}),
                "access_key": ("STRING", {"default": "", "display_name": "Access Token"}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate_speech_mix"
    CATEGORY = "CC-API/Audio"
    
    def generate_speech_mix(
        self,
        text,
        voice_1,
        mix_factor_1,
        voice_2,
        mix_factor_2,
        voice_3,
        mix_factor_3,
        app_id,
        access_key,
        speed=0.0,
        pitch=0,
        format="pcm",
        sample_rate=24000,
        channel=1
    ):
        """生成混合音色语音"""
        
        # 检查API密钥
        if not app_id:
            # 尝试从配置文件获取API密钥
            app_id = CCConfig().get_doubao_app_id()
            if not app_id:
                raise ValueError("Error: No Doubao App ID provided. Please provide a valid App ID.")
        
        if not access_key:
            # 尝试从配置文件获取API密钥
            access_key = CCConfig().get_doubao_access_key()
            if not access_key:
                raise ValueError("Error: No Doubao Access Key provided. Please provide a valid Access Key.")
        
        # 确定使用的音色数量
        use_three_voices = voice_3 != "无"
        
        # 验证混合因子总和
        if use_three_voices:
            total_factor = mix_factor_1 + mix_factor_2 + mix_factor_3
            if abs(total_factor - 1.0) > 1e-6:
                raise ValueError(f"Error: Mix factors must sum to 1.0 when using 3 voices, but current sum is {total_factor:.3f}")
        else:
            total_factor = mix_factor_1 + mix_factor_2
            if abs(total_factor - 1.0) > 1e-6:
                raise ValueError(f"Error: Mix factors must sum to 1.0 when using 2 voices, but current sum is {total_factor:.3f}")
        
        # 从分类音色名称中提取实际音色名称
        actual_voice_1 = voice_1.split("/", 1)[1] if "/" in voice_1 else voice_1
        actual_voice_2 = voice_2.split("/", 1)[1] if "/" in voice_2 else voice_2
        actual_voice_3 = voice_3.split("/", 1)[1] if "/" in voice_3 else voice_3
        
        # 根据选择的音色获取voice_type
        if actual_voice_1 in self.VOICE_MAP:
            voice_type_1 = self.VOICE_MAP[actual_voice_1][0]
        else:
            raise ValueError(f"Error: Voice '{actual_voice_1}' not found in voice map")
            
        if actual_voice_2 in self.VOICE_MAP:
            voice_type_2 = self.VOICE_MAP[actual_voice_2][0]
        else:
            raise ValueError(f"Error: Voice '{actual_voice_2}' not found in voice map")
            
        if use_three_voices:
            if actual_voice_3 in self.VOICE_MAP:
                voice_type_3 = self.VOICE_MAP[actual_voice_3][0]
            else:
                raise ValueError(f"Error: Voice '{actual_voice_3}' not found in voice map")
        
        # 打印调试信息
        if use_three_voices:
            print(f"Using App ID: {app_id[:8]}... (truncated for security)")
            print(f"Using voices: {actual_voice_1}, {actual_voice_2}, {actual_voice_3}")
            print(f"Mix factors: {mix_factor_1}, {mix_factor_2}, {mix_factor_3}")
        else:
            print(f"Using App ID: {app_id[:8]}... (truncated for security)")
            print(f"Using voices: {actual_voice_1}, {actual_voice_2}")
            print(f"Mix factors: {mix_factor_1}, {mix_factor_2}")
        
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
                "speaker": "custom_mix_bigtts",  # MIX功能必须设置为custom_mix_bigtts
                "audio_params": {
                    "format": format,
                    "sample_rate": sample_rate,
                    "channel": channel
                }
            }
        }
        
        # 构建mix_speaker参数
        if use_three_voices:
            request_data["req_params"]["mix_speaker"] = {
                "speakers": [
                    {
                        "source_speaker": voice_type_1,
                        "mix_factor": mix_factor_1
                    },
                    {
                        "source_speaker": voice_type_2,
                        "mix_factor": mix_factor_2
                    },
                    {
                        "source_speaker": voice_type_3,
                        "mix_factor": mix_factor_3
                    }
                ]
            }
        else:
            request_data["req_params"]["mix_speaker"] = {
                "speakers": [
                    {
                        "source_speaker": voice_type_1,
                        "mix_factor": mix_factor_1
                    },
                    {
                        "source_speaker": voice_type_2,
                        "mix_factor": mix_factor_2
                    }
                ]
            }
        
        # 添加可选参数
        if speed != 0.0:
            request_data["req_params"]["audio_params"]["speech_rate"] = speed
            
        if pitch != 0:
            request_data["req_params"]["audio_params"]["pitch"] = pitch
        
        try:
            # 发送请求
            headers = {
                "X-Api-App-Id": app_id,
                "X-Api-Access-Key": access_key,
                "X-Api-Resource-Id": "seed-tts-1.0",  # MIX功能仅适用于1.0音色
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                "https://openspeech.bytedance.com/api/v3/tts/unidirectional",
                headers=headers,
                json=request_data
            )
            
            if response.status_code == 200:
                # 用于存储音频数据
                audio_data = bytearray()
                
                # 逐行处理响应
                for chunk in response.iter_lines(decode_unicode=True):
                    if not chunk:
                        continue
                    
                    # 解析每一行JSON数据
                    data = json.loads(chunk)
                    
                    # 检查是否有音频数据
                    if data.get("code", 0) == 0 and "data" in data and data["data"]:
                        chunk_audio = base64.b64decode(data["data"])
                        audio_data.extend(chunk_audio)
                        continue
                    
                    # 结束标志
                    if data.get("code", 0) == 20000000:
                        break
                    
                    # 错误处理
                    if data.get("code", 0) > 0:
                        raise ValueError(f"API Error: code={data.get('code')}, message={data.get('message', '')}")
                
                # 如果没有收到音频数据，直接报错
                if not audio_data:
                    raise ValueError("Error: No audio data received from API")
                
                # 将bytearray转换为bytes
                audio_binary = bytes(audio_data)
                
                # 检查解码后的数据是否为空
                if len(audio_binary) == 0:
                    raise ValueError("Error: Decoded audio binary is empty")
                
                # 根据请求的格式处理音频数据
                if format == "pcm":
                    # 对于PCM格式，直接转换为WAV格式
                    try:
                        # 默认处理为16位PCM
                        # 确保数据长度是偶数（16位数据）
                        if len(audio_binary) % 2 != 0:
                            print("Warning: Audio binary length is odd, trimming last byte")
                            audio_binary = audio_binary[:-1]
                        
                        if len(audio_binary) >= 2:
                            pcm_data = np.frombuffer(audio_binary, dtype=np.int16)
                            # 转换为float32并归一化到[-1, 1]范围
                            pcm_data = pcm_data.astype(np.float32) / 32768.0
                        else:
                            raise ValueError("Error: Audio binary data is too short for 16-bit PCM")
                        
                        # 确保波形数据形状正确 [B, C, T]
                        if pcm_data.ndim == 1:
                            # 单声道数据
                            if channel == 1:
                                # 请求单声道，直接使用
                                waveform = pcm_data.reshape(1, 1, -1)
                            else:
                                # 请求立体声，复制单声道数据到两个声道
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
                        
                        # 转换为PyTorch张量
                        waveform_tensor = torch.from_numpy(waveform)
                        
                        # 返回ComfyUI期望的音频格式
                        audio_data = {
                            "waveform": waveform_tensor,
                            "sample_rate": sample_rate
                        }
                        return (audio_data,)
                    except Exception as e:
                        raise ValueError(f"Error processing PCM audio data: {str(e)}")
                else:
                    # 对于MP3或OGG格式，保存到临时文件并使用scipy读取
                    # 将音频数据保存到临时文件
                    with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as temp_file:
                        temp_file.write(audio_binary)
                        temp_file_path = temp_file.name
                    
                    try:
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
                            # 单声道数据
                            if channel == 1:
                                # 请求单声道，直接使用
                                waveform = waveform.reshape(1, 1, -1)
                            else:
                                # 请求立体声，复制单声道数据到两个声道
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
                        
                        # 转换为PyTorch张量
                        waveform_tensor = torch.from_numpy(waveform)
                        
                        # 删除临时文件
                        os.unlink(temp_file_path)
                        
                        # 返回ComfyUI期望的音频格式
                        audio_data = {
                            "waveform": waveform_tensor,
                            "sample_rate": audio_sample_rate
                        }
                        return (audio_data,)
                    except Exception as e:
                        # 删除临时文件
                        if os.path.exists(temp_file_path):
                            os.unlink(temp_file_path)
                        raise ValueError(f"Error reading audio file: {str(e)}")
            else:
                raise ValueError(f"API request failed with status {response.status_code}: {response.text}")
        
        except Exception as e:
            raise ValueError(f"Error generating speech: {str(e)}")

# 注册节点
NODE_CLASS_MAPPINGS = {
    "DoubaoTTS_Mix": DoubaoTTS_Mix,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DoubaoTTS_Mix": "Doubao TTS Mix",
}