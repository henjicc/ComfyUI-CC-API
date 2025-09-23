import os
import requests
import json
import tempfile
import base64
import numpy as np
import torch
from PIL import Image
import scipy.io.wavfile as wavfile
from .cc_utils import CCConfig


class Qwen3TTS:
    """Qwen3-TTS语音合成节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "voice": (
                    [
                        "Cherry (芊悦)",
                        "Ethan (晨煦)",
                        "Nofish (不吃鱼)",
                        "Jennifer (詹妮弗)",
                        "Ryan (甜茶)",
                        "Katerina (卡捷琳娜)",
                        "Elias (墨讲师)",
                        "Jada (上海-阿珍)",
                        "Dylan (北京-晓东)",
                        "Sunny (四川-晴儿)",
                        "Li (南京-老李)",
                        "Marcus (陕西-秦川)",
                        "Roy (闽南-阿杰)",
                        "Peter (天津-李彼得)",
                        "Rocky (粤语-阿强)",
                        "Kiki (粤语-阿清)",
                        "Eric (四川-程川)"
                    ],
                    {"default": "Cherry (芊悦)"},
                ),
                "language_type": (
                    [
                        "Auto",
                        "Chinese",
                        "English",
                        "German",
                        "Italian",
                        "Portuguese",
                        "Spanish",
                        "Japanese",
                        "Korean",
                        "French",
                        "Russian"
                    ],
                    {"default": "Auto"},
                ),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate_speech"
    CATEGORY = "CC-API/Audio"

    def get_dashscope_api_key(self, api_key=""):
        """获取DashScope API密钥"""
        # 首先检查传入的参数
        if api_key:
            return api_key
        
        # 然后检查环境变量
        if os.environ.get("DASHSCOPE_API_KEY") is not None:
            return os.environ["DASHSCOPE_API_KEY"]
        
        # 最后检查配置文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        config_path = os.path.join(parent_dir, "config.ini")
        
        if os.path.exists(config_path):
            import configparser
            config = configparser.ConfigParser()
            config.read(config_path)
            
            try:
                if "dashscope" in config and "API_KEY" in config["dashscope"]:
                    return config["dashscope"]["API_KEY"]
            except KeyError:
                pass
        
        return None

    def generate_speech(
        self,
        text,
        voice,
        language_type="Auto",
        api_key=""
    ):
        """生成语音"""
        # 获取API密钥
        api_key = self.get_dashscope_api_key(api_key)
        
        if not api_key:
            print("Error: No DashScope API key provided")
            return self._create_blank_audio()
        
        # 检查文本长度
        if len(text) > 600:
            print("Warning: Text exceeds maximum length of 600 characters. Truncating...")
            text = text[:600]
        
        # 提取voice参数（去掉括号中的描述）
        voice_param = voice.split(" (")[0]
        
        # 准备请求数据
        request_data = {
            "model": "qwen3-tts-flash",
            "input": {
                "text": text,
                "voice": voice_param,
                "language_type": language_type
            }
        }
        
        try:
            # 发送请求
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
                headers=headers,
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # 检查响应中是否包含音频URL
                if "output" in result and "audio" in result["output"] and "url" in result["output"]["audio"]:
                    audio_url = result["output"]["audio"]["url"]
                    
                    # 下载音频文件
                    audio_response = requests.get(audio_url)
                    
                    if audio_response.status_code == 200:
                        # 将音频数据保存到临时文件
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                            temp_file.write(audio_response.content)
                            temp_file_path = temp_file.name
                        
                        # 读取音频文件并转换为波形数据
                        sample_rate, waveform = wavfile.read(temp_file_path)
                        
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
                        
                        # 返回ComfyUI期望的音频格式
                        return ({
                            "waveform": waveform_tensor,
                            "sample_rate": sample_rate
                        },)
                    else:
                        print(f"Error downloading audio: {audio_response.status_code}")
                        return self._create_blank_audio()
                else:
                    print("Error: No audio URL in response")
                    return self._create_blank_audio()
            else:
                print(f"API request failed with status {response.status_code}: {response.text}")
                return self._create_blank_audio()
        
        except Exception as e:
            print(f"Error generating speech: {str(e)}")
            return self._create_blank_audio()

    def _create_blank_audio(self):
        """创建一个空白音频文件"""
        # 创建一个短暂的静音音频文件
        sample_rate = 24000
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


# 注册节点
NODE_CLASS_MAPPINGS = {
    "Qwen3TTS": Qwen3TTS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3TTS": "Qwen3-TTS",
}