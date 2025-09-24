import os
import requests
import json
import tempfile
import base64
import numpy as np
import torch
from PIL import Image
import scipy.io.wavfile as wavfile
import av
import io
import random
import hashlib
import time
import server
from .cc_utils import CCConfig


class Qwen3TTS:
    """Qwen3-TTS语音合成节点"""
    
    # 音色预览URL映射
    VOICE_PREVIEW_URLS = {
        "Cherry (芊悦)": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250211/tixcef/cherry.wav",
        "Ethan (晨煦)": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250211/emaqdp/ethan.wav",
        "Nofish (不吃鱼)": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250915/dqpkml/Nofish.wav",
        "Jennifer (詹妮弗)": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250910/lgwivy/Jennifer.wav",
        "Ryan (甜茶)": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250910/dvswuj/Ryan.wav",
        "Katerina (卡捷琳娜)": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250910/ziffsi/katerina.wav",
        "Elias (墨讲师)": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250910/rcpalp/Elias.wav",
        "Jada (上海-阿珍)": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250910/qjfmmi/Jada.wav",
        "Dylan (北京-晓东)": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250910/ultaxm/Dylan.wav",
        "Sunny (四川-晴儿)": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250910/jtrktt/Sunny.wav",
        "Li (南京-老李)": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250916/frgdes/Li.wav",
        "Marcus (陕西-秦川)": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250910/dwnnrg/Marcus.wav",
        "Roy (闽南-阿杰)": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250910/stsfsz/Roy.wav",
        "Peter (天津-李彼得)": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250910/twvnsp/Peter.wav",
        "Rocky (粤语-阿强)": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250910/kfxxgp/Rocky.wav",
        "Kiki (粤语-阿清)": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250910/qwinef/KiKi.wav",
        "Eric (四川-程川)": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250910/qhbznw/Eric.wav"
    }
    
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
                "preview_voice": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate_speech"
    OUTPUT_NODE = True
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
    
    def load_preview_audio(self, voice):
        """加载音色预览音频"""
        if voice not in self.VOICE_PREVIEW_URLS:
            return None
        
        url = self.VOICE_PREVIEW_URLS[voice]
        
        try:
            # 创建临时目录
            temp_dir = os.path.join(tempfile.gettempdir(), "comfyui_qwen3_tts")
            os.makedirs(temp_dir, exist_ok=True)
            
            # 生成唯一文件名
            file_hash = hashlib.md5(url.encode()).hexdigest()
            preview_path = os.path.join(temp_dir, f"preview_{file_hash}.wav")
            
            # 如果文件已存在，直接返回
            if os.path.exists(preview_path):
                return self._load_audio_file(preview_path)
            
            # 下载音频文件
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(preview_path, "wb") as f:
                    f.write(response.content)
                
                # 加载音频文件
                return self._load_audio_file(preview_path)
            else:
                print(f"下载预览音频失败: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"加载预览音频失败: {e}")
            return None
    
    def _load_audio_file(self, audio_path):
        """加载音频文件并返回波形数据"""
        try:
            with av.open(audio_path) as container:
                # 获取音频流
                audio_stream = next(s for s in container.streams if s.type == 'audio')
                
                # 获取采样率
                sample_rate = audio_stream.rate
                
                # 读取音频帧
                audio_frames = []
                for frame in container.decode(audio_stream):
                    # 将音频帧转换为numpy数组
                    arr = frame.to_ndarray()
                    audio_frames.append(arr)
                
                # 合并所有帧
                if audio_frames:
                    waveform = np.concatenate(audio_frames, axis=1)
                    
                    # 确保波形形状为 [B, C, T]
                    if waveform.ndim == 1:
                        # 单声道一维数组 -> [1, 1, T]
                        waveform = waveform.reshape(1, 1, -1)
                    elif waveform.ndim == 2:
                        # 二维数组 -> [1, C, T]
                        waveform = waveform.reshape(1, waveform.shape[0], waveform.shape[1])
                    
                    # 转换为torch张量
                    waveform_tensor = torch.from_numpy(waveform).float()
                    
                    return {
                        "waveform": waveform_tensor,
                        "sample_rate": sample_rate
                    }
                else:
                    return None
        except Exception as e:
            print(f"处理音频文件失败: {e}")
            return None

    def generate_speech(
        self,
        text,
        voice,
        language_type="Auto",
        api_key="",
        preview_voice=False
    ):
        """生成语音"""
        
        # 如果启用预览功能，直接返回预览音频
        if preview_voice:
            preview_audio = self.load_preview_audio(voice)
            if preview_audio:
                return (preview_audio,)
            else:
                # 如果预览加载失败，返回静音音频
                silence = self._create_blank_audio()
                return (silence,)
        
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
    
    def preview_voice(self, voice):
        """预览音色"""
        preview_audio = self.load_preview_audio(voice)
        if preview_audio:
            return (preview_audio,)
        else:
            # 如果预览加载失败，返回静音音频
            silence = self._create_blank_audio()
            return (silence,)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "Qwen3TTS": Qwen3TTS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3TTS": "Qwen3-TTS",
}

# 注册API路由
def setup_routes():
    """注册Qwen3-TTS音色预览API路由"""
    @server.PromptServer.instance.routes.post("/qwen3_tts_preview")
    async def qwen3_tts_preview(request):
        """处理Qwen3-TTS音色预览请求"""
        try:
            # 获取请求数据
            data = await request.json()
            voice = data.get("voice", "")
            
            if not voice:
                return web.json_response({"error": "Missing voice parameter"}, status=400)
            
            # 创建Qwen3TTS实例
            tts = Qwen3TTS()
            
            # 加载预览音频
            preview_audio = tts.load_preview_audio(voice)
            
            if preview_audio is None:
                return web.json_response({"error": "Failed to load preview audio"}, status=500)
            
            # 获取波形数据
            waveform = preview_audio["waveform"]
            sample_rate = preview_audio["sample_rate"]
            
            # 转换为numpy数组
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.numpy()
            
            # 确保波形形状正确 [T, C]
            if waveform.ndim == 3:
                # 从 [B, C, T] 转换为 [T, C]
                waveform = waveform[0].T  # 取第一个批次并转置
            
            # 确保数据类型是int16
            if waveform.dtype != np.int16:
                # 从float32 [-1.0, 1.0] 转换为 int16 [-32768, 32767]
                waveform = (waveform * 32767).astype(np.int16)
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file_path = temp_file.name
            
            # 保存为WAV文件
            wavfile.write(temp_file_path, sample_rate, waveform)
            
            # 读取文件并编码为Base64
            with open(temp_file_path, "rb") as audio_file:
                audio_data = audio_file.read()
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # 删除临时文件
            os.unlink(temp_file_path)
            
            # 返回Base64编码的音频数据
            return web.json_response({
                "audio": audio_base64,
                "sample_rate": sample_rate
            })
            
        except Exception as e:
            print(f"Error in qwen3_tts_preview: {str(e)}")
            return web.json_response({"error": str(e)}, status=500)

# 立即注册路由
setup_routes()