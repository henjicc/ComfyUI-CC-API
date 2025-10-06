import os
import json
import tempfile
import requests
import torch
import numpy as np
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

class MiniMaxVoiceClone:
    """MiniMax 声音克隆节点"""
    
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
                "model": (["speech-2.5-hd-preview", "speech-2.5-turbo-preview", "speech-02-hd", "speech-02-turbo", "speech-01-hd", "speech-01-turbo"], {"default": "speech-2.5-hd-preview", "tooltip": "指定合成试听音频使用的语音模型"}),
                "need_noise_reduction": ("BOOLEAN", {"default": False, "tooltip": "音频复刻参数，表示是否开启降噪"}),
                "need_volume_normalization": ("BOOLEAN", {"default": False, "tooltip": "音频复刻参数，是否开启音量归一化"}),
                "api_key": ("STRING", {"default": "", "tooltip": "MiniMax API的访问密钥。如果未提供，将使用配置文件中的密钥。"}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("demo_audio", "cloned_voice_id")
    FUNCTION = "clone_voice"
    CATEGORY = "CC-API/Audio"
    
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
        if not api_key:
            # 尝试从配置文件获取API密钥
            api_key = CCConfig().get_minimax_key()
            if not api_key:
                raise ValueError("Error: No MiniMax API key provided")
                return (self._create_blank_audio(), "")
        
        try:
            # 1. 上传待克隆音频文件
            clone_file_id = self._upload_audio_file(clone_audio, api_key, "voice_clone")
            if not clone_file_id:
                raise ValueError("Error: Failed to upload clone audio file")
                return (self._create_blank_audio(), "")
            
            # 2. 如果提供了示例音频，上传示例音频文件
            prompt_file_id = None
            if prompt_audio is not None:
                prompt_file_id = self._upload_audio_file(prompt_audio, api_key, "prompt_audio")
                if not prompt_file_id:
                    # print("Warning: Failed to upload prompt audio file, continuing without it")
                    pass
            
            # 3. 调用音色克隆接口
            cloned_voice_id, demo_audio_url = self._call_voice_clone_api(
                clone_file_id,
                voice_id,
                prompt_file_id,
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
        """上传音频文件到MiniMax"""
        temp_filename = None
        try:
            # 使用新的音频处理工具处理音频
            temp_filename, file_size = process_audio_for_minimax(audio_data)
            
            # 上传文件
            url = "https://api.minimaxi.com/v1/files/upload"
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
                    file_id = result.get("file", {}).get("file_id")
                    return file_id
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
        file_id,
        voice_id,
        prompt_file_id=None,
        prompt_text="",
        test_text="",
        model="speech-2.5-hd-preview",
        api_key="",
        need_noise_reduction=False,
        need_volume_normalization=False
    ):
        """调用音色克隆接口"""
        try:
            url = "https://api.minimaxi.com/v1/voice_clone"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # 构建请求数据
            payload = {
                "file_id": file_id,
                "voice_id": voice_id,
                "need_noise_reduction": need_noise_reduction,
                "need_volume_normalization": need_volume_normalization,
                "aigc_watermark": False
            }
            
            # 如果提供了示例音频和文本
            if prompt_file_id and prompt_text:
                payload["clone_prompt"] = {
                    "prompt_audio": prompt_file_id,
                    "prompt_text": prompt_text
                }
            
            # 如果提供了试听文本和模型
            if test_text and model:
                payload["text"] = test_text
                payload["model"] = model
            
            # 打印调试信息
            print(f"Voice clone API request payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
            
            response = requests.post(url, headers=headers, json=payload)
            
            print(f"Voice clone API response status: {response.status_code}")
            print(f"Voice clone API response content: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Voice clone API response JSON: {json.dumps(result, indent=2, ensure_ascii=False)}")
                
                if result.get("base_resp", {}).get("status_code", -1) == 0:
                    # 克隆成功，返回音色ID和试听音频URL
                    demo_audio_url = result.get("demo_audio", "")
                    return voice_id, demo_audio_url
                else:
                    status_msg = result.get("base_resp", {}).get("status_msg", "Unknown error")
                    raise ValueError(f"Voice clone failed: {status_msg}")
            else:
                raise ValueError(f"Voice clone failed with status {response.status_code}: {response.text}")
                
        except Exception as e:
            raise ValueError(f"Error calling voice clone API: {str(e)}")
    
    def _download_audio(self, audio_url):
        """下载音频文件并转换为ComfyUI格式"""
        try:
            response = requests.get(audio_url)
            if response.status_code == 200:
                # 保存为临时文件
                with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as temp_file:
                    temp_filename = temp_file.name
                    temp_file.write(response.content)
                
                # 尝试读取音频文件
                waveform, sample_rate = None, None
                
                # 首先尝试使用soundfile读取
                if HAS_SOUNDFILE:
                    # 直接使用已导入的sf模块
                    waveform, sample_rate = sf.read(temp_filename, dtype='float32')
                
                # 如果soundfile不可用或失败，尝试使用pydub
                if waveform is None and HAS_PYDUB:
                    # 直接使用已导入的AudioSegment
                    audio = AudioSegment.from_file(temp_filename)
                    # 转换为numpy数组
                    waveform = np.array(audio.get_array_of_samples(), dtype=np.float32)
                    # 归一化到[-1, 1]范围
                    waveform = waveform / (2 ** (audio.sample_width * 8 - 1))
                    sample_rate = audio.frame_rate
                    
                    # 如果是立体声，转换为单声道
                    if audio.channels > 1:
                        waveform = waveform.reshape((-1, audio.channels)).mean(axis=1)
                
                # 如果以上方法都失败，尝试使用scipy.io.wavfile
                if waveform is None:
                    sample_rate, waveform = wavfile.read(temp_filename)
                    # 转换为float32格式并归一化到[-1, 1]范围
                    if waveform.dtype != np.float32:
                        if np.issubdtype(waveform.dtype, np.integer):
                            max_val = np.iinfo(waveform.dtype).max
                            waveform = waveform.astype(np.float32) / max_val
                        else:
                            waveform = waveform.astype(np.float32)
                
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
            else:
                raise ValueError(f"Failed to download audio with status {response.status_code}")
        except Exception as e:
            raise ValueError(f"Error downloading audio: {str(e)}")
    
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
        return {
            "waveform": silence_tensor,
            "sample_rate": sample_rate
        }


# 注册节点
NODE_CLASS_MAPPINGS = {
    "MiniMaxVoiceClone": MiniMaxVoiceClone,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxVoiceClone": "MiniMax Voice Clone",
}