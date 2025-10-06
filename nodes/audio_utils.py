import numpy as np
import torch
import tempfile
import os

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

from scipy.io import wavfile


class AudioProcessor:
    """音频处理工具类"""
    
    @staticmethod
    def validate_audio_data(audio_data):
        """验证音频数据"""
        if audio_data is None:
            raise ValueError("Audio data is None")
        
        if not isinstance(audio_data, dict):
            raise ValueError(f"Audio data must be a dict, got {type(audio_data)}")
        
        if "waveform" not in audio_data:
            raise ValueError("Audio data missing 'waveform' key")
        
        if "sample_rate" not in audio_data:
            raise ValueError("Audio data missing 'sample_rate' key")
        
        waveform = audio_data["waveform"]
        sample_rate = audio_data["sample_rate"]
        
        if sample_rate is None or sample_rate <= 0:
            raise ValueError(f"Invalid sample rate: {sample_rate}")
        
        return waveform, sample_rate
    
    @staticmethod
    def convert_to_numpy(waveform):
        """将波形数据转换为numpy数组"""
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        elif not isinstance(waveform, np.ndarray):
            waveform = np.array(waveform)
        
        return waveform
    
    @staticmethod
    def normalize_waveform(waveform):
        """标准化波形数据到[-1, 1]范围"""
        if waveform.dtype in [np.float32, np.float64]:
            # 已经是浮点数，确保在[-1, 1]范围内
            waveform = np.clip(waveform, -1.0, 1.0)
        elif np.issubdtype(waveform.dtype, np.integer):
            # 整数类型，需要归一化
            max_val = np.iinfo(waveform.dtype).max
            min_val = np.iinfo(waveform.dtype).min
            waveform = waveform.astype(np.float32)
            waveform = 2 * (waveform - min_val) / (max_val - min_val) - 1.0
        else:
            # 其他类型，转换为浮点数并归一化
            waveform = waveform.astype(np.float32)
            if np.abs(waveform).max() > 1.0:
                waveform = waveform / np.abs(waveform).max()
        
        return waveform
    
    @staticmethod
    def reshape_waveform(waveform):
        """重新整形波形数据为一维数组"""
        # 打印原始形状信息用于调试
        print(f"Original waveform shape: {waveform.shape}, dtype: {waveform.dtype}")
        
        # 处理不同的维度
        if waveform.ndim == 1:
            # 已经是一维数组
            pass
        elif waveform.ndim == 2:
            # 可能是[T, C]或[C, T]格式
            if waveform.shape[0] <= 2 and waveform.shape[0] < waveform.shape[1]:
                # 可能是[C, T]格式（声道数较少且小于时间步数）
                print("Transposing from [C, T] to [T, C]")
                waveform = waveform.T
            # 现在是[T, C]格式，转换为单声道
            if waveform.shape[1] > 1:
                print("Converting stereo to mono")
                waveform = np.mean(waveform, axis=1)
            else:
                waveform = waveform[:, 0]
        elif waveform.ndim == 3:
            # 可能是[B, C, T]格式
            print("Squeezing batch dimension")
            waveform = waveform.squeeze(0)  # 移除批次维度
            if waveform.ndim == 2:
                if waveform.shape[0] <= 2 and waveform.shape[0] < waveform.shape[1]:
                    # 可能是[C, T]格式
                    print("Transposing from [C, T] to [T, C]")
                    waveform = waveform.T
                # 现在是[T, C]格式，转换为单声道
                if waveform.shape[1] > 1:
                    print("Converting stereo to mono")
                    waveform = np.mean(waveform, axis=1)
                else:
                    waveform = waveform[:, 0]
        else:
            raise ValueError(f"Unsupported waveform dimensions: {waveform.ndim}")
        
        print(f"Final waveform shape: {waveform.shape}")
        return waveform
    
    @staticmethod
    def check_duration(waveform, sample_rate):
        """检查音频时长"""
        if waveform.size == 0:
            raise ValueError("Waveform is empty")
        
        duration = len(waveform) / sample_rate
        print(f"Audio duration: {duration:.2f} seconds")
        
        if duration < 10.0:
            raise ValueError(f"Audio duration too short: {duration:.2f} seconds. Minimum required: 10 seconds")
        if duration > 300.0:  # 5分钟 = 300秒
            raise ValueError(f"Audio duration too long: {duration:.2f} seconds. Maximum allowed: 5 minutes")
        
        return duration
    
    @staticmethod
    def convert_to_int16(waveform):
        """将波形数据转换为16位整数"""
        # 确保在[-1, 1]范围内
        waveform = np.clip(waveform, -1.0, 1.0)
        # 转换为16位整数
        audio_int16 = (waveform * 32767).astype(np.int16)
        return audio_int16
    
    @staticmethod
    def save_wav_file(waveform, sample_rate, temp_filename):
        """保存为WAV文件"""
        # 转换为16位整数
        audio_int16 = AudioProcessor.convert_to_int16(waveform)
        
        # 使用soundfile保存WAV文件，确保格式标准化
        if HAS_SOUNDFILE:
            sf.write(temp_filename, audio_int16, sample_rate, 
                     subtype='PCM_16', endian='LITTLE', format='WAV')
        else:
            # 如果soundfile不可用，使用wavfile但确保参数正确
            wavfile.write(temp_filename, sample_rate, audio_int16)
        
        # 验证文件是否成功创建且不为空
        if not os.path.exists(temp_filename):
            raise RuntimeError("Failed to create temporary audio file")
        
        file_size = os.path.getsize(temp_filename)
        if file_size == 0:
            raise RuntimeError("Created temporary audio file is empty")
        
        print(f"Audio file saved: {temp_filename}, size: {file_size} bytes")
        return file_size


def process_audio_for_minimax(audio_data):
    """处理音频以供MiniMax API使用"""
    # 1. 验证音频数据
    waveform, sample_rate = AudioProcessor.validate_audio_data(audio_data)
    
    # 2. 转换为numpy数组
    waveform = AudioProcessor.convert_to_numpy(waveform)
    
    # 3. 标准化波形数据
    waveform = AudioProcessor.normalize_waveform(waveform)
    
    # 4. 重新整形波形数据
    waveform = AudioProcessor.reshape_waveform(waveform)
    
    # 5. 检查时长
    duration = AudioProcessor.check_duration(waveform, sample_rate)
    
    # 6. 创建临时文件
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_filename = temp_file.name
    
    try:
        # 7. 保存为WAV文件
        file_size = AudioProcessor.save_wav_file(waveform, sample_rate, temp_filename)
        return temp_filename, file_size
    except Exception as e:
        # 确保即使出错也删除临时文件
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)
        raise e