import os
import requests
import time
import io
import base64
import tempfile
from typing import List, Optional, Union
from .cc_utils import CCConfig, ImageUtils, ResultProcessor

# 尝试导入ComfyUI的视频处理模块
try:
    from comfy_api.input_impl import VideoFromFile
    from comfy_api.input import VideoInput
    HAS_COMFY_VIDEO = True
except ImportError:
    HAS_COMFY_VIDEO = False
    VideoInput = object
    VideoFromFile = None  # 定义为None以避免未绑定变量错误

class SeedancePPIOImg2VideoNode:
    """Seedance 图生视频节点 (派欧云)"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # 输入图像
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model_version": (["lite", "pro"], {"default": "lite"}),  # 模型版本
                "resolution": (["480p", "720p", "1080p"], {"default": "1080p"}),
                "duration": ([5, 10], {"default": 5}),
                "camera_fixed": ("BOOLEAN", {"default": False, "label_on": "启用", "label_off": "禁用"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "api_key": ("STRING", {"default": ""}),
            },
            "optional": {
                "last_image": ("IMAGE",),  # 结束图像 (仅Lite版本支持)
            }
        }

    RETURN_TYPES = ("VIDEO",) if HAS_COMFY_VIDEO else ("STRING",)
    RETURN_NAMES = ("video",) if HAS_COMFY_VIDEO else ("video_url",)
    FUNCTION = "generate_video"
    CATEGORY = "CC-API/Video"
    OUTPUT_NODE = False

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

    def tensor_to_base64(self, image_tensor):
        """将ComfyUI图像张量转换为base64字符串"""
        try:
            # 转换为PIL图像
            pil_image = ImageUtils.tensor_to_pil(image_tensor)
            if pil_image:
                # 转换为base64
                buffered = io.BytesIO()
                pil_image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                return f"data:image/jpeg;base64,{img_str}"
        except Exception as e:
            print(f"Error converting tensor to base64: {str(e)}")
        return None

    def get_image_aspect_ratio(self, image_tensor):
        """计算图像的宽高比并匹配最接近的预设值"""
        try:
            # 转换为PIL图像以获取尺寸
            pil_image = ImageUtils.tensor_to_pil(image_tensor)
            if pil_image:
                width, height = pil_image.width, pil_image.height
                ratio = width / height
                
                # 预设的宽高比选项
                aspect_ratios = {
                    "21:9": 21/9,
                    "16:9": 16/9,
                    "4:3": 4/3,
                    "1:1": 1/1,
                    "3:4": 3/4,
                    "9:16": 9/16,
                    "9:21": 9/21
                }
                
                # 找到最接近的宽高比
                closest_ratio = min(aspect_ratios.keys(), key=lambda x: abs(aspect_ratios[x] - ratio))
                return closest_ratio
            else:
                # 如果无法获取图像尺寸，返回默认值
                return "16:9"
        except Exception as e:
            print(f"Error calculating aspect ratio: {str(e)}")
            # 出现错误时返回默认值
            return "16:9"

    def call_seedance_img2video_api(self, api_key, image, prompt, model_version, resolution, aspect_ratio, 
                                     duration, camera_fixed, seed, last_image=None):
        """调用Seedance图生视频API"""
        try:
            # 准备请求数据
            payload = {
                "prompt": prompt,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "duration": duration,
                "camera_fixed": camera_fixed
            }
            
            # 添加图像
            payload["image"] = image
            
            # 添加结束图像 (仅Lite版本支持)
            if last_image and model_version == "lite":
                payload["last_image"] = last_image
            
            # 添加种子
            if seed != -1:
                payload["seed"] = seed
            
            # 确定API端点
            if model_version == "lite":
                api_endpoint = "https://api.ppinfra.com/v3/async/seedance-v1-lite-i2v"
            else:  # pro
                api_endpoint = "https://api.ppinfra.com/v3/async/seedance-v1-pro-i2v"
            
            # 发送请求
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            response = requests.post(
                api_endpoint,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if "task_id" in result:
                    return result["task_id"]
                else:
                    raise Exception(f"API response missing task_id: {result}")
            else:
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")
                
        except Exception as e:
            raise Exception(f"Error calling Seedance Img2Video API: {str(e)}")

    def query_task_result(self, api_key, task_id):
        """查询任务结果"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            params = {
                "task_id": task_id
            }
            
            response = requests.get(
                "https://api.ppinfra.com/v3/async/task-result",
                headers=headers,
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Query task result failed with status {response.status_code}: {response.text}")
                
        except Exception as e:
            raise Exception(f"Error querying task result: {str(e)}")

    def poll_task_result(self, api_key, task_id, poll_interval=5, max_attempts=120):
        """轮询任务结果直到完成或失败"""
        for attempt in range(max_attempts):
            try:
                # 简化日志输出，只显示轮询尝试次数
                print(f"Polling attempt {attempt + 1}/{max_attempts}")
                
                result = self.query_task_result(api_key, task_id)
                
                if "task" in result:
                    task_status = result["task"]["status"]
                    
                    if task_status == "TASK_STATUS_SUCCEEDED":
                        # 任务成功完成
                        if "videos" in result and len(result["videos"]) > 0:
                            return result["videos"][0]["video_url"]
                        else:
                            raise Exception("Task succeeded but no video found in result")
                    
                    elif task_status == "TASK_STATUS_SUCCEED":
                        # 任务成功完成 (处理API返回的另一种成功状态)
                        if "videos" in result and len(result["videos"]) > 0:
                            return result["videos"][0]["video_url"]
                        else:
                            raise Exception("Task succeeded but no video found in result")
                    
                    elif task_status == "TASK_STATUS_FAILED":
                        # 任务失败
                        reason = result["task"].get("reason", "Unknown error")
                        raise Exception(f"Task failed: {reason}")
                    
                    elif task_status == "TASK_STATUS_PROCESSING":
                        # 任务正在处理中，继续轮询
                        # 简化输出，不再显示详细进度信息
                        pass
                    
                    elif task_status == "TASK_STATUS_QUEUED":
                        # 任务排队中，继续轮询
                        # 简化输出，不再显示详细信息
                        pass
                    
                    else:
                        # 未知状态，继续轮询
                        # 简化输出，不再显示详细信息
                        pass
                
                # 等待后继续轮询
                time.sleep(poll_interval)
                
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise Exception(f"Failed to get task result after {max_attempts} attempts: {str(e)}")
                time.sleep(poll_interval)
        
        raise Exception(f"Task timeout after {max_attempts} attempts")

    def generate_video(
        self,
        image,
        prompt,
        model_version,
        resolution,
        duration,
        camera_fixed,
        seed,
        api_key="",
        last_image=None
    ):
        """生成图生视频"""
        # 获取API密钥
        api_key = self.get_api_key(api_key)
        if not api_key:
            raise ValueError("API key is required")
        
        # 验证提示词
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        # 验证输入图像
        if image is None:
            raise ValueError("Image is required")
        
        # 自动计算图像的宽高比
        aspect_ratio = self.get_image_aspect_ratio(image)
        print(f"Auto-calculated aspect ratio: {aspect_ratio}")
        
        # 处理图像
        base64_img = self.tensor_to_base64(image)
        if not base64_img:
            raise ValueError("Failed to process the image")
        
        # 处理结束图像 (仅Lite版本支持)
        base64_last_img = None
        if last_image is not None and model_version == "lite":
            base64_last_img = self.tensor_to_base64(last_image)
            if not base64_last_img:
                raise ValueError("Failed to process the last image")
        
        try:
            # 调用API生成视频
            print(f"Calling Seedance {model_version.capitalize()} Img2Video API...")
            task_id = self.call_seedance_img2video_api(
                api_key=api_key,
                image=base64_img,
                prompt=prompt,
                model_version=model_version,
                resolution=resolution,
                aspect_ratio=aspect_ratio,  # 使用自动计算的宽高比
                duration=duration,
                camera_fixed=camera_fixed,
                seed=seed if seed != -1 else None,
                last_image=base64_last_img
            )
            
            print(f"Task submitted with ID: {task_id}")
            
            # 轮询任务结果
            print("Waiting for video generation to complete...")
            video_url = self.poll_task_result(api_key, task_id)
            
            print(f"Video generated successfully: {video_url}")
            
            # 如果支持ComfyUI视频输出，下载视频并返回VIDEO对象
            if HAS_COMFY_VIDEO and VideoFromFile is not None:
                # 使用同步方式下载视频，避免事件循环冲突
                try:
                    import urllib.request
                    video_io = io.BytesIO(urllib.request.urlopen(video_url).read())
                    video_output = VideoFromFile(video_io)
                    return (video_output,)
                except Exception as e:
                    print(f"Error downloading video synchronously: {str(e)}")
                    # 如果同步下载失败，回退到返回URL
                    return (video_url,)
            else:
                # 否则返回URL字符串
                return (video_url,)
            
        except Exception as e:
            print(f"Error generating video: {str(e)}")
            raise ValueError(f"Video generation failed: {str(e)}")


class SeedancePPIOText2VideoNode:
    """Seedance 文生视频节点 (派欧云)"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model_version": (["lite", "pro"], {"default": "lite"}),  # 模型版本
                "resolution": (["480p", "720p", "1080p"], {"default": "1080p"}),
                "aspect_ratio": (["21:9", "16:9", "4:3", "1:1", "3:4", "9:16", "9:21"], {"default": "16:9"}),
                "duration": ([5, 10], {"default": 5}),
                "camera_fixed": ("BOOLEAN", {"default": False, "label_on": "启用", "label_off": "禁用"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "api_key": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("VIDEO",) if HAS_COMFY_VIDEO else ("STRING",)
    RETURN_NAMES = ("video",) if HAS_COMFY_VIDEO else ("video_url",)
    FUNCTION = "generate_video"
    CATEGORY = "CC-API/Video"
    OUTPUT_NODE = False

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

    def call_seedance_text2video_api(self, api_key, prompt, model_version, resolution, aspect_ratio, 
                                     duration, camera_fixed, seed):
        """调用Seedance文生视频API"""
        try:
            # 准备请求数据
            payload = {
                "prompt": prompt,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "duration": duration,
                "camera_fixed": camera_fixed
            }
            
            # 添加种子
            if seed != -1:
                payload["seed"] = seed
            
            # 确定API端点
            if model_version == "lite":
                api_endpoint = "https://api.ppinfra.com/v3/async/seedance-v1-lite-t2v"
            else:  # pro
                api_endpoint = "https://api.ppinfra.com/v3/async/seedance-v1-pro-t2v"
            
            # 发送请求
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            response = requests.post(
                api_endpoint,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if "task_id" in result:
                    return result["task_id"]
                else:
                    raise Exception(f"API response missing task_id: {result}")
            else:
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")
                
        except Exception as e:
            raise Exception(f"Error calling Seedance Text2Video API: {str(e)}")

    def query_task_result(self, api_key, task_id):
        """查询任务结果"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            params = {
                "task_id": task_id
            }
            
            response = requests.get(
                "https://api.ppinfra.com/v3/async/task-result",
                headers=headers,
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Query task result failed with status {response.status_code}: {response.text}")
                
        except Exception as e:
            raise Exception(f"Error querying task result: {str(e)}")

    def poll_task_result(self, api_key, task_id, poll_interval=5, max_attempts=120):
        """轮询任务结果直到完成或失败"""
        for attempt in range(max_attempts):
            try:
                # 简化日志输出，只显示轮询尝试次数
                print(f"Polling attempt {attempt + 1}/{max_attempts}")
                
                result = self.query_task_result(api_key, task_id)
                
                if "task" in result:
                    task_status = result["task"]["status"]
                    
                    if task_status == "TASK_STATUS_SUCCEEDED":
                        # 任务成功完成
                        if "videos" in result and len(result["videos"]) > 0:
                            return result["videos"][0]["video_url"]
                        else:
                            raise Exception("Task succeeded but no video found in result")
                    
                    elif task_status == "TASK_STATUS_SUCCEED":
                        # 任务成功完成 (处理API返回的另一种成功状态)
                        if "videos" in result and len(result["videos"]) > 0:
                            return result["videos"][0]["video_url"]
                        else:
                            raise Exception("Task succeeded but no video found in result")
                    
                    elif task_status == "TASK_STATUS_FAILED":
                        # 任务失败
                        reason = result["task"].get("reason", "Unknown error")
                        raise Exception(f"Task failed: {reason}")
                    
                    elif task_status == "TASK_STATUS_PROCESSING":
                        # 任务正在处理中，继续轮询
                        # 简化输出，不再显示详细进度信息
                        pass
                    
                    elif task_status == "TASK_STATUS_QUEUED":
                        # 任务排队中，继续轮询
                        # 简化输出，不再显示详细信息
                        pass
                    
                    else:
                        # 未知状态，继续轮询
                        # 简化输出，不再显示详细信息
                        pass
                
                # 等待后继续轮询
                time.sleep(poll_interval)
                
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise Exception(f"Failed to get task result after {max_attempts} attempts: {str(e)}")
                time.sleep(poll_interval)
        
        raise Exception(f"Task timeout after {max_attempts} attempts")

    def generate_video(
        self,
        prompt,
        model_version,
        resolution,
        aspect_ratio,
        duration,
        camera_fixed,
        seed,
        api_key=""
    ):
        """生成文生视频"""
        # 获取API密钥
        api_key = self.get_api_key(api_key)
        if not api_key:
            raise ValueError("API key is required")
        
        # 验证提示词
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        try:
            # 调用API生成视频
            print(f"Calling Seedance {model_version.capitalize()} Text2Video API...")
            task_id = self.call_seedance_text2video_api(
                api_key=api_key,
                prompt=prompt,
                model_version=model_version,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                duration=duration,
                camera_fixed=camera_fixed,
                seed=seed if seed != -1 else None
            )
            
            print(f"Task submitted with ID: {task_id}")
            
            # 轮询任务结果
            print("Waiting for video generation to complete...")
            video_url = self.poll_task_result(api_key, task_id)
            
            print(f"Video generated successfully: {video_url}")
            
            # 如果支持ComfyUI视频输出，下载视频并返回VIDEO对象
            if HAS_COMFY_VIDEO and VideoFromFile is not None:
                # 使用同步方式下载视频，避免事件循环冲突
                try:
                    import urllib.request
                    video_io = io.BytesIO(urllib.request.urlopen(video_url).read())
                    video_output = VideoFromFile(video_io)
                    return (video_output,)
                except Exception as e:
                    print(f"Error downloading video synchronously: {str(e)}")
                    # 如果同步下载失败，回退到返回URL
                    return (video_url,)
            else:
                # 否则返回URL字符串
                return (video_url,)
            
        except Exception as e:
            print(f"Error generating video: {str(e)}")
            raise ValueError(f"Video generation failed: {str(e)}")


NODE_CLASS_MAPPINGS = {
    "SeedancePPIOImg2VideoNode": SeedancePPIOImg2VideoNode,
    "SeedancePPIOText2VideoNode": SeedancePPIOText2VideoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedancePPIOImg2VideoNode": "即梦视频 图生视频 (派欧云)",
    "SeedancePPIOText2VideoNode": "即梦视频 文生视频 (派欧云)",
}