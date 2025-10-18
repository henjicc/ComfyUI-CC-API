import os
import requests
import time
import torch
import numpy as np
from PIL import Image
import io
import base64
import tempfile
import asyncio
import aiohttp
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

class ViduQ1Node:
    """Vidu Q1参考生视频节点 (派欧云)"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),  # 将参考图像1设置为必需输入
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "aspect_ratio": (["16:9", "9:16", "1:1"], {"default": "16:9"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "movement_amplitude": (["auto", "small", "medium", "large"], {"default": "auto"}),
                "bgm": ("BOOLEAN", {"default": False, "label_on": "启用", "label_off": "禁用"}),
                "api_key": ("STRING", {"default": ""}),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
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
        env_var_name = "PIOYUN_API_KEY"
        if os.environ.get(env_var_name) is not None:
            return os.environ[env_var_name]
        
        # 最后检查配置文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        config_path = os.path.join(parent_dir, "config.ini")
        
        if os.path.exists(config_path):
            import configparser
            config = configparser.ConfigParser()
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

    def call_vidu_q1_api(self, api_key, images, prompt, aspect_ratio, seed, movement_amplitude, bgm):
        """调用Vidu Q1参考生视频API"""
        try:
            # 准备请求数据
            payload = {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "movement_amplitude": movement_amplitude,
                "bgm": bgm
            }
            
            # 添加图像
            if images:
                payload["images"] = images
            
            # 添加种子
            if seed != -1:
                payload["seed"] = seed
            
            # 设置默认参数
            payload["duration"] = 5  # 目前仅支持5秒
            payload["resolution"] = "1080p"  # 目前仅支持1080p
            
            # 发送请求
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            response = requests.post(
                "https://api.ppinfra.com/v3/async/vidu-q1-reference2video",
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
            raise Exception(f"Error calling Vidu Q1 API: {str(e)}")

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

    def poll_task_result(self, api_key, task_id, poll_interval=5, max_attempts=60):
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

    async def download_url_to_bytesio(self, url: str, timeout: Optional[int] = None):
        """下载URL内容到BytesIO对象"""
        timeout_cfg = aiohttp.ClientTimeout(total=timeout) if timeout else None
        async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                return io.BytesIO(await resp.read())

    async def download_url_to_video_output(self, video_url: str):
        """下载视频URL并返回VIDEO输出"""
        if not HAS_COMFY_VIDEO or VideoFromFile is None:
            # 如果没有ComfyUI视频模块，直接返回URL
            return video_url
            
        try:
            video_io = await self.download_url_to_bytesio(video_url, timeout=300)
            if video_io is None:
                raise ValueError(f"Failed to download video from {video_url}")
            return VideoFromFile(video_io)
        except Exception as e:
            print(f"Error downloading video: {str(e)}")
            raise ValueError(f"Failed to download video: {str(e)}")

    def generate_video(
        self,
        image_1,  # 将image_1移到参数列表的开头，因为它现在是必需的
        prompt,
        aspect_ratio,
        seed,
        movement_amplitude,
        bgm,
        api_key="",
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        image_7=None
    ):
        """生成视频"""
        # 获取API密钥
        api_key = self.get_api_key(api_key)
        if not api_key:
            raise ValueError("API key is required")
        
        # 验证提示词
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        # 处理参考图像 - 现在image_1是必需的
        image_urls = []
        
        # 检查是否有至少一张图像
        if image_1 is None:
            raise ValueError("At least one reference image (image_1) is required")
        
        # 处理第一张图像（必需）
        base64_img_1 = self.tensor_to_base64(image_1)
        if base64_img_1:
            image_urls.append(base64_img_1)
        else:
            raise ValueError("Failed to process the required reference image (image_1)")
        
        # 处理其他可选图像
        for i, img in enumerate([image_2, image_3, image_4, image_5, image_6, image_7], 2):
            if img is not None:
                base64_img = self.tensor_to_base64(img)
                if base64_img:
                    image_urls.append(base64_img)
                else:
                    print(f"Warning: Failed to process image {i}")
        
        try:
            # 调用API生成视频
            print("Calling Vidu Q1 API...")
            task_id = self.call_vidu_q1_api(
                api_key=api_key,
                images=image_urls,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                seed=seed if seed != -1 else None,
                movement_amplitude=movement_amplitude,
                bgm=bgm
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


class ViduQ1StartEndNode:
    """Vidu Q1首尾帧视频节点 (派欧云)"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_image": ("IMAGE",),  # 起始帧图像
                "end_image": ("IMAGE",),    # 结束帧图像
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "movement_amplitude": (["auto", "small", "medium", "large"], {"default": "auto"}),
                "bgm": ("BOOLEAN", {"default": False, "label_on": "启用", "label_off": "禁用"}),
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
        env_var_name = "PIOYUN_API_KEY"
        if os.environ.get(env_var_name) is not None:
            return os.environ[env_var_name]
        
        # 最后检查配置文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        config_path = os.path.join(parent_dir, "config.ini")
        
        if os.path.exists(config_path):
            import configparser
            config = configparser.ConfigParser()
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

    def check_aspect_ratio(self, start_image, end_image):
        """检查起始帧和结束帧的宽高比是否接近"""
        try:
            # 将tensor转换为PIL图像以获取尺寸
            start_pil = ImageUtils.tensor_to_pil(start_image)
            end_pil = ImageUtils.tensor_to_pil(end_image)
            
            if start_pil and end_pil:
                start_ratio = start_pil.width / start_pil.height
                end_ratio = end_pil.width / end_pil.height
                
                # 检查宽高比是否在0.8~1.25之间
                ratio = start_ratio / end_ratio
                if 0.8 <= ratio <= 1.25:
                    return True
                else:
                    return False
            return False
        except Exception as e:
            print(f"Error checking aspect ratio: {str(e)}")
            return False

    def call_vidu_q1_start_end_api(self, api_key, images, prompt, seed, movement_amplitude, bgm):
        """调用Vidu Q1首尾帧视频API"""
        try:
            # 准备请求数据
            payload = {
                "prompt": prompt,
                "movement_amplitude": movement_amplitude,
                "bgm": bgm,
                "images": images  # 两张图像：第一张为起始帧，第二张为结束帧
            }
            
            # 添加种子
            if seed != -1:
                payload["seed"] = seed
            
            # 设置默认参数
            payload["duration"] = 5  # 目前仅支持5秒
            payload["resolution"] = "1080p"  # 目前仅支持1080p
            
            # 发送请求
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            response = requests.post(
                "https://api.ppinfra.com/v3/async/vidu-q1-startend2video",  # 首尾帧API端点
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
            raise Exception(f"Error calling Vidu Q1 Start-End API: {str(e)}")

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

    def poll_task_result(self, api_key, task_id, poll_interval=5, max_attempts=60):
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
        start_image,
        end_image,
        prompt,
        seed,
        movement_amplitude,
        bgm,
        api_key=""
    ):
        """生成首尾帧视频"""
        # 获取API密钥
        api_key = self.get_api_key(api_key)
        if not api_key:
            raise ValueError("API key is required")
        
        # 验证提示词
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        # 验证输入图像
        if start_image is None or end_image is None:
            raise ValueError("Both start and end images are required")
        
        # 检查起始帧和结束帧的宽高比是否接近
        if not self.check_aspect_ratio(start_image, end_image):
            raise ValueError("The aspect ratio of start and end images must be close (between 0.8 and 1.25)")
        
        # 处理图像
        start_base64 = self.tensor_to_base64(start_image)
        end_base64 = self.tensor_to_base64(end_image)
        
        if not start_base64:
            raise ValueError("Failed to process the start image")
        
        if not end_base64:
            raise ValueError("Failed to process the end image")
        
        image_urls = [start_base64, end_base64]
        
        try:
            # 调用API生成视频
            print("Calling Vidu Q1 Start-End API...")
            task_id = self.call_vidu_q1_start_end_api(
                api_key=api_key,
                images=image_urls,
                prompt=prompt,
                seed=seed if seed != -1 else None,
                movement_amplitude=movement_amplitude,
                bgm=bgm
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


class ViduQ1Img2VideoNode:
    """Vidu Q1图生视频节点 (派欧云)"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # 仅需要一张图像
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "movement_amplitude": (["auto", "small", "medium", "large"], {"default": "auto"}),
                "bgm": ("BOOLEAN", {"default": False, "label_on": "启用", "label_off": "禁用"}),
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
        env_var_name = "PIOYUN_API_KEY"
        if os.environ.get(env_var_name) is not None:
            return os.environ[env_var_name]
        
        # 最后检查配置文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        config_path = os.path.join(parent_dir, "config.ini")
        
        if os.path.exists(config_path):
            import configparser
            config = configparser.ConfigParser()
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

    def check_aspect_ratio(self, image):
        """检查图像的宽高比是否符合要求（小于1:4或4:1）"""
        try:
            # 将tensor转换为PIL图像以获取尺寸
            pil_image = ImageUtils.tensor_to_pil(image)
            
            if pil_image:
                width, height = pil_image.width, pil_image.height
                ratio = width / height
                
                # 检查宽高比是否小于1:4或4:1（即0.25到4之间）
                if 0.25 <= ratio <= 4:
                    return True
                else:
                    return False
            return False
        except Exception as e:
            print(f"Error checking aspect ratio: {str(e)}")
            return False

    def call_vidu_q1_img2video_api(self, api_key, image, prompt, seed, movement_amplitude, bgm):
        """调用Vidu Q1图生视频API"""
        try:
            # 准备请求数据
            payload = {
                "prompt": prompt,
                "movement_amplitude": movement_amplitude,
                "bgm": bgm,
                "images": [image]  # 仅需要一张图像
            }
            
            # 添加种子
            if seed != -1:
                payload["seed"] = seed
            
            # 设置默认参数
            payload["duration"] = 5  # 目前仅支持5秒
            payload["resolution"] = "1080p"  # 目前仅支持1080p
            
            # 发送请求
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            response = requests.post(
                "https://api.ppinfra.com/v3/async/vidu-q1-img2video",  # 图生视频API端点
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
            raise Exception(f"Error calling Vidu Q1 Img2Video API: {str(e)}")

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

    def poll_task_result(self, api_key, task_id, poll_interval=5, max_attempts=60):
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
        seed,
        movement_amplitude,
        bgm,
        api_key=""
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
        
        # 检查图像的宽高比是否符合要求
        if not self.check_aspect_ratio(image):
            raise ValueError("Image aspect ratio must be less than 1:4 or 4:1")
        
        # 处理图像
        base64_img = self.tensor_to_base64(image)
        
        if not base64_img:
            raise ValueError("Failed to process the image")
        
        try:
            # 调用API生成视频
            print("Calling Vidu Q1 Img2Video API...")
            task_id = self.call_vidu_q1_img2video_api(
                api_key=api_key,
                image=base64_img,
                prompt=prompt,
                seed=seed if seed != -1 else None,
                movement_amplitude=movement_amplitude,
                bgm=bgm
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


class ViduQ1Text2VideoNode:
    """Vidu Q1文生视频节点 (派欧云)"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "style": (["general", "anime"], {"default": "general"}),
                "aspect_ratio": (["16:9", "9:16", "1:1"], {"default": "16:9"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "movement_amplitude": (["auto", "small", "medium", "large"], {"default": "auto"}),
                "bgm": ("BOOLEAN", {"default": False, "label_on": "启用", "label_off": "禁用"}),
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
        env_var_name = "PIOYUN_API_KEY"
        if os.environ.get(env_var_name) is not None:
            return os.environ[env_var_name]
        
        # 最后检查配置文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        config_path = os.path.join(parent_dir, "config.ini")
        
        if os.path.exists(config_path):
            import configparser
            config = configparser.ConfigParser()
            config.read(config_path)
            
            try:
                if "ppio" in config and "API_KEY" in config["ppio"]:
                    return config["ppio"]["API_KEY"]
            except KeyError:
                pass
        
        return None

    def call_vidu_q1_text2video_api(self, api_key, prompt, style, aspect_ratio, seed, movement_amplitude, bgm):
        """调用Vidu Q1文生视频API"""
        try:
            # 准备请求数据
            payload = {
                "prompt": prompt,
                "style": style,
                "aspect_ratio": aspect_ratio,
                "movement_amplitude": movement_amplitude,
                "bgm": bgm
            }
            
            # 添加种子
            if seed != -1:
                payload["seed"] = seed
            
            # 设置默认参数
            payload["duration"] = 5  # 目前仅支持5秒
            payload["resolution"] = "1080p"  # 目前仅支持1080p
            
            # 发送请求
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            response = requests.post(
                "https://api.ppinfra.com/v3/async/vidu-q1-text2video",  # 文生视频API端点
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
            raise Exception(f"Error calling Vidu Q1 Text2Video API: {str(e)}")

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

    def poll_task_result(self, api_key, task_id, poll_interval=5, max_attempts=60):
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
        style,
        aspect_ratio,
        seed,
        movement_amplitude,
        bgm,
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
            print("Calling Vidu Q1 Text2Video API...")
            task_id = self.call_vidu_q1_text2video_api(
                api_key=api_key,
                prompt=prompt,
                style=style,
                aspect_ratio=aspect_ratio,
                seed=seed if seed != -1 else None,
                movement_amplitude=movement_amplitude,
                bgm=bgm
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
    "ViduQ1Node": ViduQ1Node,
    "ViduQ1StartEndNode": ViduQ1StartEndNode,
    "ViduQ1Img2VideoNode": ViduQ1Img2VideoNode,
    "ViduQ1Text2VideoNode": ViduQ1Text2VideoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ViduQ1Node": "Vidu Q1 参考生视频 (派欧云)",
    "ViduQ1StartEndNode": "Vidu Q1 首尾帧视频 (派欧云)",
    "ViduQ1Img2VideoNode": "Vidu Q1 图生视频 (派欧云)",
    "ViduQ1Text2VideoNode": "Vidu Q1 文生视频 (派欧云)",
}