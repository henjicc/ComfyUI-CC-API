import requests
import os
import time
import io
import base64
from typing import Tuple, Dict, Any
from .cc_utils import ImageUtils

# 尝试导入ComfyUI的视频处理模块
try:
    from comfy_api.input_impl import VideoFromFile
    from comfy_api.input import VideoInput
    HAS_COMFY_VIDEO = True
except ImportError:
    HAS_COMFY_VIDEO = False
    VideoInput = object
    VideoFromFile = None  # 定义为None以避免未绑定变量错误

class WanPPIOImg2VideoNode:
    """万相 Wan 2.5 Preview 图生视频节点 (派欧云)"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),  # 首帧图片
                "duration": ([5, 10], {"default": 5}),
                "resolution": (["480P", "720P", "1080P"], {"default": "1080P"}),
                "prompt_extend": ("BOOLEAN", {"default": True, "label_on": "启用", "label_off": "禁用"}),
                "watermark": ("BOOLEAN", {"default": False, "label_on": "添加", "label_off": "不添加"}),
                "audio": ("BOOLEAN", {"default": True, "label_on": "启用", "label_off": "禁用"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "api_key": ("STRING", {"default": ""}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "audio_url": ("STRING", {"default": ""}),
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
        """将图像张量转换为base64编码的字符串"""
        try:
            # 转换为PIL图像
            pil_image = ImageUtils.tensor_to_pil(image_tensor)
            
            # 转换为base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return img_str
        except Exception as e:
            print(f"图像转换失败: {e}")
            raise

    def call_wan_i2v_api(self, api_key, prompt, image, negative_prompt, audio_url, duration, resolution, prompt_extend, watermark, audio, seed):
        """调用万相图生视频API"""
        try:
            # 转换图像为base64
            image_base64 = self.tensor_to_base64(image)
            img_url = f"data:image/png;base64,{image_base64}"
            
            # 构建请求数据
            data = {
                "input": {
                    "prompt": prompt,
                    "img_url": img_url
                },
                "parameters": {
                    "resolution": resolution,
                    "duration": duration,
                    "prompt_extend": prompt_extend,
                    "watermark": watermark,
                    "audio": audio
                }
            }
            
            # 添加可选参数
            if negative_prompt:
                data["input"]["negative_prompt"] = negative_prompt
            if audio_url:
                data["input"]["audio_url"] = audio_url
            if seed != -1:
                data["parameters"]["seed"] = seed
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                "https://api.ppinfra.com/v3/async/wan-2.5-i2v-preview",
                json=data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("task_id")
            else:
                print(f"API调用失败，状态码: {response.status_code}")
                print(f"响应内容: {response.text}")
                return None
                
        except Exception as e:
            print(f"调用万相图生视频API时出错: {e}")
            return None

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
                    
                    elif task_status == "TASK_STATUS_FAILED":
                        # 任务失败
                        reason = result["task"].get("reason", "Unknown error")
                        raise Exception(f"Task failed: {reason}")
                    
                    elif task_status == "TASK_STATUS_PROCESSING":
                        # 任务正在处理中，继续轮询
                        pass
                    
                    elif task_status == "TASK_STATUS_QUEUED":
                        # 任务排队中，继续轮询
                        pass
                    
                    else:
                        # 未知状态，继续轮询
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
        image,
        duration,
        resolution,
        prompt_extend,
        watermark,
        audio,
        seed,
        api_key="",
        negative_prompt="",
        audio_url=""
    ):
        """生成视频的主函数"""
        try:
            # 获取API密钥
            api_key = self.get_api_key(api_key)
            if not api_key:
                raise ValueError("未找到PPIO API密钥。请在参数中提供，或设置环境变量PPIO_API_KEY，或在config.ini中配置。")
            
            # 调用API
            print("Calling Wan 2.5 Preview Img2Video API...")
            task_id = self.call_wan_i2v_api(
                api_key, prompt, image, negative_prompt, audio_url, 
                duration, resolution, prompt_extend, watermark, audio, seed
            )
            
            if not task_id:
                raise Exception("API调用失败，未获取到任务ID")
            
            print(f"Task submitted with ID: {task_id}")
            
            # 轮询结果
            print("Waiting for video generation to complete...")
            video_url = self.poll_task_result(api_key, task_id)
            
            if not video_url:
                raise Exception("视频生成失败或超时")
            
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
            print(f"生成视频时出错: {e}")
            raise ValueError(f"视频生成失败: {str(e)}")
            # 返回错误信息
            error_msg = f"错误: {str(e)}"
            return (error_msg,)


class WanPPIOText2VideoNode:
    """万相 Wan 2.5 Preview 文生视频节点 (派欧云)"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "size": (["832*480", "480*832", "624*624", "1280*720", "720*1280", "960*960", "1088*832", "832*1088", "1920*1080", "1080*1920", "1440*1440", "1632*1248", "1248*1632"], {"default": "1920*1080"}),
                "duration": ([5, 10], {"default": 5}),
                "prompt_extend": ("BOOLEAN", {"default": True, "label_on": "启用", "label_off": "禁用"}),
                "watermark": ("BOOLEAN", {"default": False, "label_on": "添加", "label_off": "不添加"}),
                "audio": ("BOOLEAN", {"default": True, "label_on": "启用", "label_off": "禁用"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "api_key": ("STRING", {"default": ""}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "audio_url": ("STRING", {"default": ""}),
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

    def call_wan_t2v_api(self, api_key, prompt, negative_prompt, audio_url, size, duration, prompt_extend, watermark, audio, seed):
        """调用万相文生视频API"""
        try:
            # 构建请求数据
            data = {
                "input": {
                    "prompt": prompt
                },
                "parameters": {
                    "size": size,
                    "duration": duration,
                    "prompt_extend": prompt_extend,
                    "watermark": watermark,
                    "audio": audio
                }
            }
            
            # 添加可选参数
            if negative_prompt:
                data["input"]["negative_prompt"] = negative_prompt
            if audio_url:
                data["input"]["audio_url"] = audio_url
            if seed != -1:
                data["parameters"]["seed"] = seed
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                "https://api.ppinfra.com/v3/async/wan-2.5-t2v-preview",
                json=data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("task_id")
            else:
                print(f"API调用失败，状态码: {response.status_code}")
                print(f"响应内容: {response.text}")
                return None
                
        except Exception as e:
            print(f"调用万相文生视频API时出错: {e}")
            return None

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
                    
                    elif task_status == "TASK_STATUS_FAILED":
                        # 任务失败
                        reason = result["task"].get("reason", "Unknown error")
                        raise Exception(f"Task failed: {reason}")
                    
                    elif task_status == "TASK_STATUS_PROCESSING":
                        # 任务正在处理中，继续轮询
                        pass
                    
                    elif task_status == "TASK_STATUS_QUEUED":
                        # 任务排队中，继续轮询
                        pass
                    
                    else:
                        # 未知状态，继续轮询
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
        size,
        duration,
        prompt_extend,
        watermark,
        audio,
        seed,
        api_key="",
        negative_prompt="",
        audio_url=""
    ):
        """生成视频的主函数"""
        try:
            # 获取API密钥
            api_key = self.get_api_key(api_key)
            if not api_key:
                raise ValueError("未找到PPIO API密钥。请在参数中提供，或设置环境变量PPIO_API_KEY，或在config.ini中配置。")
            
            # 调用API
            print("Calling Wan 2.5 Preview Text2Video API...")
            task_id = self.call_wan_t2v_api(
                api_key, prompt, negative_prompt, audio_url, 
                size, duration, prompt_extend, watermark, audio, seed
            )
            
            if not task_id:
                raise Exception("API调用失败，未获取到任务ID")
            
            print(f"Task submitted with ID: {task_id}")
            
            # 轮询结果
            print("Waiting for video generation to complete...")
            video_url = self.poll_task_result(api_key, task_id)
            
            if not video_url:
                raise Exception("视频生成失败或超时")
            
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
            print(f"生成视频时出错: {e}")
            raise ValueError(f"视频生成失败: {str(e)}")


# 节点映射
NODE_CLASS_MAPPINGS = {
    "WanPPIOImg2VideoNode": WanPPIOImg2VideoNode,
    "WanPPIOText2VideoNode": WanPPIOText2VideoNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanPPIOImg2VideoNode": "万相 Wan 2.5 图生视频 (派欧云)",
    "WanPPIOText2VideoNode": "万相 Wan 2.5 文生视频 (派欧云)"
}