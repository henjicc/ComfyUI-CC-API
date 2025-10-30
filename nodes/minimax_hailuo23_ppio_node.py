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

class MinimaxHailuo23PPIOImg2VideoNode:
    """Minimax Hailuo 2.3 图生视频节点 (派欧云)"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),  # 首帧图片
                "duration": ([6, 10], {"default": 6}),
                "resolution": (["768P", "1080P"], {"default": "1080P"}),
                "enable_prompt_expansion": ("BOOLEAN", {"default": True, "label_on": "启用", "label_off": "禁用"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "api_key": ("STRING", {"default": ""}),
            },
            "optional": {
                "end_image": ("IMAGE",),  # 结束帧图片
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

    def call_minimax_hailuo23_i2v_api(self, api_key, prompt, image, end_image, duration, resolution, enable_prompt_expansion, seed):
        """调用Minimax Hailuo 2.3 图生视频 API"""
        try:
            # 准备请求数据
            payload = {
                "prompt": prompt,
                "duration": duration,
                "resolution": resolution,
                "enable_prompt_expansion": enable_prompt_expansion
            }
            
            # 添加首帧图片
            base64_img = self.tensor_to_base64(image)
            if base64_img:
                payload["image"] = base64_img
            else:
                raise Exception("Failed to process the input image")
            
            # 添加结束帧图片
            if end_image is not None:
                base64_end_img = self.tensor_to_base64(end_image)
                if base64_end_img:
                    payload["end_image"] = base64_end_img
                else:
                    raise Exception("Failed to process the end image")
            
            # 添加种子
            if seed != -1:
                payload["seed"] = seed
            
            # 发送请求
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            response = requests.post(
                "https://api.ppinfra.com/v3/async/minimax-hailuo-2.3-i2v",
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
            raise Exception(f"Error calling Minimax Hailuo 2.3 I2V API: {str(e)}")

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
        image,
        duration,
        resolution,
        enable_prompt_expansion,
        seed,
        api_key="",
        end_image=None
    ):
        """生成视频"""
        # 获取API密钥
        api_key = self.get_api_key(api_key)
        if not api_key:
            raise ValueError("API key is required")
        
        # 验证提示词
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        # 验证参数组合
        if duration == 10 and resolution == "1080P":
            raise ValueError("10秒视频仅支持768P分辨率")
        
        try:
            # 调用API生成视频
            print("Calling Minimax Hailuo 2.3 I2V API...")
            task_id = self.call_minimax_hailuo23_i2v_api(
                api_key=api_key,
                prompt=prompt,
                image=image,
                end_image=end_image,
                duration=duration,
                resolution=resolution,
                enable_prompt_expansion=enable_prompt_expansion,
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


class MinimaxHailuo23PPIOText2VideoNode:
    """Minimax Hailuo 2.3 文生视频节点 (派欧云)"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "duration": ([6, 10], {"default": 6}),
                "resolution": (["768P", "1080P"], {"default": "1080P"}),
                "enable_prompt_expansion": ("BOOLEAN", {"default": True, "label_on": "启用", "label_off": "禁用"}),
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

    def call_minimax_hailuo23_t2v_api(self, api_key, prompt, duration, resolution, enable_prompt_expansion, seed):
        """调用Minimax Hailuo 2.3 文生视频 API"""
        try:
            # 准备请求数据
            payload = {
                "prompt": prompt,
                "duration": duration,
                "resolution": resolution,
                "enable_prompt_expansion": enable_prompt_expansion
            }
            
            # 添加种子
            if seed != -1:
                payload["seed"] = seed
            
            # 发送请求
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            response = requests.post(
                "https://api.ppinfra.com/v3/async/minimax-hailuo-2.3-t2v",
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
            raise Exception(f"Error calling Minimax Hailuo 2.3 T2V API: {str(e)}")

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
        duration,
        resolution,
        enable_prompt_expansion,
        seed,
        api_key=""
    ):
        """生成视频"""
        # 获取API密钥
        api_key = self.get_api_key(api_key)
        if not api_key:
            raise ValueError("API key is required")
        
        # 验证提示词
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        # 验证参数组合
        if duration == 10 and resolution == "1080P":
            raise ValueError("10秒视频仅支持768P分辨率")
        
        try:
            # 调用API生成视频
            print("Calling Minimax Hailuo 2.3 T2V API...")
            task_id = self.call_minimax_hailuo23_t2v_api(
                api_key=api_key,
                prompt=prompt,
                duration=duration,
                resolution=resolution,
                enable_prompt_expansion=enable_prompt_expansion,
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


class MinimaxHailuo23FastPPIOImg2VideoNode:
    """Minimax Hailuo 2.3 Fast 图生视频节点 (派欧云)"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),  # 首帧图片
                "duration": ([6, 10], {"default": 6}),
                "resolution": (["768P", "1080P"], {"default": "1080P"}),
                "enable_prompt_expansion": ("BOOLEAN", {"default": True, "label_on": "启用", "label_off": "禁用"}),
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

    def call_minimax_hailuo23_fast_i2v_api(self, api_key, prompt, image, duration, resolution, enable_prompt_expansion, seed):
        """调用Minimax Hailuo 2.3 Fast 图生视频 API"""
        try:
            # 准备请求数据
            payload = {
                "prompt": prompt,
                "duration": duration,
                "resolution": resolution,
                "enable_prompt_expansion": enable_prompt_expansion
            }
            
            # 添加首帧图片
            base64_img = self.tensor_to_base64(image)
            if base64_img:
                payload["image"] = base64_img
            else:
                raise Exception("Failed to process the input image")
            
            # 添加种子
            if seed != -1:
                payload["seed"] = seed
            
            # 发送请求
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            response = requests.post(
                "https://api.ppinfra.com/v3/async/minimax-hailuo-2.3-fast-i2v",
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
            raise Exception(f"Error calling Minimax Hailuo 2.3 Fast I2V API: {str(e)}")

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
        image,
        duration,
        resolution,
        enable_prompt_expansion,
        seed,
        api_key=""
    ):
        """生成视频"""
        # 获取API密钥
        api_key = self.get_api_key(api_key)
        if not api_key:
            raise ValueError("API key is required")
        
        # 验证提示词
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        # 验证参数组合
        if duration == 10 and resolution == "1080P":
            raise ValueError("10秒视频仅支持768P分辨率")
        
        try:
            # 调用API生成视频
            print("Calling Minimax Hailuo 2.3 Fast I2V API...")
            task_id = self.call_minimax_hailuo23_fast_i2v_api(
                api_key=api_key,
                prompt=prompt,
                image=image,
                duration=duration,
                resolution=resolution,
                enable_prompt_expansion=enable_prompt_expansion,
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


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "MinimaxHailuo23PPIOImg2VideoNode": MinimaxHailuo23PPIOImg2VideoNode,
    "MinimaxHailuo23PPIOText2VideoNode": MinimaxHailuo23PPIOText2VideoNode,
    "MinimaxHailuo23FastPPIOImg2VideoNode": MinimaxHailuo23FastPPIOImg2VideoNode
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "MinimaxHailuo23PPIOImg2VideoNode": "Minimax Hailuo 2.3 图生视频 (派欧云)",
    "MinimaxHailuo23PPIOText2VideoNode": "Minimax Hailuo 2.3 文生视频 (派欧云)",
    "MinimaxHailuo23FastPPIOImg2VideoNode": "Minimax Hailuo 2.3 Fast 图生视频 (派欧云)"
}