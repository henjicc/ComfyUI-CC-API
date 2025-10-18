import requests
import os
import json
from typing import Tuple, Dict, Any

class PPIOQueryTaskResultNode:
    """派欧云查询任务结果节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_id": ("STRING", {"default": "", "multiline": False}),
                "api_key": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("响应内容", "视频链接")
    FUNCTION = "query_task_result"
    CATEGORY = "CC-API/Utils"
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
            config.read(config_path)
            
            try:
                if "ppio" in config and "API_KEY" in config["ppio"]:
                    return config["ppio"]["API_KEY"]
            except KeyError:
                pass
        
        return None

    def extract_video_url(self, response_dict: Dict[str, Any]) -> str:
        """从响应字典中提取视频链接"""
        try:
            # 检查响应中是否有videos字段
            if "videos" in response_dict and isinstance(response_dict["videos"], list):
                # 如果有视频且不为空，返回第一个视频的URL
                if len(response_dict["videos"]) > 0 and "video_url" in response_dict["videos"][0]:
                    return response_dict["videos"][0]["video_url"]
            # 如果没有视频链接，返回空字符串
            return ""
        except Exception as e:
            print(f"Error extracting video URL: {str(e)}")
            return ""

    def query_task_result(self, task_id: str, api_key: str = "") -> Tuple[str, str]:
        """查询派欧云任务结果"""
        # 获取API密钥
        actual_api_key = self.get_api_key(api_key)
        if not actual_api_key:
            raise ValueError("API key is required")
        
        # 验证任务ID
        if not task_id:
            raise ValueError("Task ID is required")
        
        try:
            # 准备请求，明确禁用缓存
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {actual_api_key}",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache"
            }
            
            params = {
                "task_id": task_id
            }
            
            # 创建一个不使用缓存的会话
            session = requests.Session()
            session.headers.update(headers)
            
            # 发送请求
            response = session.get(
                "https://api.ppinfra.com/v3/async/task-result",
                params=params,
                timeout=30
            )
            
            # 检查响应状态
            if response.status_code == 200:
                # 返回原始响应字符串和提取的视频链接
                result_str = response.text
                result_dict = response.json()
                video_url = self.extract_video_url(result_dict)
                return (result_str, video_url)
            else:
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")
                
        except Exception as e:
            raise Exception(f"Error querying task result: {str(e)}")


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "PPIOQueryTaskResultNode": PPIOQueryTaskResultNode
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "PPIOQueryTaskResultNode": "派欧云查询任务结果"
}