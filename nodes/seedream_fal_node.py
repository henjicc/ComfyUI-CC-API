from .cc_utils import ImageUtils, ResultProcessor
import math
import requests
import json
import os
import configparser
import io
import numpy as np
import torch
from PIL import Image


class Seedream4Fal:
    # 用于存储请求缓存，防止重复请求
    _request_cache = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_size": (
                    [
                        "16:9 (2560x1440)",  # 16_9
                        "3:2 (2496x1664)",   # 3_2
                        "4:3 (2304x1728)",   # 4_3
                        "1:1 (2048x2048)",   # 1_1
                        "3:4 (1728x2304)",   # 3_4
                        "2:3 (1664x2496)",   # 2_3
                        "9:16 (1440x2560)",  # 9_16
                        "跟随参考",  # follow_reference
                        "自定义",  # custom
                    ],
                    {"default": "1:1 (2048x2048)"},
                ),
                "width": (
                    "INT",
                    {"default": 2048, "min": 1024, "max": 4096, "step": 16},
                ),
                "height": (
                    "INT",
                    {"default": 2048, "min": 1024, "max": 4096, "step": 16},
                ),
                "seed": (
                    "INT",
                    {"default": -1, "min": -1, "max": 2147483647},
                ),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 6}),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 6}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
                "image_9": ("IMAGE",),
                "image_10": ("IMAGE",),
                "enable_4k": ("BOOLEAN", {"default": False, "label_on": "启用", "label_off": "禁用"}),
                "enable_safety_checker": ("BOOLEAN", {"default": False, "label_on": "启用", "label_off": "禁用"}),
                "sync_mode": ("BOOLEAN", {"default": False, "label_on": "启用", "label_off": "禁用"}),
                "api_key": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "CC-API/Image"

    @staticmethod
    def calculate_optimal_size(reference_width, reference_height, enable_4k=False):
        """
        根据参考图片尺寸计算符合API限制的最佳匹配尺寸
        API限制:
        - 宽高必须介于1024和4096之间
        """
        # 计算参考图片的宽高比
        aspect_ratio = reference_width / reference_height
        
        # 检查宽高比是否在允许范围内
        if aspect_ratio < 1/16 or aspect_ratio > 16:
            # 如果超出范围，调整到最接近的允许值
            if aspect_ratio < 1/16:
                aspect_ratio = 1/16
            else:
                aspect_ratio = 16
        
        # 如果启用4K，将参考尺寸放大到4K级别
        if enable_4k:
            # 计算放大后的尺寸，保持宽高比
            if reference_width >= reference_height:
                # 以宽度为基准放大到4096
                scale_factor = 4096 / reference_width
            else:
                # 以高度为基准放大到4096
                scale_factor = 4096 / reference_height
            
            reference_width = reference_width * scale_factor
            reference_height = reference_height * scale_factor
        
        # 计算参考图片的总像素
        reference_pixels = reference_width * reference_height
        
        # 如果参考图片的总像素超出范围，按比例缩放
        if reference_pixels > 4096*4096:
            # 缩小到最大允许尺寸
            scale_factor = math.sqrt(4096*4096 / reference_pixels)
        elif reference_pixels < 1024*1024:
            # 放大到最小允许尺寸
            scale_factor = math.sqrt(1024*1024 / reference_pixels)
        else:
            # 在允许范围内，不需要缩放
            scale_factor = 1.0
        
        # 计算新尺寸
        new_width = reference_width * scale_factor
        new_height = reference_height * scale_factor
        
        # 确保宽高在允许范围内
        new_width = max(1024, min(4096, new_width))
        new_height = max(1024, min(4096, new_height))
        
        # 确保宽高是16的倍数
        width = math.floor(new_width / 16) * 16
        height = math.floor(new_height / 16) * 16
        
        return {"width": width, "height": height}

    @staticmethod
    def _generate_request_key(prompt, image_size, width, height, seed, num_images, max_images, 
                            enable_4k, enable_safety_checker, sync_mode, image_urls):
        """
        生成请求的唯一键，用于缓存判断
        """
        # 将所有参数组合成一个字符串作为键
        key_parts = [
            str(prompt),
            str(image_size),
            str(width),
            str(height),
            str(seed),
            str(num_images),
            str(max_images),
            str(enable_4k),
            str(enable_safety_checker),
            str(sync_mode)
        ]
        
        # 如果有图片URL，也加入键的计算
        if image_urls:
            key_parts.extend(image_urls)
        
        return "|".join(key_parts)
    
    def generate_image(
        self,
        prompt,
        image_size,
        width,
        height,
        seed,
        num_images,
        max_images,
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        image_7=None,
        image_8=None,
        image_9=None,
        image_10=None,
        enable_4k=False,
        enable_safety_checker=True,
        sync_mode=False,
        api_key=""
    ):
        # Get API key from parameter or config
        if not api_key:
            api_key = self.get_fal_api_key()
        
        if not api_key:
            print("Error: No FAL API key provided")
            return ResultProcessor.create_blank_image()
        
        # Upload all provided images
        image_urls = []
        reference_image = None

        for i, img in enumerate([image_1, image_2, image_3, image_4, image_5, 
                               image_6, image_7, image_8, image_9, image_10], 1):
            if img is not None:
                pil_image = ImageUtils.tensor_to_pil(img)
                if pil_image:
                    img_base64 = ImageUtils.pil_to_base64(pil_image)
                    if img_base64:
                        image_urls.append(img_base64)
                        # 保存第一张图片作为参考
                        if i == 1:
                            reference_image = pil_image
                else:
                    print(f"Error: Failed to process image {i} for Seedream 4.0 (fal)")
        
        # 生成请求键
        request_key = self._generate_request_key(
            prompt, image_size, width, height, seed, num_images, max_images,
            enable_4k, enable_safety_checker, sync_mode, image_urls
        )
        
        # 检查缓存中是否已有相同请求的结果
        if request_key in self._request_cache:
            print(f"Using cached result for request with seed: {seed}")
            return self._request_cache[request_key]
        
        # Convert image_size to the format expected by the API
        size = None
        if image_size == "自定义":
            size = {"width": width, "height": height}
        elif image_size == "跟随参考":
            if reference_image:
                # 获取参考图片的尺寸
                ref_width, ref_height = reference_image.size
                # 计算最佳匹配尺寸
                size = self.calculate_optimal_size(ref_width, ref_height, enable_4k)
                print(f"Following reference image size: {size} (original: {ref_width}x{ref_height}, 4K: {enable_4k})")
            else:
                # 如果没有参考图片，使用默认尺寸
                if enable_4k:
                    size = {"width": 4096, "height": 4096}
                else:
                    size = {"width": 2048, "height": 2048}
                print(f"No reference image found, using default size: {size}")
        else:
            # 预设尺寸的4K和普通分辨率映射
            if enable_4k:
                if image_size == "16:9 (2560x1440)":
                    size = {"width": 4096, "height": 2304}
                elif image_size == "3:2 (2496x1664)":
                    size = {"width": 4096, "height": 2731}
                elif image_size == "4:3 (2304x1728)":
                    size = {"width": 4096, "height": 3072}
                elif image_size == "1:1 (2048x2048)":
                    size = {"width": 4096, "height": 4096}
                elif image_size == "3:4 (1728x2304)":
                    size = {"width": 3072, "height": 4096}
                elif image_size == "2:3 (1664x2496)":
                    size = {"width": 2731, "height": 4096}
                elif image_size == "9:16 (1440x2560)":
                    size = {"width": 2304, "height": 4096}
            else:
                if image_size == "16:9 (2560x1440)":
                    size = {"width": 2560, "height": 1440}
                elif image_size == "3:2 (2496x1664)":
                    size = {"width": 2496, "height": 1664}
                elif image_size == "4:3 (2304x1728)":
                    size = {"width": 2304, "height": 1728}
                elif image_size == "1:1 (2048x2048)":
                    size = {"width": 2048, "height": 2048}
                elif image_size == "3:4 (1728x2304)":
                    size = {"width": 1728, "height": 2304}
                elif image_size == "2:3 (1664x2496)":
                    size = {"width": 1664, "height": 2496}
                elif image_size == "9:16 (1440x2560)":
                    size = {"width": 1440, "height": 2560}
        
        # 根据是否有图片决定使用哪个API端点
        if image_urls:
            # 有图片，使用图片编辑API
            endpoint = "https://fal.run/fal-ai/bytedance/seedream/v4/edit"
        else:
            # 没有图片，使用文生图API
            endpoint = "https://fal.run/fal-ai/bytedance/seedream/v4/text-to-image"
        
        try:
            # 准备请求参数
            payload = {
                "prompt": prompt,
                "image_size": size,
                "num_images": num_images,
                "max_images": max_images,
                "enable_safety_checker": enable_safety_checker,
                "sync_mode": sync_mode
            }
            
            # 如果有图片，添加到请求中
            if image_urls:
                payload["image_urls"] = image_urls
            
            # 如果指定了种子，添加到请求中
            if seed != -1:
                payload["seed"] = seed
            
            # 调用FAL API
            result = self.call_fal_api(
                api_key=api_key,
                endpoint=endpoint,
                payload=payload
            )
            
            if result:
                processed_result = self.process_fal_result(result)
                # 将结果存储到缓存中
                if processed_result is not None:
                    self._request_cache[request_key] = processed_result
                return processed_result
            else:
                blank_result = ResultProcessor.create_blank_image()
                # 将空白结果也存储到缓存中
                self._request_cache[request_key] = blank_result
                return blank_result
        except Exception as e:
            error_result = self.handle_fal_error("Seedream 4.0 (fal)", e)
            # 将错误结果也存储到缓存中
            self._request_cache[request_key] = error_result
            return error_result
    
    def get_fal_api_key(self):
        """Get the FAL API key from config or environment variable."""
        # 首先尝试从环境变量获取
        if os.environ.get("FAL_API_KEY") is not None:
            return os.environ["FAL_API_KEY"]
        
        # 然后尝试从配置文件获取
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        config_path = os.path.join(parent_dir, "config.ini")

        config = configparser.ConfigParser()
        config.read(config_path)

        try:
            if "fal" in config:
                api_key = config["fal"]["API_KEY"]
                # 检查是否是默认占位符
                if api_key == "<your_fal_api_key_here>":
                    print("WARNING: You are using the default FAL API key placeholder!")
                    print("Please set your actual FAL API key in either:")
                    print("1. The config.ini file under [fal] section")
                    print("2. Or as an environment variable named FAL_API_KEY")
                    return ""
                return api_key
            else:
                print("FAL API key not found in config.ini")
                return ""
        except KeyError:
            print("Error: FAL API_KEY not found in config.ini or environment variables")
            return ""
    
    def call_fal_api(self, api_key, endpoint, payload):
        """Call FAL API and return result."""
        try:
            # Make the API request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Key {api_key}"
            }
            
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"FAL API request failed with status {response.status_code}: {response.text}")
                return None
        except Exception as e:
            print(f"Error calling FAL API: {str(e)}")
            return None
    
    def process_fal_result(self, result):
        """Process FAL API result and return tensor."""
        try:
            images = []
            for img_info in result["images"]:
                if "url" in img_info:
                    img_response = requests.get(img_info["url"])
                    img = Image.open(io.BytesIO(img_response.content))
                    img_array = np.array(img).astype(np.float32) / 255.0
                    images.append(img_array)

            if not images:
                return ResultProcessor.create_blank_image()

            # Stack the images along a new first dimension
            if isinstance(images[0], torch.Tensor):
                stacked_images = torch.cat(images, dim=0)
            else:
                stacked_images = np.stack(images, axis=0)
                # Convert to PyTorch tensor
                stacked_images = torch.from_numpy(stacked_images)

            return (stacked_images,)
        except Exception as e:
            print(f"Error processing FAL result: {str(e)}")
            return ResultProcessor.create_blank_image()
    
    def handle_fal_error(self, model_name, error):
        """Handle FAL API errors consistently."""
        print(f"Error generating image with {model_name}: {str(error)}")
        return ResultProcessor.create_blank_image()


NODE_CLASS_MAPPINGS = {
    "Seedream4Fal": Seedream4Fal,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Seedream4Fal": "即梦4.0 (fal)",
}