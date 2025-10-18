import requests
import json
import os
import configparser
import io
import numpy as np
import torch
from PIL import Image
import math
from .cc_utils import ImageUtils, ResultProcessor, CCConfig


class Seedream4PPIO:
    # 用于存储请求缓存，防止重复请求
    _request_cache = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_size": (
                    [
                        "1K",  # 1K分辨率
                        "2K",  # 2K分辨率
                        "4K",  # 4K分辨率
                        "1:1 (2048x2048)",   # 1_1
                        "4:3 (2304x1728)",   # 4_3
                        "3:4 (1728x2304)",   # 3_4
                        "16:9 (2560x1440)",  # 16_9
                        "9:16 (1440x2560)",  # 9_16
                        "3:2 (2496x1664)",   # 3_2
                        "2:3 (1664x2496)",   # 2_3
                        "21:9 (3024x1296)",  # 21_9
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
                "sequential_image_generation": ("BOOLEAN", {"default": False, "label_on": "启用", "label_off": "禁用"}),
                "watermark": ("BOOLEAN", {"default": False, "label_on": "启用", "label_off": "禁用"}),
                "api_key": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "CC-API/Image"

    @staticmethod
    def calculate_optimal_size(reference_width, reference_height):
        """
        根据参考图片尺寸计算符合API限制的最佳匹配尺寸
        API限制:
        - 宽高比（宽度/高度）：范围为[1/3, 3]
        - 宽度和高度（像素）：> 14
        - 大小：不超过10 MB
        - 总像素值：不超过 6000×6000 PX
        """
        # 计算参考图片的宽高比
        aspect_ratio = reference_width / reference_height
        
        # 检查宽高比是否在允许范围内
        if aspect_ratio < 1/3:
            # 如果宽高比小于1/3，调整高度
            reference_height = reference_width * 3
        elif aspect_ratio > 3:
            # 如果宽高比大于3，调整宽度
            reference_width = reference_height * 3
        
        # 确保尺寸在允许范围内
        reference_width = max(14, min(6000, reference_width))
        reference_height = max(14, min(6000, reference_height))
        
        # 确保总像素不超过限制
        total_pixels = reference_width * reference_height
        if total_pixels > 6000 * 6000:
            scale_factor = math.sqrt((6000 * 6000) / total_pixels)
            reference_width *= scale_factor
            reference_height *= scale_factor
        
        # 确保宽高是16的倍数
        width = math.floor(reference_width / 16) * 16
        height = math.floor(reference_height / 16) * 16
        
        return {"width": width, "height": height}

    @staticmethod
    def _generate_request_key(prompt, image_size, width, height, seed, num_images, max_images, 
                            sequential_image_generation, watermark, image_urls):
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
            str(sequential_image_generation),
            str(watermark)
        ]
        
        # 如果有图片URL，也加入键的计算
        if image_urls:
            key_parts.extend(image_urls)
        
        return "|".join(key_parts)
    
    def get_ppio_api_key(self):
        """获取派欧云API密钥"""
        # 首先尝试从环境变量获取
        if os.environ.get("PPIO_API_KEY") is not None:
            return os.environ["PPIO_API_KEY"]
        
        # 然后尝试从配置文件获取
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        config_path = os.path.join(parent_dir, "config.ini")

        config = configparser.ConfigParser()
        config.read(config_path)

        try:
            if "ppio" in config:
                api_key = config["ppio"]["API_KEY"]
                # 检查是否是默认占位符
                if api_key == "<your_ppio_api_key_here>":
                    print("WARNING: You are using the default PPIO API key placeholder!")
                    print("Please set your actual PPIO API key in either:")
                    print("1. The config.ini file under [ppio] section")
                    print("2. Or as an environment variable named PPIO_API_KEY")
                    return ""
                return api_key
            else:
                print("PPIO API key not found in config.ini")
                return ""
        except KeyError:
            print("Error: PPIO API_KEY not found in config.ini or environment variables")
            return ""
    
    def call_ppio_seedream_api(self, api_key, prompt, images, size, sequential_image_generation, max_images, watermark):
        """调用派欧云即梦4.0 API"""
        try:
            # 准备请求数据
            payload = {
                "prompt": prompt,
            }
            
            # 添加图像
            if images:
                payload["images"] = images
            
            # 添加尺寸参数
            if isinstance(size, str) and size in ["1K", "2K", "4K"]:
                payload["size"] = size
            elif isinstance(size, dict):
                payload["size"] = f"{size['width']}x{size['height']}"
            elif isinstance(size, str) and "x" in size:
                payload["size"] = size
            
            # 添加序列化图像生成参数
            payload["sequential_image_generation"] = "auto" if sequential_image_generation else "disabled"
            
            # 添加最大图像数量参数
            payload["max_images"] = max_images
            
            # 添加水印参数
            payload["watermark"] = watermark
            
            # 发送请求
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            response = requests.post(
                "https://api.ppinfra.com/v3/seedream-4.0",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"PPIO Seedream API request failed with status {response.status_code}: {response.text}")
                return None
        except Exception as e:
            print(f"Error calling PPIO Seedream API: {str(e)}")
            return None

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
        sequential_image_generation=False,
        watermark=False,
        api_key=""
    ):
        # 获取API密钥
        if not api_key:
            api_key = self.get_ppio_api_key()
        
        if not api_key:
            print("Error: No PPIO API key provided")
            return ResultProcessor.create_blank_image()
        
        # 处理所有提供的图像
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
                    print(f"Error: Failed to process image {i} for Seedream 4.0 (PPIO)")
        
        # 生成请求键
        request_key = self._generate_request_key(
            prompt, image_size, width, height, seed, num_images, max_images,
            sequential_image_generation, watermark, image_urls
        )
        
        # 检查缓存中是否已有相同请求的结果
        if request_key in self._request_cache:
            print(f"Using cached result for request with seed: {seed}")
            return self._request_cache[request_key]
        
        # 转换image_size为API期望的格式
        size = None
        if image_size == "自定义":
            size = {"width": width, "height": height}
        elif image_size == "跟随参考":
            if reference_image:
                # 获取参考图片的尺寸
                ref_width, ref_height = reference_image.size
                # 计算最佳匹配尺寸
                size = self.calculate_optimal_size(ref_width, ref_height)
                print(f"Following reference image size: {size} (original: {ref_width}x{ref_height})")
            else:
                # 如果没有参考图片，使用默认尺寸
                size = {"width": 2048, "height": 2048}
                print(f"No reference image found, using default size: {size}")
        elif image_size in ["1K", "2K", "4K"]:
            # 直接使用分辨率选项
            size = image_size
        else:
            # 预设尺寸映射
            size_mapping = {
                "1:1 (2048x2048)": {"width": 2048, "height": 2048},
                "4:3 (2304x1728)": {"width": 2304, "height": 1728},
                "3:4 (1728x2304)": {"width": 1728, "height": 2304},
                "16:9 (2560x1440)": {"width": 2560, "height": 1440},
                "9:16 (1440x2560)": {"width": 1440, "height": 2560},
                "3:2 (2496x1664)": {"width": 2496, "height": 1664},
                "2:3 (1664x2496)": {"width": 1664, "height": 2496},
                "21:9 (3024x1296)": {"width": 3024, "height": 1296}
            }
            size = size_mapping.get(image_size, {"width": 2048, "height": 2048})
        
        try:
            result = self.call_ppio_seedream_api(
                api_key=api_key,
                prompt=prompt,
                images=image_urls if image_urls else None,
                size=size,
                sequential_image_generation=sequential_image_generation,
                max_images=max_images,
                watermark=watermark
            )
            
            if result:
                processed_result = self.process_ppio_result(result)
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
            error_result = self.handle_ppio_error("Seedream 4.0 (PPIO)", e)
            # 将错误结果也存储到缓存中
            self._request_cache[request_key] = error_result
            return error_result
    
    def process_ppio_result(self, result):
        """处理派欧云API结果并返回tensor"""
        try:
            images = []
            for img_url in result["images"]:
                img_response = requests.get(img_url)
                img = Image.open(io.BytesIO(img_response.content))
                img_array = np.array(img).astype(np.float32) / 255.0
                images.append(img_array)

            if not images:
                return ResultProcessor.create_blank_image()

            # 沿新第一维度堆叠图像
            if isinstance(images[0], torch.Tensor):
                stacked_images = torch.cat(images, dim=0)
            else:
                stacked_images = np.stack(images, axis=0)
                # 转换为PyTorch张量
                stacked_images = torch.from_numpy(stacked_images)

            return (stacked_images,)
        except Exception as e:
            print(f"Error processing PPIO result: {str(e)}")
            return ResultProcessor.create_blank_image()
    
    def handle_ppio_error(self, model_name, error):
        """一致地处理派欧云API错误"""
        print(f"Error generating image with {model_name}: {str(error)}")
        return ResultProcessor.create_blank_image()


NODE_CLASS_MAPPINGS = {
    "Seedream4PPIO": Seedream4PPIO,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Seedream4PPIO": "即梦4.0 (派欧云)",
}