from .cc_utils import ApiHandler, ImageUtils, ResultProcessor, CCConfig
import math


class Seedream4:
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
                "sequential_image_generation": ("BOOLEAN", {"default": False, "label_on": "启用", "label_off": "禁用"}),
                "enable_4k": ("BOOLEAN", {"default": False, "label_on": "启用", "label_off": "禁用"}),
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
        - 总像素取值范围：[1280x720, 4096x4096]
        - 宽高比取值范围：[1/16, 16]
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
        
        # 如果参考图片的总像素在允许范围内，直接使用参考尺寸
        if 1280*720 <= reference_pixels <= 4096*4096:
            # 确保宽高是16的倍数
            width = math.floor(reference_width / 16) * 16
            height = math.floor(reference_height / 16) * 16
            
            # 确保总像素在允许范围内（调整后可能超出范围）
            while width * height > 4096*4096:
                if width > height:
                    width -= 16
                else:
                    height -= 16
            
            while width * height < 1280*720:
                if width < height:
                    width += 16
                else:
                    height += 16
            
            return f"{width}x{height}"
        
        # 如果参考图片的总像素超出范围，按比例缩放
        if reference_pixels > 4096*4096:
            # 缩小到最大允许尺寸
            scale_factor = math.sqrt(4096*4096 / reference_pixels)
        else:
            # 放大到最小允许尺寸
            scale_factor = math.sqrt(1280*720 / reference_pixels)
        
        # 计算新尺寸
        new_width = reference_width * scale_factor
        new_height = reference_height * scale_factor
        
        # 确保宽高是16的倍数
        width = math.floor(new_width / 16) * 16
        height = math.floor(new_height / 16) * 16
        
        # 确保总像素在允许范围内
        while width * height > 4096*4096:
            if width > height:
                width -= 16
            else:
                height -= 16
        
        while width * height < 1280*720:
            if width < height:
                width += 16
            else:
                height += 16
        
        return f"{width}x{height}"

    @staticmethod
    def _generate_request_key(prompt, image_size, width, height, seed, num_images, max_images, 
                            sequential_image_generation, enable_4k, image_urls):
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
            str(enable_4k)
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
        sequential_image_generation=False,
        enable_4k=False,
        api_key=""
    ):
        # Get API key from parameter or config
        if not api_key:
            api_key = CCConfig().get_key()
        
        if not api_key:
            print("Error: No API key provided")
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
                    print(f"Error: Failed to process image {i} for Seedream 4.0")
        
        # 生成请求键
        request_key = self._generate_request_key(
            prompt, image_size, width, height, seed, num_images, max_images,
            sequential_image_generation, enable_4k, image_urls
        )
        
        # 检查缓存中是否已有相同请求的结果
        if request_key in self._request_cache:
            print(f"Using cached result for request with seed: {seed}")
            return self._request_cache[request_key]
        
        # Convert image_size to the format expected by the API
        size = None
        if image_size == "自定义":
            size = f"{width}x{height}"
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
                    size = "4096x4096"
                else:
                    size = "2048x2048"
                print(f"No reference image found, using default size: {size}")
        else:
            # 预设尺寸的4K和普通分辨率映射
            if enable_4k:
                if image_size == "16:9 (2560x1440)":
                    size = "4096x2304"
                elif image_size == "3:2 (2496x1664)":
                    size = "4096x2731"
                elif image_size == "4:3 (2304x1728)":
                    size = "4096x3072"
                elif image_size == "1:1 (2048x2048)":
                    size = "4096x4096"
                elif image_size == "3:4 (1728x2304)":
                    size = "3072x4096"
                elif image_size == "2:3 (1664x2496)":
                    size = "2731x4096"
                elif image_size == "9:16 (1440x2560)":
                    size = "2304x4096"
            else:
                if image_size == "16:9 (2560x1440)":
                    size = "2560x1440"
                elif image_size == "3:2 (2496x1664)":
                    size = "2496x1664"
                elif image_size == "4:3 (2304x1728)":
                    size = "2304x1728"
                elif image_size == "1:1 (2048x2048)":
                    size = "2048x2048"
                elif image_size == "3:4 (1728x2304)":
                    size = "1728x2304"
                elif image_size == "2:3 (1664x2496)":
                    size = "1664x2496"
                elif image_size == "9:16 (1440x2560)":
                    size = "1440x2560"
        
        try:
            result = ApiHandler.call_seedream_api(
                api_key=api_key,
                prompt=prompt,
                images=image_urls if image_urls else None,
                size=size,
                sequential_image_generation="auto" if sequential_image_generation else "disabled",
                max_images=max_images
            )
            
            if result:
                processed_result = ResultProcessor.process_image_result(result)
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
            error_result = ApiHandler.handle_image_generation_error("Seedream 4.0", e)
            # 将错误结果也存储到缓存中
            self._request_cache[request_key] = error_result
            return error_result


NODE_CLASS_MAPPINGS = {
    "Seedream4": Seedream4,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Seedream4": "即梦4.0",
}