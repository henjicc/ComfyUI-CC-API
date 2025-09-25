import configparser
import io
import os
import tempfile
import hashlib
import time
import base64
import json

import numpy as np
import requests
import torch
from PIL import Image


class CCConfig:
    """Singleton class to handle CC API configuration and client setup."""

    _instance = None
    _key = None
    _minimax_key = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CCConfig, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize configuration and API key."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        config_path = os.path.join(parent_dir, "config.ini")

        config = configparser.ConfigParser()
        config.read(config_path)

        try:
            if os.environ.get("VOLCENGINE_API_KEY") is not None:
                print("VOLCENGINE_API_KEY found in environment variables")
                self._key = os.environ["VOLCENGINE_API_KEY"]
            else:
                print("VOLCENGINE_API_KEY not found in environment variables")
                self._key = config["volcengine"]["API_KEY"]
                print("VOLCENGINE_API_KEY found in config.ini")
                os.environ["VOLCENGINE_API_KEY"] = self._key
                print("VOLCENGINE_API_KEY set in environment variables")

            # Check if API key is the default placeholder
            if self._key == "<your_volcengine_api_key_here>":
                print("WARNING: You are using the default API key placeholder!")
                print("Please set your actual API key in either:")
                print("1. The config.ini file under [volcengine] section")
                print("2. Or as an environment variable named VOLCENGINE_API_KEY")
        except KeyError:
            print("Error: API_KEY not found in config.ini or environment variables")
            
        # Initialize MiniMax API key
        try:
            if os.environ.get("MINIMAX_API_KEY") is not None:
                self._minimax_key = os.environ["MINIMAX_API_KEY"]
            else:
                self._minimax_key = config["minimax"]["API_KEY"]
                if self._minimax_key != "<your_minimax_api_key_here>":
                    os.environ["MINIMAX_API_KEY"] = self._minimax_key
        except KeyError:
            # MiniMax key is optional, so we don't print an error
            self._minimax_key = None

    def get_key(self):
        """Get the API key for volcengine."""
        return self._key

    def get_minimax_key(self):
        """Get the API key for MiniMax."""
        # First check if MiniMax API key is provided as environment variable
        if os.environ.get("MINIMAX_API_KEY") is not None:
            return os.environ["MINIMAX_API_KEY"]
        
        # Return the stored key
        return self._minimax_key


class ImageUtils:
    """Utility functions for image processing."""
    
    # Class-level cache for processed images
    _image_cache = {}
    _cache_max_size = 50  # Maximum number of cached images
    _cache_max_age = 3600  # Maximum cache age in seconds (1 hour)
    
    @staticmethod
    def _get_image_hash(image_tensor):
        """Generate a hash for the image tensor to use as cache key."""
        # Convert tensor to numpy array for hashing
        if isinstance(image_tensor, torch.Tensor):
            image_np = image_tensor.cpu().numpy()
        else:
            image_np = np.array(image_tensor)
        
        # Generate hash
        return hashlib.md5(image_np.tobytes()).hexdigest()
    
    @staticmethod
    def _clean_cache():
        """Clean old entries from the cache if it exceeds max size."""
        current_time = time.time()
        
        # Remove expired entries
        expired_keys = [
            key for key, (timestamp, _) in ImageUtils._image_cache.items()
            if current_time - timestamp > ImageUtils._cache_max_age
        ]
        for key in expired_keys:
            del ImageUtils._image_cache[key]
        
        # If still too large, remove oldest entries
        if len(ImageUtils._image_cache) > ImageUtils._cache_max_size:
            sorted_items = sorted(ImageUtils._image_cache.items(), key=lambda x: x[1][0])
            for key, _ in sorted_items[:len(ImageUtils._image_cache) - ImageUtils._cache_max_size]:
                del ImageUtils._image_cache[key]
    
    @staticmethod
    def _get_cached_image(image_hash):
        """Get a cached image if it exists and is not expired."""
        if image_hash in ImageUtils._image_cache:
            timestamp, pil_image = ImageUtils._image_cache[image_hash]
            if time.time() - timestamp < ImageUtils._cache_max_age:
                return pil_image
            else:
                # Remove expired entry
                del ImageUtils._image_cache[image_hash]
        return None
    
    @staticmethod
    def _cache_image(image_hash, pil_image):
        """Cache a processed image."""
        # Clean cache before adding new entry
        ImageUtils._clean_cache()
        
        # Add new entry
        ImageUtils._image_cache[image_hash] = (time.time(), pil_image)
    
    @staticmethod
    def clear_image_cache():
        """Clear the entire image cache."""
        ImageUtils._image_cache.clear()
        print("Image cache cleared")
    
    @staticmethod
    def get_cache_info():
        """Get information about the current cache state."""
        current_time = time.time()
        valid_entries = 0
        expired_entries = 0
        
        for timestamp, _ in ImageUtils._image_cache.values():
            if current_time - timestamp < ImageUtils._cache_max_age:
                valid_entries += 1
            else:
                expired_entries += 1
        
        return {
            "total_entries": len(ImageUtils._image_cache),
            "valid_entries": valid_entries,
            "expired_entries": expired_entries,
            "max_size": ImageUtils._cache_max_size,
            "max_age_seconds": ImageUtils._cache_max_age
        }

    @staticmethod
    def tensor_to_pil(image):
        """Convert image tensor to PIL Image."""
        try:
            # Convert the image tensor to a numpy array
            if isinstance(image, torch.Tensor):
                image_np = image.cpu().numpy()
            else:
                image_np = np.array(image)

            # Ensure the image is in the correct format (H, W, C)
            if image_np.ndim == 4:
                image_np = image_np.squeeze(0)  # Remove batch dimension if present
            if image_np.ndim == 2:
                image_np = np.stack([image_np] * 3, axis=-1)  # Convert grayscale to RGB
            elif image_np.shape[0] == 3:
                image_np = np.transpose(
                    image_np, (1, 2, 0)
                )  # Change from (C, H, W) to (H, W, C)

            # Normalize the image data to 0-255 range
            if image_np.dtype == np.float32 or image_np.dtype == np.float64:
                image_np = (image_np * 255).astype(np.uint8)

            # Convert to PIL Image
            return Image.fromarray(image_np)
        except Exception as e:
            print(f"Error converting tensor to PIL: {str(e)}")
            return None

    @staticmethod
    def pil_to_base64(pil_image, format="JPEG"):
        """Convert PIL Image to base64 string."""
        try:
            buffered = io.BytesIO()
            pil_image.save(buffered, format=format)
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return f"data:image/{format.lower()};base64,{img_str}"
        except Exception as e:
            print(f"Error converting PIL to base64: {str(e)}")
            return None

    @staticmethod
    def base64_to_tensor(base64_str):
        """Convert base64 string to image tensor."""
        try:
            # Extract the base64 part if it's in data URI format
            if base64_str.startswith("data:image/"):
                base64_str = base64_str.split(",")[1]
            
            # Decode base64 to bytes
            img_data = base64.b64decode(base64_str)
            
            # Convert to PIL Image
            img = Image.open(io.BytesIO(img_data))
            
            # Convert to numpy array
            img_array = np.array(img).astype(np.float32) / 255.0
            
            # Convert to tensor
            if len(img_array.shape) == 2:  # Grayscale
                img_array = np.stack([img_array] * 3, axis=-1)
            
            # Ensure shape is (H, W, C)
            if img_array.shape[2] > 3:  # RGBA
                img_array = img_array[:, :, :3]
            
            # Add batch dimension and convert to tensor
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            
            return img_tensor
        except Exception as e:
            print(f"Error converting base64 to tensor: {str(e)}")
            return None


class ResultProcessor:
    """Utility functions for processing API results."""

    @staticmethod
    def process_image_result(result):
        """Process image generation result and return tensor."""
        try:
            images = []
            for img_info in result["data"]:
                if "b64_json" in img_info:
                    img_tensor = ImageUtils.base64_to_tensor(img_info["b64_json"])
                    if img_tensor is not None:
                        images.append(img_tensor)
                elif "url" in img_info:
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
            print(f"Error processing image result: {str(e)}")
            return ResultProcessor.create_blank_image()

    @staticmethod
    def create_blank_image():
        """Create a blank black image tensor."""
        blank_img = Image.new("RGB", (512, 512), color="black")
        img_array = np.array(blank_img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]
        return (img_tensor,)


class ApiHandler:
    """Utility functions for API interactions."""

    @staticmethod
    def call_seedream_api(api_key, prompt, images=None, size="2048x2048", 
                         sequential_image_generation="disabled", max_images=15):
        """Call Seedream 4.0 API and return result."""
        try:
            # Prepare the request payload
            payload = {
                "model": "doubao-seedream-4-0-250828",
                "prompt": prompt,
                "size": size,
                "response_format": "b64_json",
                "sequential_image_generation": sequential_image_generation,
                "stream": False,
                "watermark": False
            }
            
            # Add images if provided
            if images is not None:
                if isinstance(images, list) and len(images) > 0:
                    payload["image"] = images
                elif isinstance(images, str):
                    payload["image"] = images
            
            # Add sequential image generation options if needed
            if sequential_image_generation == "auto":
                payload["sequential_image_generation_options"] = {
                    "max_images": max_images
                }
            
            # Make the API request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            response = requests.post(
                "https://ark.cn-beijing.volces.com/api/v3/images/generations",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"API request failed with status {response.status_code}: {response.text}")
                return None
        except Exception as e:
            print(f"Error calling Seedream API: {str(e)}")
            return None

    @staticmethod
    def handle_image_generation_error(model_name, error):
        """Handle image generation errors consistently."""
        print(f"Error generating image with {model_name}: {str(error)}")
        return ResultProcessor.create_blank_image()