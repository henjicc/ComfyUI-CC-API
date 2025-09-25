import importlib
import importlib.util
import os

# 定义WEB_DIRECTORY变量，指向JavaScript文件目录
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "js")

node_list = [
    "seedream_node",
    "seedream_fal_node",
    "qwen3_tts_node",
    "minimax_tts_node",
    "minimax_voice_clone_node",
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for module_name in node_list:
    imported_module = importlib.import_module(f".nodes.{module_name}", __name__)

    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **imported_module.NODE_CLASS_MAPPINGS}
    NODE_DISPLAY_NAME_MAPPINGS = {
        **NODE_DISPLAY_NAME_MAPPINGS,
        **imported_module.NODE_DISPLAY_NAME_MAPPINGS,
    }


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]