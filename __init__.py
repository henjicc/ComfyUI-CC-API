import importlib
import importlib.util
import os

# 定义WEB_DIRECTORY变量，指向JavaScript文件目录
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "js")

node_list = [
    "seedream_node",
    "seedream_fal_node",
    "seedream_ppio_node",  # 添加派欧云即梦4.0节点
    "seedance_ppio_node",  # 添加派欧云即梦视频节点
    "minimax_hailuo_ppio_node",  # 添加派欧云Minimax Hailuo-02节点
    "ppio_task_result_node",  # 添加派欧云查询任务结果节点
    "pixverse_ppio_node",  # 添加派欧云PixVerse节点
    "minimax_ppio_node",  # 添加派欧云MiniMax节点
    "qwen3_tts_node",
    "minimax_tts_node",
    "minimax_voice_clone_node",
    "tools_node",
    "doubao_tts_node",
    "doubao_tts_mix_node",
    "vidu_q1_node",
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