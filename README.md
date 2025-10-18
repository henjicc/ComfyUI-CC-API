# ComfyUI-CC-API

ComfyUI自定义节点，用于调用各种API

## 已适配的API

- [即梦4.0 官方版](https://console.volcengine.com/ark/region:ark+cn-beijing/model/detail?Id=doubao-seedream-4-0)
- [即梦4.0 (fal)](https://fal.ai/models/fal-ai/bytedance/seedream/v4/edit)
- [豆包语音合成大模型](https://console.volcengine.com/ark/region:ark+cn-beijing/model/detail?Id=ve-tts)
- [海螺 MiniMax 语音合成与克隆](https://platform.minimaxi.com/document/t2a_api_intro)
- [Qwen3-TTS-Flash](https://bailian.console.aliyun.com/#/model-market/detail/qwen3-tts-flash)
- [派欧云 Vidu Q1、MiniMax Hailuo 02、MiniMax Audio、PixVerse V4.5、Seedance V1、即梦 4.0](https://ppio.com/user/register?invited_by=MLBDS6)

## 配置文件说明

将 `config.ini.example` 复制并重命名为 `config.ini`，然后根据需要填写各平台的API密钥：

或者也可以直接在节点中设置 API 密钥

然后在 ComfyUI 的 CC-API 目录下可以就能找到所有节点