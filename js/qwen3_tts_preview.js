import { app } from "../../scripts/app.js";

// 注册Qwen3-TTS节点的音频预览扩展
app.registerExtension({
    name: "Comfy.CC_API.Qwen3TTS",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // 检查是否是Qwen3-TTS节点
        if (nodeData.name === "Qwen3TTS") {
            // 获取原始的onNodeCreated方法
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            // 重写onNodeCreated方法
            nodeType.prototype.onNodeCreated = function() {
                // 调用原始的onNodeCreated方法
                const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                // 创建音频预览容器
                this.audioContainer = document.createElement("div");
                this.audioContainer.style.display = "none";
                this.audioContainer.style.width = "100%";
                this.audioContainer.style.height = "35px"; // 调整为原来高度的一半
                this.audioContainer.style.marginTop = "5px";
                
                // 创建音频预览元素
                this.audioPreview = document.createElement("audio");
                this.audioPreview.controls = true;
                this.audioPreview.style.width = "100%";
                this.audioPreview.style.height = "35px"; // 调整为原来高度的一半
                
                // 将音频预览元素添加到容器
                this.audioContainer.appendChild(this.audioPreview);
                
                // 将容器添加到节点
                this.addDOMWidget("audio_container", "div", this.audioContainer, {
                    getValue: () => "",
                    setValue: (value) => {},
                    serialize: false,
                    hideOnZoom: false,
                });
                
                // 获取voice widget
                const voiceWidget = this.widgets?.find(w => w.name === "voice");
                
                // 如果voice widget存在，为其添加值变化监听器
                if (voiceWidget) {
                    // 保存原始的callback方法
                    const originalCallback = voiceWidget.callback;
                    
                    // 重写callback方法
                    voiceWidget.callback = function(value) {
                        // 调用原始callback
                        if (originalCallback) {
                            originalCallback.apply(this, arguments);
                        }
                        
                        // 当音色值变化时，自动加载预览音频
                        this.loadPreviewAudio(value);
                    }.bind(this);
                    
                    // 初始加载预览音频
                    this.loadPreviewAudio(voiceWidget.value);
                }
                
                return ret;
            };
            
            // 添加加载预览音频的方法
            nodeType.prototype.loadPreviewAudio = function(voice) {
                if (!voice) return;
                
                console.log("自动加载预览音频:", voice);
                
                // 显示容器
                this.audioContainer.style.display = "block";
                
                // 发送API请求获取预览音频
                fetch("/qwen3_tts_preview", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({
                        voice: voice
                    })
                })
                .then(response => {
                    console.log("API响应状态:", response.status);
                    console.log("API响应头:", response.headers);
                    
                    if (!response.ok) {
                        return response.text().then(text => {
                            throw new Error(`HTTP error! status: ${response.status}, text: ${text}`);
                        });
                    }
                    
                    return response.text().then(text => {
                        console.log("API响应文本:", text);
                        try {
                            return JSON.parse(text);
                        } catch (e) {
                            console.error("JSON解析错误:", e);
                            console.error("响应文本:", text);
                            throw new Error(`Invalid JSON response: ${text}`);
                        }
                    });
                })
                .then(data => {
                    console.log("API响应数据:", data);
                    
                    if (data.error) {
                        throw new Error(data.error);
                    }
                    
                    // 创建音频URL并设置到audio元素
                    const audioBlob = this.base64ToBlob(data.audio, "audio/wav");
                    const audioUrl = URL.createObjectURL(audioBlob);
                    
                    // 如果已有URL，先释放
                    if (this.audioPreview.currentSrc) {
                        URL.revokeObjectURL(this.audioPreview.currentSrc);
                    }
                    
                    this.audioPreview.src = audioUrl;
                })
                .catch(error => {
                    console.error("加载预览音频失败:", error);
                    
                    // 3秒后隐藏整个容器
                    setTimeout(() => {
                        this.audioContainer.style.display = "none";
                    }, 3000);
                });
            };
            
            // 添加Base64转Blob的方法
            nodeType.prototype.base64ToBlob = function(base64, mimeType) {
                try {
                    const byteCharacters = atob(base64);
                    const byteNumbers = new Array(byteCharacters.length);
                    
                    for (let i = 0; i < byteCharacters.length; i++) {
                        byteNumbers[i] = byteCharacters.charCodeAt(i);
                    }
                    
                    const byteArray = new Uint8Array(byteNumbers);
                    return new Blob([byteArray], { type: mimeType });
                } catch (e) {
                    console.error("Base64转Blob失败:", e);
                    throw new Error("Failed to convert base64 to blob: " + e.message);
                }
            };
        }
    }
});