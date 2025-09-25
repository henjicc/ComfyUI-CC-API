import { app } from "../../scripts/app.js";

// 注册MiniMax音色选择器节点的刷新按钮扩展
app.registerExtension({
    name: "Comfy.CC_API.MiniMaxVoiceSelector",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // 检查是否是MiniMax音色选择器节点
        if (nodeData.name === "MiniMaxVoiceSelector") {
            // 获取原始的onNodeCreated方法
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            // 重写onNodeCreated方法
            nodeType.prototype.onNodeCreated = function() {
                // 调用原始的onNodeCreated方法
                const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                // 创建刷新按钮
                const refreshButton = document.createElement("button");
                refreshButton.innerText = "刷新音色列表";
                refreshButton.style.display = "block";
                refreshButton.style.width = "100%";
                refreshButton.style.marginTop = "5px";
                refreshButton.style.marginBottom = "5px";
                refreshButton.style.backgroundColor = "#4CAF50";
                refreshButton.style.color = "white";
                refreshButton.style.border = "none";
                refreshButton.style.padding = "8px 16px";
                refreshButton.style.textAlign = "center";
                refreshButton.style.textDecoration = "none";
                refreshButton.style.display = "inline-block";
                refreshButton.style.fontSize = "14px";
                refreshButton.style.cursor = "pointer";
                refreshButton.style.borderRadius = "4px";
                
                // 添加鼠标悬停效果
                refreshButton.addEventListener("mouseenter", () => {
                    refreshButton.style.backgroundColor = "#45a049";
                });
                
                refreshButton.addEventListener("mouseleave", () => {
                    refreshButton.style.backgroundColor = "#4CAF50";
                });
                
                // 添加点击事件处理程序
                refreshButton.addEventListener("click", async () => {
                    // 获取API密钥widget
                    const apiKeyWidget = this.widgets?.find(w => w.name === "api_key");
                    const voiceNameWidget = this.widgets?.find(w => w.name === "voice_name");
                    const voiceTypeWidget = this.widgets?.find(w => w.name === "voice_type");
                    
                    if (!apiKeyWidget || !voiceNameWidget) {
                        alert("找不到必要的widgets");
                        return;
                    }
                    
                    const apiKey = apiKeyWidget.value;
                    if (!apiKey) {
                        alert("请先输入API密钥");
                        return;
                    }
                    
                    // 获取voice_type参数，默认为all
                    const voiceType = voiceTypeWidget ? voiceTypeWidget.value : "all";
                    
                    // 禁用按钮并显示加载状态
                    refreshButton.disabled = true;
                    const originalText = refreshButton.innerText;
                    refreshButton.innerText = "正在刷新...";
                    
                    try {
                        // 发送API请求获取音色列表
                        const response = await fetch("/minimax_refresh_voices", {
                            method: "POST",
                            headers: {
                                "Content-Type": "application/json",
                            },
                            body: JSON.stringify({
                                api_key: apiKey,
                                voice_type: voiceType
                            })
                        });
                        
                        if (!response.ok) {
                            throw new Error(`HTTP error! status: ${response.status}`);
                        }
                        
                        const data = await response.json();
                        
                        if (data.status === "success") {
                            // 更新voice_name widget的选项
                            voiceNameWidget.options.values = data.voice_names;
                            
                            // 如果当前选择的值不在新列表中，设置为第一个值
                            if (!data.voice_names.includes(voiceNameWidget.value)) {
                                voiceNameWidget.value = data.voice_names[0] || "";
                            }
                            
                            // 触发widget变化事件
                            if (voiceNameWidget.callback) {
                                voiceNameWidget.callback(voiceNameWidget.value);
                            }
                            
                            console.log("音色列表刷新成功，共获取到 " + data.count + " 个音色");
                        } else {
                            throw new Error(data.error || "刷新失败");
                        }
                    } catch (error) {
                        console.error("刷新音色列表失败:", error);
                        alert("刷新音色列表失败: " + error.message);
                    } finally {
                        // 恢复按钮状态
                        refreshButton.disabled = false;
                        refreshButton.innerText = originalText;
                    }
                });
                
                // 将按钮添加到节点
                this.addDOMWidget("refresh_button", "button", refreshButton, {
                    getValue: () => "",
                    setValue: (value) => {},
                    serialize: false,
                    hideOnZoom: false,
                });
                
                return ret;
            };
        }
    }
});