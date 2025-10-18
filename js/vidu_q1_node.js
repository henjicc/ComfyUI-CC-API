import { app } from "../../../scripts/app.js";

// 添加自定义UI元素
app.registerExtension({
    name: "CC-API.ViduQ1Node",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // 在节点注册前执行
        if (nodeData.name === "ViduQ1Node") {
            // 可以在这里添加自定义行为
            console.log("ViduQ1Node registered");
        }
    },
    
    async setup() {
        // 扩展初始化
        console.log("ViduQ1Node extension loaded");
    },
    
    async addCustomNodeDefs(defs) {
        // 添加自定义节点定义
    },
});