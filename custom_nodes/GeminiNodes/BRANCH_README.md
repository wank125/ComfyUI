# Gemini Custom Nodes Branch

这个分支 (`gemini-custom-nodes`) 包含了为 ComfyUI 开发的 Gemini API 自定义节点。

## 安装方法

1. **克隆这个分支**：
   ```bash
   git clone -b gemini-custom-nodes git@github.com:wank125/ComfyUI.git
   cd ComfyUI
   ```

2. **安装依赖**：
   ```bash
   pip install -r custom_nodes/GeminiNodes/requirements.txt
   ```

3. **运行 ComfyUI**：
   ```bash
   python main.py
   ```

## 节点列表

### API 生成节点
- **Gemini Image Generator**: 使用 Gemini API 生成图像
- **Gemini Advanced Image Gen**: 高级图像生成（更多控制选项）

### API 解析节点
- **Gemini Response Parser**: 解析完整的 API 响应
- **Gemini Image Extractor**: 仅提取图像
- **Gemini Text Extractor**: 仅提取文本
- **Gemini Response Analyzer**: 分析响应结构

### 工具节点
- **Gemini API Tester**: 测试 API 连接

## 使用示例

查看以下示例文件：
- `example_gemini_generation_workflow.json` - 基础图像生成工作流
- `example_workflow.json` - 解析器使用示例

## 快速开始

1. 将 API key 输入到 Gemini Image Generator 节点
2. 输入提示词
3. 点击 Queue Prompt

## 注意事项

- 需要有效的 Gemini API key
- API endpoint: `https://xiaoai.plus/v1beta/models`
- 支持的模型: `gemini-2.5-flash-image-preview`, `gemini-2.0-flash-exp`

## 贡献

欢迎提交 issue 和 pull request 来改进这些节点！