# Batch Gen AI 工具使用指南

这是一个基于 Google Gemini API 的批量图片/视频生成工具。

## 1. 环境准备

确保你的系统已安装 Python 3.8 或更高版本。

## 2. 创建并激活虚拟环境

打开终端 (Terminal) 或命令行工具，进入项目根目录。

**macOS / Linux:**
```bash
# 创建名为 venv 的虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate
```

**Windows:**
```bash
# 创建名为 venv 的虚拟环境
python -m venv venv

# 激活虚拟环境
venv\Scripts\activate
```

## 3. 安装依赖

在激活的虚拟环境中运行：
```bash
pip install -r requirements.txt
```

## 4. 配置环境变量

1. 复制 `.env.example` 文件并重命名为 `.env`：
   ```bash
   cp .env.example .env
   # Windows 用户可以使用 copy .env.example .env
   ```
2. 打开 `.env` 文件，填入你的 Google Gemini API Key：
   ```
   GEMINI_API_KEY=你的API密钥
   ```
   (可以在 [Google AI Studio](https://aistudio.google.com/) 获取 API Key)

## 5. 运行应用

```bash
python app.py
```

## 6. 使用

打开浏览器访问终端中显示的地址（通常是 `http://127.0.0.1:5001`）。

- **图片生成模式**：上传参考图片，输入 Prompt，选择生成数量即可。
- **视频生成模式**：上传首帧和尾帧图片，输入 Prompt，生成过渡视频。

