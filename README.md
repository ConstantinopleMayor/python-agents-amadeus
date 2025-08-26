
# LiveKit - Agent - Amadeus  

这是一个基于 [LiveKit Agents SDK](https://github.com/livekit/agents) 构建的 Python 项目，可与Amadeus红莉栖（アマデウス くりす）进行实时视频语音对话，并附加一个实时翻译字幕悬浮窗。

## 🌟 功能特性

- **实时语音处理**: 集成了先进的 STT (语音转文本) 和 TTS (文本转语音) 服务，实现流畅的语音对话。
- **视觉能力**: 能够接收并处理来自用户摄像头或者共享屏幕的视频流。
- **可扩展插件**: 利用 LiveKit 的 MCP (Model Context Protocol) 集成了多种外部工具，如长期记忆、网页抓取、在线搜索等。
- **实时翻译与字幕**: 能够将LLM输出的文本实时翻译成中文，并以双语或仅译文形式显示在浮动字幕窗口中。

## 🛠️ 技术栈

- **核心框架**: LiveKit Agents SDK for Python
- **图形界面**: Tkinter (用于字幕悬浮窗)
- **AI 服务**:
    - **LLM**: Google Gemini
    - **STT**: SiliconFlow (`FunAudioLLM/SenseVoiceSmall`)
    - **TTS**: SiliconFlow (`FunAudioLLM/CosyVoice2-0.5B`)
    - **翻译**: OpenAI-compatible LLM (可配置)

## � 安装与配置

1.  **克隆仓库**
    
    ```bash
    git clone https://github.com/ConstantinopleMayor/python-agents-amadeus.git
    ```

    或者可以通过 [Download Zip](https://codeload.github.com/ConstantinopleMayor/python-agents-amadeus/zip/refs/heads/main)下载源代码，并在本地解压到文件。
    
2.  **创建并激活虚拟环境**
    
    ```bash
    cd <your-repo-directory>
    # Windows
    python -m venv venv
    .\venv\Scripts\activate
    
    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```
    
3.  **安装依赖**
    
    ```bash
    pip install -r requirements.txt
    ```
    
4.  **配置环境变量**
    -   在项目根目录创建一个名为 `.env` 的文件。
    -   根据你的服务提供商，填入必要的 API Keys 和 URL。文件内容应如下所示：

    ```env
    # LiveKit 服务器信息 (必需)
    LIVEKIT_URL=
    LIVEKIT_API_KEY=
    LIVEKIT_API_SECRET=
    
    # OpenAI API Key (用于 STT/TTS,必需)
    # 注意：代码中 TTS 和 STT 的 base_url 已硬编码为 siliconflow, 只需提供 siliconflow key
    # 如果使用其他服务, 请在 agent.py 中修改 base_url
    OPENAI_API_KEY=
    
    # Google Gemini API Key (必需)
    GOOGLE_API_KEY=
    
    # 字幕翻译服务配置 (使用兼容OpenAI的接口)
    SUBTITLE_TRANS_ENABLED=1
    SUBTITLE_TRANS_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 # 或其他模型, 如 glm-4-flash
    SUBTITLE_TRANS_API_KEY= # 你的服务API Key
    SUBTITLE_TRANS_BASE_URL= # 你的服务API地址, 例如https://api.siliconflow.cn/v1
    
    # MCP (模型上下文协议) 服务配置 (可选)
    # 用于增强 Agent 的能力，如联网搜索和长期记忆
    TAVILY_API_KEY=      # Tavily 搜索服务的 API Key
    OPENMEMORY_API_KEY=  # OpenMemory 长期记忆服务的 API Key
    ```

## MCP (模型上下文协议) 说明

本项目通过 MCP 集成了以下几个扩展功能，以增强智能体的能力：

-   **mcp-server-fetch**: 允许智能体抓取网页内容。
-   **mcp-server-time**: 允许智能体获取当前时间。
-   **Tavily**: 提供强大的在线搜索能力。你需要注册 [Tavily AI](https://tavily.com/) 并获取 API Key。
-   **OpenMemory**: 为智能体提供长期记忆存储。你需要注册 [OpenMemory](https://mem0.ai/openmemory-mcp) 并获取 API Key。

要使用这些功能，请确保配置有npx和uvx工具，并在`.env` 文件中填入对应的 `TAVILY_API_KEY` 和 `OPENMEMORY_API_KEY`，最后取消agent.py中对MCP服务代码的注释。

## ▶️ 如何运行

1.  确保你的 `.env` 文件已正确配置。
2.  在终端中运行以下命令启动 Agent:
    ```bash
    cd <your-repo-directory>
    .\venv\Scripts\activate
    python agent.py dev
    ```
3.  程序启动后，访问[livekit官网](https://cloud.livekit.io/)，通过沙箱启动前端网页。

## <caption> 字幕窗口使用说明

- **拖动**: 按住字幕文本区域（非按钮区域）拖动窗口。
- **缩放**: 将鼠标悬停在窗口的边缘或角落，当光标变化后，按住并拖动以调整大小。
- **切换模式**: 点击右下角的 `[M]` 按钮，在“双语”和“仅翻译”模式间切换。
- **最小化**: 点击右下角的 `—` 按钮，窗口将收起至屏幕右下角的停靠栏。
- **关闭**: 点击右下角的 `×` 按钮，彻底关闭字幕窗口。

