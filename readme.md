# ComfyUI_SparkTTS

ComfyUI_SparkTTS 是一个 ComfyUI 的自定义节点，它集成了 Spark-TTS 文本转语音（TTS）功能，让您能够在 ComfyUI 工作流中轻松生成高质量的语音。

## 简介

Spark-TTS 是一款先进的文本转语音系统，利用大型语言模型 (LLM) 的强大功能实现高度准确且自然的语音合成。它旨在高效、灵活且功能强大，适合研究和生产使用。本插件将 Spark-TTS 的强大功能无缝集成到 ComfyUI 中，使您能够在工作流程中轻松添加语音合成能力。

## 功能特点

- **简洁高效**：Spark-TTS 完全基于 Qwen2.5 构建，无需使用流匹配等额外生成模型。它直接从 LLM 预测的代码中重建音频，简化流程，提高效率并降低复杂性。
- **高质量语音克隆**：支持零样本语音克隆，即使没有针对该语音的特定训练数据，也可以复制说话者的声音。
- **双语支持**：支持中英文，并具备跨语言、代码切换场景的零样本语音克隆能力，使模型能够高自然度、高准确度地合成多种语言的语音。
- **可控语音生成**：支持通过调整性别、音调、语速等参数创建虚拟说话人。
- **与 ComfyUI 工作流无缝集成**：轻松在您的 ComfyUI 项目中添加高质量语音合成功能。

## 安装方法

### 前提条件

- 已安装 ComfyUI
- Python 3.8 或更高版本
- CUDA 支持的 GPU（推荐用于更快的推理速度）

### 安装步骤

1. 克隆本仓库到 ComfyUI 的 custom_nodes 目录：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/ComfyUI_SparkTTS.git
```

2. 安装依赖：

```bash
cd ComfyUI_SparkTTS
pip install -r requirements.txt
```

3. 下载预训练模型：

```bash
# 使用 huggingface_hub 下载
python -c "from huggingface_hub import snapshot_download; snapshot_download('SparkAudio/Spark-TTS-0.5B', local_dir='models/Spark-TTS-0.5B')"

# 或使用 git-lfs 下载
mkdir -p models
git lfs install
git clone https://huggingface.co/SparkAudio/Spark-TTS-0.5B models/Spark-TTS-0.5B
```

## 使用方法

1. 启动 ComfyUI
2. 在节点列表中找到 "SparkTTS" 相关节点
3. 将节点添加到您的工作流中
4. 配置参数并运行工作流生成语音

## 节点类型

### SparkTTS 基础节点

- **输入**：文本内容
- **输出**：生成的音频文件

### SparkTTS 声音克隆节点

- **输入**：文本内容、参考音频文件、参考音频文本
- **输出**：模仿参考声音的合成音频

### SparkTTS 声音控制节点

- **输入**：文本内容、声音参数（性别、音调、语速等）
- **输出**：根据参数定制的合成音频

## 参数说明

- **文本内容**：要转换为语音的文本
- **设备**：选择用于推理的设备（CPU/GPU）
- **参考音频**：用于声音克隆的音频文件
- **参考文本**：参考音频对应的文本内容
- **语音参数**：控制合成语音的各种特性（性别、音调、语速等）

## 示例工作流

![示例工作流](./examples/example_workflow.png)

## 常见问题

1. **问题**：生成速度较慢  
   **解决方案**：确保使用 GPU 进行推理，并尝试减小模型大小

2. **问题**：声音克隆效果不理想  
   **解决方案**：提供更清晰的参考音频，并确保参考文本准确匹配

## 许可证

本项目采用 Apache-2.0 许可证

## 致谢

- 感谢 [SparkAudio/Spark-TTS](https://github.com/SparkAudio/Spark-TTS) 提供的出色 TTS 模型
- 感谢 ComfyUI 团队提供的优秀框架

## 免责声明

本项目提供的零样本声音克隆 TTS 模型仅用于学术研究、教育目的和合法应用。请勿将此模型用于未经授权的声音克隆、冒充、欺诈或任何非法活动。使用本模型时请确保遵守当地法律法规。
