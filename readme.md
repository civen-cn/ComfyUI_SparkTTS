# ComfyUI_SparkTTS

ComfyUI_SparkTTS is a custom node for ComfyUI that integrates [SparkAudio/Spark-TTS](https://github.com/SparkAudio/Spark-TTS) text-to-speech (TTS) functionality, allowing you to easily generate high-quality speech in your ComfyUI workflows.

## Introduction

Spark-TTS is an advanced text-to-speech system that leverages the power of Large Language Models (LLM) to achieve highly accurate and natural speech synthesis. It is designed to be efficient, flexible, and powerful, suitable for both research and production use. This plugin seamlessly integrates Spark-TTS capabilities into ComfyUI, enabling you to easily add speech synthesis capabilities to your workflows.

## Features

- **Simple and Efficient**: Spark-TTS is built entirely on Qwen2.5, eliminating the need for additional generative models like flow matching. It reconstructs audio directly from LLM-predicted codes, simplifying the process, improving efficiency, and reducing complexity.
- **High-Quality Voice Cloning**: Supports zero-shot voice cloning, capable of replicating speaker voices even without specific training data.
- **Bilingual Support**: Supports both Chinese and English, with zero-shot voice cloning capabilities for cross-lingual and code-switching scenarios, enabling natural and accurate speech synthesis in multiple languages.
- **Controllable Speech Generation**: Create virtual speakers by adjusting parameters such as gender, pitch, and speech rate.
- **Seamless ComfyUI Integration**: Easily add high-quality speech synthesis to your ComfyUI projects.

## Installation

### Prerequisites

- ComfyUI installed
- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster inference)

### Installation Steps

1. Clone this repository to ComfyUI's custom_nodes directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/civen-cn/ComfyUI_SparkTTS.git
```

2. Install dependencies:

```bash
cd ComfyUI_SparkTTS
pip install -r requirements.txt
```

3. Model Location:
https://huggingface.co/SparkAudio/Spark-TTS-0.5B models/Spark-TTS-0.5B

## Usage

1. Start ComfyUI
2. Find "SparkTTS" related nodes in the node list
3. Add nodes to your workflow
4. Configure parameters and run the workflow to generate speech

## Node Types

### SparkTTS Basic Node

- **Input**: Text content
- **Output**: Generated audio file

### SparkTTS Voice Cloning Node

- **Input**: Text content, reference audio file, reference audio text
- **Output**: Synthesized audio mimicking the reference voice

### SparkTTS Voice Control Node

- **Input**: Text content, voice parameters (gender, pitch, speed, etc.)
- **Output**: Customized synthesized audio based on parameters

## Parameter Description

- **Text Content**: Text to be converted to speech
- **Device**: Select device for inference (CPU/GPU)
- **Reference Audio**: Audio file for voice cloning
- **Reference Text**: Text corresponding to the reference audio
- **Voice Parameters**: Control various characteristics of synthesized speech (gender, pitch, speed, etc.)

## Example Workflow

![Example Workflow](./examples/example_workflow.png)

## Common Issues

1. **Issue**: Slow generation speed  
   **Solution**: Ensure GPU inference is used and try reducing model size

2. **Issue**: Suboptimal voice cloning results  
   **Solution**: Provide clearer reference audio and ensure accurate reference text matching

## License

This project is licensed under the Apache-2.0 License

## Acknowledgments

- Thanks to [SparkAudio/Spark-TTS](https://github.com/SparkAudio/Spark-TTS) for providing the excellent TTS model
- Thanks to the ComfyUI team for providing the excellent framework

## Disclaimer

The zero-shot voice cloning TTS model provided in this project is intended for academic research, educational purposes, and legitimate applications only. Do not use this model for unauthorized voice cloning, impersonation, fraud, or any illegal activities. Please ensure compliance with local laws and regulations when using this model. 