import os
import sys
import torch
import numpy as np
from huggingface_hub import snapshot_download
import soundfile as sf
from PIL import Image
import folder_paths

# 添加 Spark-TTS 到路径
SPARK_TTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "Spark-TTS")
if not os.path.exists(SPARK_TTS_PATH):
    print(f"Spark-TTS 路径不存在: {SPARK_TTS_PATH}，请确保已克隆 Spark-TTS 仓库")
    # 可以选择自动克隆仓库
    # os.system(f"git clone https://github.com/SparkAudio/Spark-TTS.git {SPARK_TTS_PATH}")

sys.path.append(SPARK_TTS_PATH)

# 导入 Spark-TTS 相关模块
try:
    from cli.inference import load_model, inference
except ImportError:
    print("无法导入 Spark-TTS 模块，请确保已正确安装 Spark-TTS")

class SparkTTSNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "你好，这是一段测试文本。"}),
                "model_path": ("STRING", {"default": "pretrained_models/Spark-TTS-0.5B"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            },
            "optional": {
                "prompt_text": ("STRING", {"multiline": True, "default": ""}),
                "prompt_speech_path": ("STRING", {"default": ""}),
                "gender": (["male", "female", "neutral"], {"default": "neutral"}),
                "pitch": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("audio_path",)
    FUNCTION = "generate_speech"
    CATEGORY = "audio"

    def __init__(self):
        self.model = None
        self.model_path = None
        self.device_str = None
        self.output_dir = os.path.join(folder_paths.get_output_directory(), "audio")
        os.makedirs(self.output_dir, exist_ok=True)

    def load_model_if_needed(self, model_path, device_str):
        device = torch.device(device_str)
        
        # 如果模型路径变更或设备变更，重新加载模型
        if self.model is None or model_path != self.model_path or device_str != self.device_str:
            print(f"加载 Spark-TTS 模型: {model_path}")
            
            # 检查模型是否存在，不存在则从 HuggingFace 下载
            if not os.path.exists(model_path):
                print(f"模型不存在，从 HuggingFace 下载: {model_path}")
                snapshot_download("SparkAudio/Spark-TTS-0.5B", local_dir=model_path)
            
            self.model = load_model(model_path, device)
            self.model_path = model_path
            self.device_str = device_str
        
        return self.model

    def generate_speech(self, text, model_path, device, prompt_text="", prompt_speech_path="", 
                        gender="neutral", pitch=0.0, speed=1.0):
        try:
            # 加载模型
            model = self.load_model_if_needed(model_path, device)
            
            # 准备控制参数
            control_params = {
                "gender": gender,
                "pitch": pitch,
                "speed": speed
            }
            
            # 生成唯一文件名
            output_filename = f"spark_tts_{hash(text + prompt_text)}.wav"
            output_path = os.path.join(self.output_dir, output_filename)
            
            # 执行推理
            audio_array = inference(
                model=model,
                text=text,
                prompt_text=prompt_text if prompt_text else None,
                prompt_speech_path=prompt_speech_path if prompt_speech_path else None,
                control_params=control_params if any(control_params.values()) else None,
                device=torch.device(device)
            )
            
            # 保存音频文件
            sf.write(output_path, audio_array, 24000)  # Spark-TTS 使用 24kHz 采样率
            
            print(f"已生成音频: {output_path}")
            return (output_path,)
            
        except Exception as e:
            print(f"生成语音时出错: {str(e)}")
            return (f"错误: {str(e)}",)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "SparkTTS": SparkTTSNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SparkTTS": "Spark TTS 语音合成"
} 