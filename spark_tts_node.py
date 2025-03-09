import os
import sys
import torch
import numpy as np
from huggingface_hub import snapshot_download
import soundfile as sf
import io
import folder_paths

# 添加 Spark-TTS 到路径
SPARK_TTS_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Spark_TTS")
if not os.path.exists(SPARK_TTS_PATH):
    print(f"Spark-TTS 路径不存在: {SPARK_TTS_PATH}，请确保已克隆 Spark-TTS 仓库")
    # 可以选择自动克隆仓库
    os.system(f"git clone https://github.com/SparkAudio/Spark-TTS.git {SPARK_TTS_PATH}")

sys.path.append(SPARK_TTS_PATH)

# 导入 Spark-TTS 相关模块
try:
    from .Spark_TTS.cli.SparkTTS import SparkTTS
except ImportError as e:
    print(f"无法导入 Spark-TTS 模块，请确保已正确安装 Spark-TTS: {e}")


class SparkTTSNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ref_audio": ("AUDIO",),
                "text": ("STRING", {"multiline": True, "default": "你好，这是一段测试文本。"}),
                "prompt_text": ("STRING", {"multiline": True, "default": ""}),
                "model_path": (["Spark-TTS-0.5B"], {"default": "Spark-TTS-0.5B"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
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

        model_path = os.path.join(folder_paths.models_dir, "sparktts", model_path)
        # 如果模型路径变更或设备变更，重新加载模型
        if self.model is None or model_path != self.model_path or device_str != self.device_str:
            print(f"加载 Spark-TTS 模型: {model_path}")

            # 检查模型是否存在，不存在则从 HuggingFace 下载
            if not os.path.exists(model_path):
                print(f"模型不存在，从 HuggingFace 下载: {model_path}")
                snapshot_download("SparkAudio/Spark-TTS-0.5B", local_dir=model_path)

            self.model = SparkTTS(model_path, device)
            self.model_path = model_path
            self.device_str = device_str

        return self.model

    def generate_speech(self, ref_audio, text, prompt_text, model_path, device):
        # 加载模型
        model = self.load_model_if_needed(model_path, device)

        # TODO: 从输入参数获取控制参数
        prompt_text = ""
        gender = "neutral"
        pitch = 0.0
        speed = 1.0

        # 准备控制参数
        control_params = {
            "gender": gender,
            "pitch": pitch,
            "speed": speed
        }

        # 处理参考音频
        prompt_speech_path = None
        if ref_audio is not None:
            # 参考 SeedVC-ComfyUI 的实现，保存临时参考音频文件
            temp_ref_path = os.path.join(self.output_dir, f"ref_audio_temp_{hash(str(ref_audio))}.wav")

            # 假设 ref_audio 是包含波形数据和采样率的元组
            waveform = ref_audio["waveform"]
            sample_rate = ref_audio["sample_rate"]

            # 将波形数据转换为正确的格式
            if len(waveform.shape) == 3:  # [batch, channels, samples]
                waveform = waveform.squeeze(0)  # 移除批次维度
            if len(waveform.shape) == 2:  # [channels, samples]
                waveform = waveform.transpose(0, 1)  # 转换为 [samples, channels]
            
            # 确保数据是 numpy 数组
            if torch.is_tensor(waveform):
                waveform = waveform.cpu().numpy()

            # 保存为临时文件
            sf.write(temp_ref_path, waveform, sample_rate)
            prompt_speech_path = temp_ref_path

        # 执行推理
        audio_array = inference(
            model=model,
            text=text,
            prompt_text=prompt_text if prompt_text else None,
            prompt_speech_path=prompt_speech_path,
            control_params=control_params if any(control_params.values()) else None,
            device=torch.device(device)
        )
        audio = {
            "waveform": torch.FloatTensor(audio_array).unsqueeze(0).unsqueeze(0),
            "sample_rate": 16000  # Spark-TTS 的采样率是 16000
        }
        return (audio,)


def inference(model, text, prompt_text=None, prompt_speech_path=None, control_params=None, device=None):
    """
    使用Spark-TTS模型生成语音
    
    参数:
        model: 加载的Spark-TTS模型
        text: 要转换为语音的文本
        prompt_text: 提示文本（可选）
        prompt_speech_path: 参考音频文件路径（可选）
        control_params: 控制参数字典，包含gender、pitch、speed等
        device: 运行设备
        
    返回:
        numpy数组形式的音频数据
    """
    # 准备输入
    inputs = {
        "text": text,
    }

    # 添加提示文本（如果有）
    if prompt_text:
        inputs["prompt_text"] = prompt_text

    # 添加参考音频（如果有）
    if prompt_speech_path and os.path.exists(prompt_speech_path):
        inputs["prompt_speech_path"] = prompt_speech_path

    # 添加控制参数（如果有）
    if control_params:
        for key, value in control_params.items():
            if value is not None and (isinstance(value, (int, float)) and value != 0) or (
                    isinstance(value, str) and value != "neutral"):
                inputs[key] = value

    # 执行推理
    with torch.no_grad():
        # 根据Spark-TTS的API调整这部分代码
        audio = model.inference(**inputs)

        # 如果返回的是张量，转换为numpy数组
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

    return audio


# 注册节点
NODE_CLASS_MAPPINGS = {
    "SparkTTS": SparkTTSNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SparkTTS": "Spark TTS 语音合成"
}
