import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


class Wav2Vec2Adapter:
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-large-960h-lv60-self",
        device: str = "cpu",
        window_s: float = 1.0,
        stride_s: float = 0.5
    ):
        
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model     = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
        self.device    = device

        self.window_bytes = int(window_s * 16000) * 2
        self.stride_bytes = int(stride_s * 16000) * 2

        self.buffer = b""

    def accept_audio(self, pcm_bytes: bytes) -> bool:
        self.buffer += pcm_bytes
        return len(self.buffer) >= self.window_bytes

    def get_result(self) -> str:
        
        # chunk = self.buffer[:self.window_bytes]
        # self.buffer = self.buffer[self.stride_bytes:]
        audio_tensor = (
            torch.frombuffer(self.buffer, dtype=torch.int16)
                 .float() / 32768.0
        )
        inputs = self.processor(audio_tensor, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.model(inputs.input_values.to(self.device)).logits
        pred_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(pred_ids)[0]
    
    def transcribe_full(self, pcm_bytes: bytes) -> str:
        audio = torch.frombuffer(pcm_bytes, dtype=torch.int16).float() / 32768.0
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.model(inputs.input_values.to(self.device)).logits
        pred_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(pred_ids)[0]