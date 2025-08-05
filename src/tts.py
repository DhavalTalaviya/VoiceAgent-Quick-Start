from TTS.api import TTS
import sounddevice as sd

class TTSAdapter:
    def __init__(self, model_name: str, speed: float = 1.2):
        self.tts = TTS(model_name)
        self.speed = speed
        self.sr = self.tts.synthesizer.output_sample_rate

    def speak(self, text: str):
        wav = self.tts.tts(text=text, speed=self.speed)
        sd.play(wav, self.sr)
        sd.wait()