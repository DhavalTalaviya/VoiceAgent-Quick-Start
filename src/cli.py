from .audio import trim_silence_pcm, reduce_noise_pcm
from .stt import Wav2Vec2Adapter
from .agent import Agent
from .tts import TTSAdapter
import pyaudio
import webrtcvad
from webrtc_noise_gain import AudioProcessor


def listen_and_process(
    sr, frame_ms, silence_threshold_ms,
    ns_level, agc_dbfs, aggressiveness,
    stt_model, device,
    agent: Agent,
    tts: TTSAdapter
):
    pa = pyaudio.PyAudio()
    ten_ms = int(sr * frame_ms / 1000)
    stream = pa.open(format=pyaudio.paInt16,
                     channels=1,
                     rate=sr,
                     input=True,
                     frames_per_buffer=ten_ms)

    vad = webrtcvad.Vad(aggressiveness)
    noise_proc = AudioProcessor(agc_dbfs, ns_level)
    stt = Wav2Vec2Adapter(stt_model, device, window_s=2.0, stride_s=1.0)

    buffer = b""
    silence = 0
    max_silence = silence_threshold_ms // frame_ms
    print("Listening... Ctrl+C to stop.")
    try:
        while True:
            raw = stream.read(ten_ms, exception_on_overflow=False)
            proc = noise_proc.Process10ms(raw).audio
            if vad.is_speech(proc, sr):
                silence = 0
            else:
                silence += 1
            buffer += raw
            if silence >= max_silence and buffer:
                trimmed = trim_silence_pcm(buffer, sr, aggressiveness, frame_ms, silence_threshold_ms)
                cleaned = reduce_noise_pcm(trimmed, sr, ns_level, agc_dbfs)
                
                stt.buffer = b"" 
                text = []
                if stt.accept_audio(cleaned):
                    text.append(stt.transcribe_full(cleaned))
                full_text = " ".join(text).strip()
                if full_text != "":
                    print(f"🗣️ {full_text}")
                    reply = agent.chat(full_text)
                    print(f"🤖 {reply}")
                    tts.speak(reply)
                buffer = b""
                silence = 0
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()