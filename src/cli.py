from .audio import trim_silence_pcm, reduce_noise_pcm
from .stt import Wav2Vec2Adapter
from .agent import Agent
from .tts import TTSAdapter
import pyaudio
import webrtcvad
from webrtc_noise_gain import AudioProcessor
import time


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
                t0 = time.perf_counter()
                trimmed = trim_silence_pcm(buffer, sr, aggressiveness, frame_ms, silence_threshold_ms)
                cleaned = reduce_noise_pcm(trimmed, sr, ns_level, agc_dbfs)
                t1 = time.perf_counter()
                audio_duration_s = len(cleaned) / (sr * 2)
                print(f"[Stats] Audio duration: {audio_duration_s:.2f} s")
                audio_proc_ms = (t1 - t0) * 1000
                print(f"[Timing] Audio processing took {audio_proc_ms:.1f} ms")
                stt.buffer = b"" 
                text = []
                if stt.accept_audio(cleaned):
                    t2 = time.perf_counter()
                    text.append(stt.transcribe_full(cleaned))
                    t3 = time.perf_counter()
                    stt_ms = (t3 - t2) * 1000
                    print(f"[Timing] STT took {stt_ms:.1f} ms")
                    rtf = stt_ms / (audio_duration_s * 1000)
                    print(f"[Stats]  Real-time factor (STT_time / audio_duration): {rtf:.2f}×")
                full_text = " ".join(text).strip()
                if full_text != "":
                    # text = stt.transcribe_full(cleaned)
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