import webrtcvad
import soundfile as sf
import resampy
import numpy as np
import librosa
from webrtc_noise_gain import AudioProcessor

class Frame:
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def read_frames(pcm_data, sr, frame_duration_ms=30):
    duration = frame_duration_ms / 1000.0
    bytes_per_frame = int(sr * duration * 2)
    offset = 0
    timestamp = 0.0
    
    while offset + bytes_per_frame < len(pcm_data):
        yield Frame(pcm_data[offset:offset+bytes_per_frame], timestamp, duration)
        timestamp += duration
        offset += bytes_per_frame

def preprocess_wav(data, sr):
    # data = np.frombuffer(data, dtype=np.int16)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    target_sr = 16000
    if sr != target_sr:
        data = resampy.resample(data, sr, target_sr)
        sr = target_sr
    pcm_bytes = (data * 32767).astype(np.int16).tobytes()
    return pcm_bytes, sr

def convert_to_hd(input_path: str, output_path: str, target_sr: int = 16000):
    data, sr = librosa.load(input_path, sr=None, mono=True)
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    int_data = (data * 32767.0).astype('int16')
    sf.write(output_path, int_data, sr, subtype='PCM_16')

def trim_silence_pcm(
    pcm: bytes,
    sr: int,
    aggressiveness: int = 2,
    frame_ms: int = 20,
    padding_ms: int = 300
) -> bytes:
    
    vad = webrtcvad.Vad(aggressiveness)
    frames = list(read_frames(pcm, sr, frame_ms))
    expected_bytes = int(sr * frame_ms / 1000) * 2

    speech_flags = []
    for f in frames:
        if len(f.bytes) != expected_bytes:
            continue
        speech_flags.append(vad.is_speech(f.bytes, sr))

    if not any(speech_flags):
        return pcm
    
    first = next(i for i, flag in enumerate(speech_flags) if flag)
    last  = len(speech_flags) - 1 - next(i for i, flag in enumerate(reversed(speech_flags)) if flag)

    pad_frames  = int(padding_ms / frame_ms)
    start_idx   = max(0, first - pad_frames)
    end_idx     = min(len(frames)-1, last + pad_frames)

    trimmed = b''.join(f.bytes for f in frames[start_idx:end_idx+1])
    return trimmed


def reduce_noise_pcm(
    pcm_bytes: bytes,
    sr: int,
    ns_level: int = 2,
    agc_dbfs: int = 3
) -> bytes:
    if sr != 16000:
        data = np.frombuffer(pcm_bytes, dtype=np.int16)
        pcm_bytes, sr = preprocess_wav(data, sr)
    processor = AudioProcessor(agc_dbfs, ns_level)
    frame_size = int(sr * 10 / 1000) * 2
    output = []
    for i in range(0, len(pcm_bytes), frame_size):
        frame = pcm_bytes[i:i + frame_size]
        if len(frame) < frame_size:
            break
        result = processor.Process10ms(frame)
        output.append(result.audio)
    return b''.join(output)


def trim_silence_file(
    input_wav: str,
    output_wav: str,
    aggressiveness: int = 2,
    frame_ms: int = 20,
    padding_ms: int = 300
):
    audio, sr = sf.read(input_wav, dtype='int16')
    pcm = audio.tobytes()
    if sr not in (8000, 16000, 32000, 48000):
        pcm, sr = preprocess_wav(pcm, sr)
    
    trimmed = trim_silence_pcm(pcm, sr, aggressiveness, frame_ms, padding_ms) 
    audio_out = np.frombuffer(trimmed, dtype=np.int16)
    sf.write(output_wav, audio_out, sr)

def trim_bytes(
    raw_wav: bytes,
    aggressiveness: int = 2,
    frame_ms: int = 30,
    padding_ms: int = 300
) -> bytes:
    
    buf = io.BytesIO(raw_wav)
    data, sr = sf.read(buf, dtype='int16')
    pcm_bytes = data.tobytes()
    trimmed = trim_silence_pcm(pcm_bytes, sr, aggressiveness, frame_ms, padding_ms)

    out_buf = io.BytesIO()
    out_arr = np.frombuffer(trimmed, dtype=np.int16)
    sf.write(out_buf, out_arr, sr, format='WAV')
    return out_buf.getvalue()

def reduce_noise_file(
    input_wav: str,
    output_wav: str,
    ns_level: int = 2,
    agc_dbfs: int = 3
) -> None:
    temp_path = 'temp.wav'
    convert_to_hd(input_wav, temp_path)
    data, sr = sf.read(temp_path, dtype='int16')
    pcm_bytes = data.tobytes()
    reduced = reduce_noise_pcm(pcm_bytes, sr, ns_level, agc_dbfs)
    out_arr = np.frombuffer(reduced, dtype=np.int16)
    sf.write(output_wav, out_arr, sr)
    if os.path.exists(temp_path):
         os.remove(temp_path)


def reduce_noise_bytes(
    raw_wav: bytes,
    ns_level: int = 2,
    agc_dbfs: int = 3
) -> bytes:
    buf = io.BytesIO(raw_wav)
    data, sr = sf.read(buf, dtype='int16')
    reduced = reduce_noise_pcm(data.tobytes(), sr, ns_level, agc_dbfs)
    out_buf = io.BytesIO()
    sf.write(out_buf, np.frombuffer(reduced, dtype=np.int16), sr, format='WAV')
    return out_buf.getvalue()