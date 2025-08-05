# VoiceAgentQuickStart

A comprehensive **start** for building real-time, low-latency Voice AI agents in Python.

## 🚀 Features

* **Audio I/O**: 10 ms frame capture & playback with PyAudio
* **Silence Trimming**: Dynamic, frame-accurate WebRTC VAD
* **Noise Reduction**: WebRTC AudioProcessing with auto-gain
* **ASR**: Streaming transcription with Hugging face Wav2Vec2
* **Conversational AI**: Persona + global rules + NVIDIA Llama 3/OpenAI integration
* **TTS**: Natural, fast speech via Coqui FastPitch or Tacotron2
* **Modular**: File, byte-stream and live modes

## ⚙️ Optimization Status

* **Audio processing & STT**: fully optimized for low-latency (<100 ms) and minimal CPU overhead.
* **Agent response generation & TTS**: functional but pending performance and quality optimizations; improvements underway and will be released soon.
* **Current Performance (example benchmark):** On a mid-range laptop CPU, STT runs at \~0.33× real-time with <250 ms of pre-processing; On modern GPUs or high-end CPUs, real-time factors <0.1× are readily attainable, showcasing the code’s potential for sub-100 ms end-to-end latency.

## 📦 Installation

```bash
git clone https://github.com/DhavalTalaviya/VoiceAgent-Quick-Start.git
cd VoiceAgentQuickStart
pip install -e .
```

## 🔧 Usage

### Live Agent (CLI)

Set your API key in a `.env` file:

```ini
API_KEY=sk-...
```

Run the live agent:

```bash
voiceagent-quickstart live
```
