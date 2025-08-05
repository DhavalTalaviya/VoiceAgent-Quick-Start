from dotenv import load_dotenv
load_dotenv()

import argparse
import os
from .agent import Agent
from .tts import TTSAdapter
from .cli import listen_and_process

def main():
    p = argparse.ArgumentParser(prog="voiceagent-quickstart")
    subs = p.add_subparsers(dest="cmd")

    live = subs.add_parser("live", help="Run live voice agent")
    live.add_argument("--api-key", default=os.environ["API_KEY"])
    live.add_argument("--agent-model", default="nvidia/llama-3.3-nemotron-super-49b-v1.5")
    live.add_argument("--tts-model", default="tts_models/en/ljspeech/fast_pitch")
    live.add_argument("--tts-speed", type=float, default=1.2)
    live.add_argument("--stt-model", default="facebook/wav2vec2-large-960h-lv60-self")
    live.add_argument("--device", default="cpu")
    live.add_argument("--sr", type=int, default=16000)
    live.add_argument("--frame-ms", type=int, default=10)
    live.add_argument("--silence-threshold-ms", type=int, default=500)
    live.add_argument("--ns-level", type=int, default=2)
    live.add_argument("--agc-dbfs", type=int, default=3)
    live.add_argument("--vad-agg", type=int, default=2)

    args = p.parse_args()
    if args.cmd == "live":
        agent = Agent(model=args.agent_model, api_key=args.api_key,
                      base_url="https://integrate.api.nvidia.com/v1")
        tts = TTSAdapter(args.tts_model, speed=args.tts_speed)
        listen_and_process(
            sr=args.sr,
            frame_ms=args.frame_ms,
            silence_threshold_ms=args.silence_threshold_ms,
            ns_level=args.ns_level,
            agc_dbfs=args.agc_dbfs,
            aggressiveness=args.vad_agg,
            stt_model=args.stt_model,
            device=args.device,
            agent=agent,
            tts=tts
        )

if __name__ == "__main__":
    main()