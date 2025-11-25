#!/usr/bin/env python3
from __future__ import annotations
import argparse
import asyncio
import threading

from client.shared_audio import get_shared_audio
from client.monitor_engine import MonitorConfig, MonitorEngine


def main():
    ap = argparse.ArgumentParser(description="Standalone Marvin monitor (debug)")
    ap.add_argument("--broker-url", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--voice", default=None)
    ap.add_argument("--voice-mode", default="standard")
    args = ap.parse_args()

    player, playback_mute = get_shared_audio()
    cfg = MonitorConfig(
        broker_url=args.broker_url,
        session=args.session,
        voice_id=args.voice,
        voice_mode=args.voice_mode,
        camera_index=args.camera_index,
        mic_index=None,
    )
    engine = MonitorEngine(cfg, player, playback_mute)

    async def loop():
        while not engine.stop_flag:
            await engine.step()
            await asyncio.sleep(0.03)
        engine.cleanup()

    asyncio.run(loop())


if __name__ == "__main__":
    main()
