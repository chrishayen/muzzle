from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import shutil
import subprocess
import sys

import websockets


def build_player_command(player: str, sample_rate: int, latency_ms: int) -> list[str] | None:
    if player == "none":
        return None

    pw_cat = shutil.which("pw-cat")
    if not pw_cat:
        if player == "pw-cat":
            raise RuntimeError("pw-cat was requested but was not found in PATH")
        return None

    return [
        pw_cat,
        "--playback",
        "--raw",
        "--format",
        "s16",
        "--rate",
        str(sample_rate),
        "--channels",
        "1",
        "--latency",
        f"{latency_ms}ms",
        "-",
    ]


def stop_player(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None:
        return
    try:
        if proc.stdin is not None:
            proc.stdin.close()
    except BrokenPipeError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.terminate()


async def run(args: argparse.Namespace) -> int:
    headers = {}
    token = args.auth_token or os.getenv("MUZZLE_AUTH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request_id = "example-tts"
    message = {
        "type": "tts.speak",
        "request_id": request_id,
        "text": args.text,
        "voice_id": args.voice_id,
        "quality": args.quality,
    }

    if args.chunk_tokens is not None:
        message["chunk_tokens"] = args.chunk_tokens
    if args.crossfade_ms is not None:
        message["crossfade_ms"] = args.crossfade_ms

    player_proc: subprocess.Popen[bytes] | None = None
    player_sample_rate: int | None = None
    pcm_file = open(args.save_pcm, "wb") if args.save_pcm else None
    total_bytes = 0
    chunk_count = 0

    try:
        async with websockets.connect(args.url, additional_headers=headers) as ws:
            created = await ws.recv()
            if isinstance(created, bytes):
                raise RuntimeError("expected session.created JSON, got binary frame")
            print(f"< {created}", flush=True)

            await ws.send(json.dumps(message))
            print(f"> {json.dumps(message)}", flush=True)

            while True:
                raw = await ws.recv()
                if isinstance(raw, bytes):
                    raise RuntimeError(f"unexpected binary frame without metadata: {len(raw)} bytes")

                event = json.loads(raw)
                event_type = event.get("type")
                print(f"< {json.dumps(event, separators=(',', ':'))}", flush=True)

                if event_type == "tts.audio.chunk":
                    audio = await ws.recv()
                    if not isinstance(audio, bytes):
                        raise RuntimeError("expected binary audio frame after tts.audio.chunk")
                    expected = event.get("bytes")
                    if expected is not None and expected != len(audio):
                        raise RuntimeError(f"audio byte mismatch: metadata={expected}, actual={len(audio)}")

                    sample_rate = int(event["sample_rate"])
                    if player_proc is None and args.player != "none":
                        command = build_player_command(args.player, sample_rate, args.latency_ms)
                        if command is None:
                            print("pw-cat not found; continuing without playback.", file=sys.stderr, flush=True)
                        else:
                            player_sample_rate = sample_rate
                            print(f"player={shlex.join(command)}", flush=True)
                            player_proc = subprocess.Popen(command, stdin=subprocess.PIPE)
                    elif player_sample_rate is not None and sample_rate != player_sample_rate:
                        raise RuntimeError(f"sample rate changed from {player_sample_rate} to {sample_rate}")

                    if player_proc is not None and player_proc.stdin is not None:
                        try:
                            player_proc.stdin.write(audio)
                            player_proc.stdin.flush()
                        except BrokenPipeError:
                            print("Player stopped accepting audio; continuing without playback.", file=sys.stderr)
                            player_proc = None

                    if pcm_file is not None:
                        pcm_file.write(audio)
                        pcm_file.flush()

                    total_bytes += len(audio)
                    chunk_count += 1

                elif event_type == "tts.done":
                    if event.get("request_id") == request_id:
                        print(f"done chunks={chunk_count} bytes={total_bytes}", flush=True)
                        return 0 if event.get("status") == "completed" else 1

                elif event_type == "error":
                    return 1
    finally:
        stop_player(player_proc)
        if pcm_file is not None:
            pcm_file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call the muzzle streaming TTS API and play returned PCM through PipeWire."
    )
    parser.add_argument("--url", default="ws://127.0.0.1:8000/v1/sessions")
    parser.add_argument("--text", default="Hello from muzzle, streaming through PipeWire on Hyprland.")
    parser.add_argument("--voice-id", default="default")
    parser.add_argument("--quality", choices=["fast", "balanced", "high", "cpu-smooth"], default="balanced")
    parser.add_argument("--auth-token", default=None)
    parser.add_argument("--player", choices=["auto", "pw-cat", "none"], default="auto")
    parser.add_argument("--latency-ms", type=int, default=120)
    parser.add_argument("--save-pcm", default=None, help="Write received raw pcm_s16le bytes to this file.")
    parser.add_argument("--chunk-tokens", type=int, default=None)
    parser.add_argument("--crossfade-ms", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    try:
        raise SystemExit(asyncio.run(run(parse_args())))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
