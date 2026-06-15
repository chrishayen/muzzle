from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import shutil
import sys
from pathlib import Path

import websockets

SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2
CHANNELS = 1


def pipewire_record_command() -> list[str]:
    pw_cat = shutil.which("pw-cat")
    if not pw_cat:
        raise RuntimeError("pw-cat was not found in PATH")
    return [
        pw_cat,
        "--record",
        "--raw",
        "--format",
        "s16",
        "--rate",
        str(SAMPLE_RATE),
        "--channels",
        str(CHANNELS),
        "-",
    ]


async def receive_events(ws) -> None:
    while True:
        raw = await ws.recv()
        if isinstance(raw, bytes):
            print(f"< binary {len(raw)} bytes", flush=True)
            continue
        event = json.loads(raw)
        event_type = event.get("type")
        if event_type in {"stt.partial", "stt.final", "input_audio.speech_started", "input_audio.speech_stopped", "error"}:
            print(f"< {json.dumps(event, separators=(',', ':'))}", flush=True)


async def stream_file(ws, path: Path, frame_bytes: int, realtime: bool) -> None:
    frame_seconds = frame_bytes / (SAMPLE_RATE * SAMPLE_WIDTH)
    with path.open("rb") as pcm:
        while chunk := pcm.read(frame_bytes):
            if len(chunk) % SAMPLE_WIDTH:
                chunk = chunk[:-1]
            if chunk:
                await ws.send(chunk)
            if realtime:
                await asyncio.sleep(frame_seconds)


async def stream_pipewire(ws, frame_bytes: int, duration_seconds: float | None) -> None:
    command = pipewire_record_command()
    print(f"recorder={shlex.join(command)}", flush=True)
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdout is not None

    deadline = None if duration_seconds is None else asyncio.get_running_loop().time() + duration_seconds
    try:
        while True:
            if deadline is not None and asyncio.get_running_loop().time() >= deadline:
                break
            chunk = await proc.stdout.read(frame_bytes)
            if not chunk:
                break
            if len(chunk) % SAMPLE_WIDTH:
                chunk = chunk[:-1]
            if chunk:
                await ws.send(chunk)
    finally:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=2)
        except asyncio.TimeoutError:
            proc.kill()


async def run(args: argparse.Namespace) -> int:
    headers = {}
    token = args.auth_token or os.getenv("MUZZLE_AUTH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    frame_bytes = int(SAMPLE_RATE * SAMPLE_WIDTH * args.frame_ms / 1000)
    frame_bytes -= frame_bytes % SAMPLE_WIDTH

    async with websockets.connect(args.url, additional_headers=headers) as ws:
        created = await ws.recv()
        print(f"< {created}", flush=True)

        receiver = asyncio.create_task(receive_events(ws))
        try:
            if args.input_pcm:
                await stream_file(ws, Path(args.input_pcm), frame_bytes, realtime=not args.no_realtime)
            else:
                duration = None if args.duration_seconds <= 0 else args.duration_seconds
                await stream_pipewire(ws, frame_bytes, duration)

            await ws.send(json.dumps({"type": "input_audio.commit"}))
            await asyncio.sleep(args.final_wait_seconds)
        finally:
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream microphone PCM from PipeWire to the muzzle STT WebSocket API."
    )
    parser.add_argument("--url", default="ws://127.0.0.1:8000/v1/sessions")
    parser.add_argument("--auth-token", default=None)
    parser.add_argument("--duration-seconds", type=float, default=10.0, help="0 means record until Ctrl+C.")
    parser.add_argument("--frame-ms", type=int, default=100)
    parser.add_argument("--input-pcm", default=None, help="Read raw 16 kHz mono pcm_s16le from a file instead of mic.")
    parser.add_argument("--no-realtime", action="store_true", help="Do not sleep between file chunks.")
    parser.add_argument("--final-wait-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    try:
        raise SystemExit(asyncio.run(run(parse_args())))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
