from __future__ import annotations

import argparse
import asyncio
import shlex
import shutil
import subprocess
import sys
import time

from muzzle.adapters.chatterbox import ChatterboxTTSAdapter
from muzzle.config import Settings
from muzzle.domain import TTSOptions, TTS_QUALITY_PROFILES


def configure_torch_threads(args: argparse.Namespace) -> None:
    if args.torch_threads is None and args.torch_interop_threads is None:
        return

    import torch

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    if args.torch_interop_threads is not None:
        torch.set_num_interop_threads(args.torch_interop_threads)


def build_options(args: argparse.Namespace) -> TTSOptions:
    profile = dict(TTS_QUALITY_PROFILES[args.quality])
    if args.chunk_tokens is not None:
        profile["chunk_tokens"] = args.chunk_tokens
    if args.crossfade_ms is not None:
        profile["crossfade_ms"] = args.crossfade_ms

    return TTSOptions(
        request_id="cpu-example",
        text=args.text,
        voice_id="default",
        quality=args.quality,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        max_gen_len=args.max_gen_len,
        **profile,
    )


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
    configure_torch_threads(args)

    settings = Settings(tts_device="cpu", model_backend="real")
    adapter = ChatterboxTTSAdapter(settings)
    player_proc: subprocess.Popen[bytes] | None = None
    player_sample_rate: int | None = None
    pcm_file = open(args.save_pcm, "wb") if args.save_pcm else None

    try:
        load_started = time.perf_counter()
        await adapter.start()
        load_seconds = time.perf_counter() - load_started
        print(f"loaded device=cpu seconds={load_seconds:.2f}", flush=True)

        options = build_options(args)
        print(
            "profile "
            f"quality={options.quality} "
            f"chunk_tokens={options.chunk_tokens} "
            f"crossfade_ms={options.crossfade_ms:g} "
            f"player={args.player} "
            f"latency_ms={args.latency_ms}",
            flush=True,
        )
        synth_started = time.perf_counter()
        first_chunk_seconds: float | None = None
        chunk_count = 0
        total_audio_seconds = 0.0
        total_bytes = 0

        async for chunk in adapter.synthesize_stream(options, voice=None):
            elapsed = time.perf_counter() - synth_started
            if first_chunk_seconds is None:
                first_chunk_seconds = elapsed

            chunk_audio_seconds = len(chunk.audio) / (2 * chunk.sample_rate)
            total_audio_seconds += chunk_audio_seconds
            total_bytes += len(chunk.audio)
            chunk_count += 1

            if player_proc is None and args.player != "none":
                command = build_player_command(args.player, chunk.sample_rate, args.latency_ms)
                if command is None:
                    print("pw-cat not found; continuing without playback.", file=sys.stderr, flush=True)
                else:
                    player_sample_rate = chunk.sample_rate
                    print(f"player={shlex.join(command)}", flush=True)
                    player_proc = subprocess.Popen(command, stdin=subprocess.PIPE)
            elif player_sample_rate is not None and chunk.sample_rate != player_sample_rate:
                raise RuntimeError(f"sample rate changed from {player_sample_rate} to {chunk.sample_rate}")

            if player_proc is not None and player_proc.stdin is not None:
                try:
                    player_proc.stdin.write(chunk.audio)
                    player_proc.stdin.flush()
                except BrokenPipeError:
                    print("Player stopped accepting audio; continuing without playback.", file=sys.stderr)
                    player_proc = None

            if pcm_file is not None:
                pcm_file.write(chunk.audio)

            speed = total_audio_seconds / elapsed if elapsed > 0 else 0.0
            print(
                "chunk="
                f"{chunk.index} elapsed={elapsed:.2f}s "
                f"chunk_audio={chunk_audio_seconds:.2f}s "
                f"total_audio={total_audio_seconds:.2f}s "
                f"speed={speed:.2f}x "
                f"tokens={chunk.generated_tokens} "
                f"final={str(chunk.is_final).lower()}",
                flush=True,
            )

        if pcm_file is not None:
            pcm_file.flush()

        synth_seconds = time.perf_counter() - synth_started
        speed = total_audio_seconds / synth_seconds if synth_seconds > 0 else 0.0
        rtf = synth_seconds / total_audio_seconds if total_audio_seconds > 0 else 0.0
        first = first_chunk_seconds if first_chunk_seconds is not None else 0.0
        print(
            "summary "
            f"chunks={chunk_count} bytes={total_bytes} "
            f"first_chunk={first:.2f}s "
            f"audio={total_audio_seconds:.2f}s "
            f"wall={synth_seconds:.2f}s "
            f"speed={speed:.2f}x "
            f"rtf={rtf:.2f}",
            flush=True,
        )
        return 0
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 1
    finally:
        stop_player(player_proc)
        if pcm_file is not None:
            pcm_file.close()
        await adapter.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Chatterbox streaming TTS directly on CPU, play it, and print timing metrics."
    )
    parser.add_argument("--text", default="Hello from Muzzle, streaming text to speech on the CPU.")
    parser.add_argument("--quality", choices=["fast", "balanced", "high", "cpu-smooth"], default="cpu-smooth")
    parser.add_argument("--player", choices=["auto", "pw-cat", "none"], default="auto")
    parser.add_argument("--latency-ms", type=int, default=120)
    parser.add_argument("--save-pcm", default=None, help="Write generated raw mono pcm_s16le bytes to this file.")
    parser.add_argument("--chunk-tokens", type=int, default=None)
    parser.add_argument("--crossfade-ms", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--max-gen-len", type=int, default=1000)
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--torch-interop-threads", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    try:
        raise SystemExit(asyncio.run(run(parse_args())))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
