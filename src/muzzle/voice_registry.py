from __future__ import annotations

import json
import shutil
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path

from .domain import VoiceRecord


class VoiceRegistry:
    def __init__(self, root: Path):
        self.root = root

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def list_records(self) -> list[VoiceRecord]:
        self.ensure()
        records: list[VoiceRecord] = []
        for metadata_path in sorted(self.root.glob("*/metadata.json")):
            try:
                records.append(self._load_record(metadata_path.parent))
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
        return records

    def get(self, voice_id: str) -> VoiceRecord | None:
        if voice_id == "default":
            return None
        path = self.root / voice_id
        if not path.is_dir():
            return None
        return self._load_record(path)

    def create(self, *, name: str, filename: str, content: bytes) -> VoiceRecord:
        self.ensure()
        voice_id = uuid.uuid4().hex
        voice_dir = self.root / voice_id
        voice_dir.mkdir(parents=False)

        safe_filename = Path(filename or "reference.wav").name
        reference_audio_path = voice_dir / safe_filename
        reference_audio_path.write_bytes(content)

        wav_info = _inspect_wav(reference_audio_path)
        created_at = datetime.now(timezone.utc).isoformat()
        metadata = {
            "voice_id": voice_id,
            "name": name,
            "source_filename": safe_filename,
            "created_at": created_at,
            "sample_rate": wav_info.get("sample_rate"),
            "duration_seconds": wav_info.get("duration_seconds"),
        }
        (voice_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
        return self._load_record(voice_dir)

    def delete(self, voice_id: str) -> bool:
        if voice_id == "default":
            return False
        path = self.root / voice_id
        if not path.exists():
            return False
        shutil.rmtree(path)
        return True

    def _load_record(self, voice_dir: Path) -> VoiceRecord:
        metadata = json.loads((voice_dir / "metadata.json").read_text())
        reference_audio_path = voice_dir / metadata["source_filename"]
        return VoiceRecord(
            voice_id=metadata["voice_id"],
            name=metadata["name"],
            source_filename=metadata["source_filename"],
            reference_audio_path=reference_audio_path,
            directory=voice_dir,
            created_at=metadata["created_at"],
            sample_rate=metadata.get("sample_rate"),
            duration_seconds=metadata.get("duration_seconds"),
        )


def _inspect_wav(path: Path) -> dict[str, float | int]:
    try:
        with wave.open(str(path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            frames = wav_file.getnframes()
            duration = frames / sample_rate if sample_rate else 0.0
            return {"sample_rate": sample_rate, "duration_seconds": duration}
    except (wave.Error, OSError, EOFError):
        return {}

