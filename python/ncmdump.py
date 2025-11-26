"""Python implementation of NetEase NCM decryption."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import multiprocessing
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from Crypto.Cipher import AES
from mutagen.flac import FLAC, Picture
from mutagen.id3 import APIC, ID3, TALB, TIT2, TPE1
from mutagen.mp3 import MP3

CORE_KEY = b"hzHRAmso5kInbaxW"
MODIFY_KEY = b"#14ljk_!\\]&0U<'("
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
BUFFER_SIZE = 0x8000


class NCMFile:
    def __init__(self, path: Path):
        self.path = path
        self.key_box: List[int] = []
        self.metadata: Optional[dict] = None
        self.cover_data: bytes = b""
        self.format: Optional[str] = None

    @staticmethod
    def _aes_ecb_decrypt(key: bytes, data: bytes) -> bytes:
        cipher = AES.new(key, AES.MODE_ECB)
        decrypted = cipher.decrypt(data)
        padding = decrypted[-1]
        if 0 < padding <= AES.block_size:
            return decrypted[:-padding]
        return decrypted

    @staticmethod
    def _read_int(file_obj) -> int:
        data = file_obj.read(4)
        if len(data) != 4:
            raise ValueError("Unexpected EOF while reading integer")
        return int.from_bytes(data, byteorder="little")

    def _build_key_box(self, key: bytes) -> None:
        box = list(range(256))
        swap = last_byte = key_offset = 0
        for i in range(256):
            swap = box[i]
            c = (swap + last_byte + key[key_offset]) & 0xFF
            key_offset = (key_offset + 1) % len(key)
            box[i] = box[c]
            box[c] = swap
            last_byte = c
        self.key_box = box

    def _parse_metadata(self, meta_data: bytes) -> None:
        swap_data = bytes(b ^ 0x63 for b in meta_data)
        decoded = base64.b64decode(swap_data[22:])
        decrypted = self._aes_ecb_decrypt(MODIFY_KEY, decoded)
        json_bytes = decrypted[6:]
        try:
            meta = json.loads(json_bytes.decode("utf-8"))
        except json.JSONDecodeError:
            self.metadata = None
            return

        artists = []
        for artist_entry in meta.get("artist", []):
            if isinstance(artist_entry, list) and artist_entry:
                artists.append(str(artist_entry[0]))
        self.metadata = {
            "name": meta.get("musicName", ""),
            "album": meta.get("album", ""),
            "artist": "/".join(artists),
        }

    def _parse_cover(self, file_obj, frame_length: int) -> None:
        cover_length = self._read_int(file_obj)
        if cover_length > 0:
            self.cover_data = file_obj.read(cover_length)
        else:
            self.cover_data = b""
        remaining = frame_length - cover_length
        if remaining > 0:
            file_obj.seek(remaining, 1)

    def _parse_headers(self, file_obj) -> None:
        header = file_obj.read(8)
        if header != b"CTENFDAM":
            raise ValueError(f"{self.path} is not a valid NCM file")

        file_obj.seek(2, 1)

        key_length = self._read_int(file_obj)
        key_data = bytearray(file_obj.read(key_length))
        if len(key_data) != key_length:
            raise ValueError("Unexpected EOF while reading key data")
        key_data = bytes(b ^ 0x64 for b in key_data)

        decrypted_key = self._aes_ecb_decrypt(CORE_KEY, key_data)
        self._build_key_box(decrypted_key[17:])

        meta_length = self._read_int(file_obj)
        if meta_length > 0:
            raw_meta = file_obj.read(meta_length)
            if len(raw_meta) != meta_length:
                raise ValueError("Unexpected EOF while reading metadata")
            self._parse_metadata(raw_meta)
        else:
            self.metadata = None

        file_obj.seek(5, 1)
        cover_frame_length = self._read_int(file_obj)
        self._parse_cover(file_obj, cover_frame_length)

    def _decrypt_chunk(self, data: bytearray) -> None:
        for i in range(len(data)):
            j = (i + 1) & 0xFF
            data[i] ^= self.key_box[(self.key_box[j] + self.key_box[(self.key_box[j] + j) & 0xFF]) & 0xFF]

    def _target_path(self, output_dir: Optional[Path], suffix: str, base_dir: Path) -> Path:
        relative = self.path.relative_to(base_dir)
        destination_dir = output_dir if output_dir else base_dir
        return destination_dir.joinpath(relative).with_suffix(suffix)

    def dump(self, output_dir: Optional[Path] = None, base_dir: Optional[Path] = None) -> Path:
        base_dir = base_dir or self.path.parent
        with self.path.open("rb") as f:
            self._parse_headers(f)

            out_file: Optional[Path] = None
            temp_path: Optional[Path] = None

            while True:
                chunk = bytearray(f.read(BUFFER_SIZE))
                if not chunk:
                    break
                self._decrypt_chunk(chunk)

                if out_file is None:
                    if chunk.startswith(b"ID3"):
                        suffix = ".mp3"
                        self.format = "mp3"
                    else:
                        suffix = ".flac"
                        self.format = "flac"
                    out_path = self._target_path(output_dir, suffix, base_dir)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    temp_path = out_path
                    out_file = out_path.open("wb")

                out_file.write(chunk)

            if out_file is None or temp_path is None:
                raise ValueError("No audio data decrypted")

        self._apply_metadata(temp_path)
        return temp_path

    def _apply_metadata(self, output_path: Path) -> None:
        if not self.metadata:
            return

        cover_mime = "image/png" if self.cover_data.startswith(PNG_MAGIC) else "image/jpeg"

        if self.format == "mp3":
            audio = MP3(output_path, ID3=ID3)
            try:
                tags = audio.tags
                if tags is None:
                    audio.add_tags()
            except Exception:
                audio.add_tags()
            audio.tags.add(TIT2(encoding=3, text=self.metadata.get("name", "")))
            audio.tags.add(TPE1(encoding=3, text=self.metadata.get("artist", "")))
            audio.tags.add(TALB(encoding=3, text=self.metadata.get("album", "")))
            if self.cover_data:
                audio.tags.add(APIC(encoding=3, mime=cover_mime, type=3, desc="Cover", data=self.cover_data))
            audio.save(v2_version=3)
        elif self.format == "flac":
            audio = FLAC(output_path)
            audio["title"] = [self.metadata.get("name", "")]
            audio["artist"] = [self.metadata.get("artist", "")]
            audio["album"] = [self.metadata.get("album", "")]
            if self.cover_data:
                picture = Picture()
                picture.data = self.cover_data
                picture.type = 3
                picture.mime = cover_mime
                audio.add_picture(picture)
            audio.save()


def iter_ncm_files(directory: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*.ncm" if recursive else "*.ncm"
    yield from (p for p in directory.glob(pattern) if p.is_file())


def process_file(file_path: Path, output_dir: Optional[Path], remove_source: bool, base_dir: Optional[Path]):
    ncm = NCMFile(file_path)
    target = ncm.dump(output_dir=output_dir, base_dir=base_dir)
    if remove_source:
        try:
            file_path.unlink()
        except OSError as exc:
            print(f"Failed to remove {file_path}: {exc}")
    return target


def collect_files(args) -> List[Tuple[Path, Path]]:
    results: List[Tuple[Path, Path]] = []
    if args.directory:
        directory = Path(args.directory)
        files = list(iter_ncm_files(directory, args.recursive))
        results.extend((f, directory) for f in files)
    for file_path in args.files:
        path = Path(file_path)
        results.append((path, path.parent))
    return results


def main():
    parser = argparse.ArgumentParser(description="Decrypt NetEase NCM files to mp3 or flac.")
    parser.add_argument("files", nargs="*", help="Individual .ncm files to convert")
    parser.add_argument("-d", "--directory", help="Directory containing .ncm files")
    parser.add_argument("-r", "--recursive", action="store_true", help="Recursively search the directory")
    parser.add_argument("-o", "--output", type=Path, help="Output directory for converted files")
    parser.add_argument("-m", "--remove", action="store_true", help="Remove source files after successful conversion")
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of worker threads for concurrent conversions",
    )

    args = parser.parse_args()

    targets = collect_files(args)
    if not targets:
        parser.error("No input files provided")

    worker_count = max(1, args.jobs)
    mp_context = multiprocessing.get_context("spawn")

    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count, mp_context=mp_context) as executor:
        future_map = {
            executor.submit(process_file, path, args.output, args.remove, base): path
            for path, base in targets
        }
        for future in concurrent.futures.as_completed(future_map):
            source_path = future_map[future]
            try:
                target_path = future.result()
                print(f"Converted {source_path} -> {target_path}")
            except Exception as exc:
                print(f"Failed to convert {source_path}: {exc}")


if __name__ == "__main__":
    main()
