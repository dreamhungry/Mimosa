#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download sherpa-onnx ASR model for Mimosa."""

import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path


# ModelScope mirror (faster in China) or GitHub release
MODELSCOPE_URL = (
    "https://www.modelscope.cn/models/zhaochaoqun/sherpa-onnx-asr-models"
    "/resolve/master/sherpa-onnx-zipformer-zh-en-2023-11-22.tar.bz2"
)
GITHUB_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
    "/sherpa-onnx-zipformer-zh-en-2023-11-22.tar.bz2"
)
MODEL_DIR_NAME = "sherpa-onnx-zipformer-zh-en-2023-11-22"
TARGET_DIR = Path(__file__).parent.parent / "models"


def download_model():
    """Download and extract the ASR model."""
    target_path = TARGET_DIR / MODEL_DIR_NAME
    if target_path.exists():
        print(f"Model already exists at: {target_path}")
        return

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    tmp_file = TARGET_DIR / "model_download.tar.bz2"

    # Try ModelScope first (faster in China), then GitHub
    urls = [MODELSCOPE_URL, GITHUB_URL]
    for url in urls:
        print(f"Trying: {url}")
        try:
            urllib.request.urlretrieve(url, str(tmp_file), reporthook=_progress)
            print("\nExtracting...")
            with tarfile.open(str(tmp_file), "r:bz2") as tar:
                tar.extractall(path=str(TARGET_DIR))
            print(f"Done! Model extracted to: {target_path}")
            return
        except Exception as e:
            print(f"\nFailed: {e}")
            if tmp_file.exists():
                tmp_file.unlink()
            continue

    print("\nAll download sources failed.")
    print("Please download manually from:")
    print(f"  {GITHUB_URL}")
    print(f"Extract to: {TARGET_DIR}")


def _progress(block_num, block_size, total_size):
    """Show download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        mb_done = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\r  {percent:.1f}% ({mb_done:.1f}/{mb_total:.1f} MB)", end="", flush=True)


if __name__ == "__main__":
    download_model()
