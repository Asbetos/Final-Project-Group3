"""
No-mmap safetensors loader for memory-constrained systems.

`safetensors.safe_open()` calls `mmap()` on the entire shard file. On systems
with limited RAM and no swap, the kernel's heuristic overcommit refuses the
mmap outright with `Cannot allocate memory` even though mmap pages are backed
by the file on disk (not by RAM). This module provides a drop-in replacement
that reads tensors via `os.pread()` instead, which has no virtual-memory
allocation cost.

Usage:
    from safetensors_nommap import patch_transformers_safetensors_loader
    patch_transformers_safetensors_loader()
    # ...now AutoModelForCausalLM.from_pretrained will use the non-mmap path
"""

from __future__ import annotations

import json
import os
import struct

import numpy as np
import torch


# safetensors dtype string → torch dtype
_TORCH_DTYPE_MAP = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}

# safetensors dtype string → numpy dtype (BF16 has no numpy equivalent)
_NP_DTYPE_MAP = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}


def _raw_to_tensor(raw: bytes, dtype_str: str, shape) -> torch.Tensor:
    """Convert raw safetensors bytes → torch.Tensor with the given dtype/shape."""
    if dtype_str == "BF16":
        # NumPy has no BF16, so read as int16 and bit-reinterpret as bfloat16.
        arr = np.frombuffer(raw, dtype=np.int16).copy()
        tensor = torch.from_numpy(arr).view(torch.bfloat16)
    elif dtype_str in _NP_DTYPE_MAP:
        arr = np.frombuffer(raw, dtype=_NP_DTYPE_MAP[dtype_str]).copy()
        tensor = torch.from_numpy(arr)
        torch_dtype = _TORCH_DTYPE_MAP[dtype_str]
        if tensor.dtype != torch_dtype:
            tensor = tensor.to(torch_dtype)
    else:
        raise NotImplementedError(
            f"NoMmapSafeOpen: unsupported safetensors dtype '{dtype_str}'"
        )

    if shape:
        tensor = tensor.reshape(shape)
    return tensor


class _NoMmapSlice:
    """PySafeSlice-compatible object that materializes the full tensor on
    `__getitem__`. Implements the narrow subset of the safetensors slice
    interface that transformers actually uses:

      - get_dtype() / get_shape()
      - slice[...] to materialize the full tensor as a torch.Tensor
    """

    __slots__ = ("_parent", "_name", "_dtype", "_shape", "_offset", "_size")

    def __init__(self, parent: "NoMmapSafeOpen", name: str):
        entry = parent._meta[name]
        self._parent = parent
        self._name = name
        self._dtype = entry["dtype"]
        self._shape = list(entry["shape"])
        begin, end = entry["data_offsets"]
        self._offset = parent._data_offset + begin
        self._size = end - begin

    def get_dtype(self) -> str:
        return self._dtype

    def get_shape(self) -> list:
        return list(self._shape)

    def _materialize(self) -> torch.Tensor:
        # os.pread is thread-safe (it does not share file position), so
        # loader threads can read concurrently without a lock. BUT: Linux
        # caps a single read() / pread() syscall at 0x7ffff000 bytes
        # (~2 GB - 4 KB), so tensors larger than that require chunked reads.
        # Gemma 4 31B's embed_tokens (262,144 × 5,376 × bf16 = 2.8 GB) hits
        # this limit.
        CHUNK = 1 << 30  # 1 GiB per pread call — well under the 2 GiB cap
        fd = self._parent._fd
        if self._size <= CHUNK:
            raw = os.pread(fd, self._size, self._offset)
        else:
            buf = bytearray(self._size)
            view = memoryview(buf)
            done = 0
            while done < self._size:
                n = min(CHUNK, self._size - done)
                chunk = os.pread(fd, n, self._offset + done)
                if not chunk:
                    raise IOError(
                        f"Unexpected EOF reading '{self._name}' at offset "
                        f"{self._offset + done} after {done}/{self._size} bytes"
                    )
                view[done:done + len(chunk)] = chunk
                done += len(chunk)
            raw = buf  # bytearray is accepted by np.frombuffer
        if len(raw) != self._size:
            raise IOError(
                f"Short read for tensor '{self._name}': "
                f"got {len(raw)} of {self._size} bytes"
            )
        return _raw_to_tensor(raw, self._dtype, self._shape)

    def __getitem__(self, key):
        tensor = self._materialize()
        if key is ...:
            return tensor
        # Fall back to torch indexing for any slice/index argument
        return tensor[key]


class NoMmapSafeOpen:
    """Drop-in replacement for `safetensors.safe_open` that reads tensors via
    `os.pread()` instead of `mmap()`. Safe to use anywhere `safe_open` appears.

    Supports both forms of usage seen in transformers:
        # Context manager:
        with safe_open(file, framework="pt") as f:
            ...

        # Explicit open / close:
        f = safe_open(file, framework="pt", device="cpu")
        ...
        f.__exit__(None, None, None)
    """

    def __init__(self, filename: str, framework: str = "pt", device: str = "cpu"):
        if framework != "pt":
            raise NotImplementedError(
                f"NoMmapSafeOpen only supports framework='pt' (got {framework!r})"
            )
        self._filename = filename
        self._device = device
        self._fd = os.open(filename, os.O_RDONLY)

        # Parse the safetensors header:
        #   [0:8]   u64 little-endian header byte-length
        #   [8:8+N] JSON header mapping tensor name -> {dtype, shape, data_offsets}
        #   [8+N:]  raw tensor data, offsets relative to end of header
        header_len = struct.unpack("<Q", os.pread(self._fd, 8, 0))[0]
        header_bytes = os.pread(self._fd, header_len, 8)
        header = json.loads(header_bytes.decode("utf-8"))

        self._user_metadata = header.pop("__metadata__", None)
        self._meta = header
        self._data_offset = 8 + header_len

    # ---- Context-manager protocol ----
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._close()

    def __del__(self):
        try:
            self._close()
        except Exception:
            pass

    def _close(self):
        fd = getattr(self, "_fd", None)
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
            self._fd = None

    # ---- safetensors API (what transformers actually uses) ----
    def keys(self):
        return list(self._meta.keys())

    def metadata(self):
        return self._user_metadata if isinstance(self._user_metadata, dict) else {}

    def get_tensor(self, name: str) -> torch.Tensor:
        slice_ = _NoMmapSlice(self, name)
        tensor = slice_._materialize()
        if self._device != "cpu":
            tensor = tensor.to(self._device)
        return tensor

    def get_slice(self, name: str) -> _NoMmapSlice:
        return _NoMmapSlice(self, name)


# ---------------------------------------------------------------------------
# Patching helpers
# ---------------------------------------------------------------------------

_PATCHED = False


def patch_transformers_safetensors_loader() -> None:
    """Replace transformers' `safe_open` with the no-mmap variant.

    Idempotent — calling multiple times has no additional effect.
    """
    global _PATCHED
    if _PATCHED:
        return
    import transformers.modeling_utils as _mu
    _mu.safe_open = NoMmapSafeOpen
    _PATCHED = True
