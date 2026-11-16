# Third-party provenance and notices

Vendored core: [ahn-ml/gram-release](https://github.com/ahn-ml/gram-release), local upstream snapshot archived 2026-10-17.
License: MIT, copyright 2025 Samsung Electronics Co., Ltd. Full text: [GRAM-LICENSE.txt](GRAM-LICENSE.txt).
The original upstream commit could not yet be established from the archived directory; snapshot identity and exact file adaptation diff are retained in the private release audit. Do not claim a verified upstream commit or full official reproduction.

Included files: common.py, layers.py, sparse_embedding.py, recursive_reasoning/gram.py, losses.py.
Changes: whitespace-only normalization; namespaced imports; existing SDPA layout reshape; Maze no-puzzle halt readout; Maze task correctness in ACT loss. The inherited local snapshot already uses PyTorch SDPA instead of FlashAttention.
Third-party copyright/license remains applicable; project-owned adapters are MIT; data and checkpoint distribution is approved separately (see ../ASSET_TERMS.md).

Dependencies: PyTorch (BSD-style), NumPy (BSD-3-Clause), einops (MIT), Pydantic (MIT), pytest (MIT), setuptools/wheel (MIT). They are installed separately, not redistributed in the candidate archive. See installed package metadata for exact versions and license texts.

Repository organization reference only: [ADIOCLASSMATE/cq-pretrain](https://github.com/ADIOCLASSMATE/cq-pretrain).
No code, architecture, checkpoint or production training infrastructure from that repository is included.
