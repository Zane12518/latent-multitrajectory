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

## GRU 模块的历史许可

`src/latent_multi/gru/` 的模型、目标与原指标提取自项目留存历史源树，该源树附带 MIT 与版权 2026 Aleksey Komissarov。完整声明保留在 [HISTORICAL-GRAM-LICENSE.txt](HISTORICAL-GRAM-LICENSE.txt)，不替代官方 GRAM 核心的许可。新接口按项目 MIT；数据和权重不由这份代码许可授权。
