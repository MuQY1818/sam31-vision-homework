# SAM 3.1 图像处理课程实验

本仓库为“图像处理与计算机视觉”课程的实验材料，基于 Meta 开源项目 [SAM 3: Segment Anything with Concepts](https://github.com/facebookresearch/sam3) 的 SAM 3.1 Object Multiplex checkpoint，完成开放词汇视频目标分割与单帧图像文本提示分割实验。

## 仓库结构

- `report/SAM3_1_experiment_report.pdf`：课程实验报告终稿。
- `report/source/`：LaTeX 源文件、参考文献及配图。
- `scripts/run_sam31_player.py`：足球短视频三组 prompt 分割实验脚本。
- `scripts/run_sam31_image_cases.py`：单帧图像补充分割实验脚本。
- `scripts/build_sam31_report_figures.py`：根据已有结果截图生成报告中的组合展示图。
- `results/video/`：`person`、`ball`、`player in red` 三组视频分割结果截图。
- `results/image/`：校园夜景、车厢物品、街景货车等单帧图像分割结果。
- `metadata/`：实验脚本自动记录的 JSON 元数据（耗时、GPU、帧数、目标数等）。

本仓库不含 Meta 官方 SAM3 源码与模型权重。源码请从 `facebookresearch/sam3` 获取，SAM 3.1 权重可从 ModelScope `facebook/sam3.1` 下载。

## 实验环境

所有实验在独立 Conda 环境 `sam31` 中完成，未向 base 环境安装额外 Python 包。

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 |
| Python | 3.12.13 |
| PyTorch | 2.6.0+cu124 |
| TorchVision | 0.21.0+cu124 |
| CUDA Runtime | 12.4 |
| Checkpoint | `sam3.1_multiplex.pt` |

## 实验复现

首先克隆官方 SAM3 仓库，并下载 SAM 3.1 权重：

```bash
git clone https://github.com/facebookresearch/sam3.git
mkdir -p checkpoints/sam3.1
conda run -n sam31 modelscope download \
    --model facebook/sam3.1 \
    sam3.1_multiplex.pt config.json \
    --local_dir checkpoints/sam3.1
```

视频分割实验（三组 prompt）：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run -n sam31 python scripts/run_sam31_player.py \
    --max-frames 8 --frame-step 6 \
    --prompt person --output-prefix sam31_player_person

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run -n sam31 python scripts/run_sam31_player.py \
    --max-frames 8 --frame-step 6 \
    --prompt ball --output-prefix sam31_player_ball

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run -n sam31 python scripts/run_sam31_player.py \
    --max-frames 8 --frame-step 6 \
    --prompt "player in red" --output-prefix sam31_player_red
```

单帧图像实验：

```bash
conda run -n sam31 python scripts/run_sam31_image_cases.py \
    --confidence-threshold 0.35 --max-instances 8
```

## 实验结果

视频实验的三组文本提示结果：

| Prompt | 每帧目标数 | 加载时间 | 推理时间 | 观察结果 |
|---|---:|---:|---:|---|
| `person` | 3 | 9.35 s | 1.46 s | 检测并跟踪三名球员。 |
| `ball` | 1 | 9.47 s | 1.43 s | 检测并跟踪足球小目标。 |
| `player in red` | 1 | 9.42 s | 1.49 s | 根据颜色属性锁定红衣球员。 |

单帧图像实验展示了 `building`、`truck`、`wheel` 等不同粒度 prompt 的输出结果，同时记录了 `tree` 和 `bottle` 在默认阈值下未保留实例的对照情况。

## 编译报告

报告 LaTeX 源文件位于 `report/source/main.tex`，使用 XeLaTeX 编译：

```bash
cd report/source
latexmk -xelatex -interaction=nonstopmode main.tex
```

## 声明

- 本仓库仅用于课程实验材料展示与复现说明。
- 官方 SAM3 源码版权归 Meta Platforms, Inc. 及其贡献者所有。
- 模型权重未上传至本仓库，请遵循 ModelScope 及 Meta 官方许可要求获取和使用。
