# MambaSOD: Mamba-based Small Object Detection for UAV Imagery

MambaSOD is an end-to-end detection framework that combines a Vision Mamba (ViM) backbone with a bidirectional feature pyramid and a Mamba-based query decoder. It targets the small-object-heavy regime of UAV datasets such as VisDrone.

## Abstract 
Drone-perspective small object detection requires processing high-resolution imagery where objects typically occupy fewer than 32$\times$32 pixels, demanding both fine-grained spatial preservation and global context modeling under strict computational constraints. Convolutional neural networks are limited by local receptive fields and cannot effectively model global context, while Transformer-based approaches, despite their global modeling capability, suffer from $O(N^2)$ computational complexity that becomes a severe bottleneck when processing high-resolution inputs. This paper proposes MambaSOD, 
an end-to-end detection framework with an encoder-decoder architecture based on state space models that achieves linear computational complexity throughout the entire pipeline of feature extraction, multi-scale fusion, and query interaction. On the encoder side, MambaSOD builds a linear-complexity multi-scale representation by coupling a Vision Mamba backbone with a P2 high-resolution enhancement module, which recovers fine-grained texture details through dual-path fusion of shallow image features and up-sampled backbone output, together with a BiFPN that performs weighted bidirectional fusion across five scales. On the decoder side, we replace both self-attention and cross-attention with Mamba-driven query interaction: the Mamba-based Query Self-Interaction (MQSI) module enables implicit inter-query communication through bidirectional state propagation at $O(N_q)$ cost, while the Mamba-based Query-Feature Interaction (MQFI) module reformulates query-feature cross-attention as a sequence modeling problem, reducing its complexity from $O(N_q \times N)$ to $O(N_q + N)$. Experiments on the VisDrone2019 dataset demonstrate that MambaSOD achieves 23.8\% AP, a 52.6\% relative improvement over the Vision Mamba baseline, while requiring fewer FLOPs than mainstream detectors including Cascade R-CNN, ViT, Deformable DETR, and DINO, offering a competitive accuracy--computation trade-off for high-resolution drone-perspective small object detection.


## Architecture

![MambaSOD Architecture](/image/MambaSOD.png)

The model has three main parts:

**Encoder.** A ViM-Tiny backbone produces four equal-resolution token sequences, which are projected to four spatial scales (stride 8 / 16 / 32 / 64). A light stem generates an additional stride-4 feature map (P-1) directly from the input image, giving the decoder a high-resolution view of very small targets without the cost of upsampling from the backbone. The five-level pyramid is then refined by a BiFPN neck.

**Decoder.** Six stacked Mamba decoder layers iterate over a fixed set of object queries. Each layer contains three components:

* **MQSI** (Mamba Query Self-Interaction): bidirectional Mamba scan over queries for self-refinement.
* **MQI** (Mamba Query-memory Interaction): one branch per feature scale, each aligning the feature map to the query grid via 2D pooling/interpolation and then doing a CrossMamba pass. Outputs are fused uniformly across scales.
* **FFN.**

**Heads.** A shared classification head and an MLP box head are applied after every decoder layer. Box predictions are produced as offsets in logit space relative to per-query reference points.

### Key design choices

* **Group-DETR** is used at training time (queries are replicated across *G* groups with independent Hungarian matching) to accelerate convergence. Inference uses a single group.
* **Focal loss** is used for classification, with optional per-class alpha derived from inverse-square-root frequencies to handle VisDrone's long-tailed class distribution.
* **Reference points** are initialized as a uniform 2D grid in the image plane, matching the shape of the aligned feature maps inside MQI.

## Installation

```bash
# Python >= 3.9, CUDA >= 11.8 recommended
conda create -n mambasod python=3.10 -y
conda activate mambasod

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install mamba-ssm causal-conv1d
pip install timm einops opencv-python albumentations scipy
```

The Mamba kernels (`mamba-ssm`, `causal-conv1d`) require a CUDA-capable GPU and a matching nvcc toolchain. Triton RMSNorm is optional; a pure PyTorch fallback is provided.

## Data Preparation

Download [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset) and organize it as:

```
VisDrone/
├── VisDrone2019-DET-train/
│   ├── images/
│   └── annotations/
├── VisDrone2019-DET-val/
│   ├── images/
│   └── annotations/
└── VisDrone2019-DET-test-dev/
    ├── images/
    └── annotations/
```

## Pretrained Weights

Download the ViM-Tiny ImageNet checkpoint (`vim_t_midclstok_76p1acc.pth`) from the [Vim repository](https://github.com/hustvl/Vim) and point `--vim_pretrained` to it.

## Training

```bash
python train.py \
    --data_root /path/to/VisDrone \
    --vim_pretrained /path/to/vim_t_midclstok_76p1acc.pth \
    --output_dir ./output/mambasod \
    --batch_size 16 --grad_accum_steps 2 \
    --epochs 300 --lr 1e-4 \
    --num_queries 400 --num_groups 6 \
    --amp
```

## Repository Layout

```
MambaSOD/
├── train.py                    # training entry point
├── model.py                    # MambaSOD, decoder, MQSI, MQI
├── encoder.py                  # ViM backbone + BiFPN neck
├── mamba_block.py              # Block / CrossBlock wrappers
├── cross_mamba.py       # CrossMamba SSM module
├── ablation_modules.py         # StandardCA / StandardSA / StandardFPN
├── vim_pretrained_loader.py    # ImageNet checkpoint loader
├── vim/                        # Vim upstream code (PatchEmbed, create_block, ...)
├── docs/
│   └── architecture.png        # architecture figure
└── README.md
```

## Main Results

Comparative results on VisDrone2019.


| Method | Backbone | FLOPs (G) | Params (M) | AP (%) | AP<sub>small</sub> (%) | AP<sub>small</sub><sup>50</sup> (%) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| YOLOv8-L | CSPDarknet | 165.2 | 43.7 | 21.8 | 11.5 | - |
| YOLOv12-L | C3k2-A2C2f | 88.9 | 26.5 | 22.9 | 12.8 | - |
| Cascade R-CNN | ResNet-50 | 236.8 | 69.1 | 16.8 | 9.2 | 17.1 |
| ViT | ViT-B | 478.7 | 115.0 | 19.5 | 10.6 | 21.8 |
| Vision Mamba | ViM-T | 115.3 | 45.4 | 15.6 | 7.8 | 16.5 |
| Deformable DETR | ResNet-50 | 173.5 | 40.1 | 21.3 | 11.6 | 23.7 |
| Conditional DETR | ResNet-50 | 93.1 | 44.0 | 14.5 | 7.4 | 15.6 |
| DINO | ResNet-50 | 245.6 | 48.2 | 23.6 | 13.4 | 27.4 |
| **MambaSOD (ours)** | ViM-T | 187.6 | 74.9 | **23.8** | **14.1** | **28.7** |

```

## Acknowledgements

This work builds on [Vim](https://github.com/hustvl/Vim) (Zhu et al.) and [Mamba](https://github.com/state-spaces/mamba) (Gu & Dao), and borrows matching/loss design from [DETR](https://github.com/facebookresearch/detr), [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), and [Group-DETR](https://github.com/Atten4Vis/GroupDETR).
