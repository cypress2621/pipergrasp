
## 项目简介

PiperGrasp 基于 GraspNet-1Billion (CVPR 2020) 基线模型，在深度学习抓取检测的基础上，引入大语言模型（LLM）构建智能交互前端，实现从视觉感知到自然语言推理再到抓取决策的端到端闭环。硬件使用piper六自由度机械臂

传统抓取检测系统输出的是无语义的几何抓取位姿，PiperGrasp 通过 LLM 赋予系统可解释的推理能力——用户可以用自然语言描述需求（如"抓取那个红色杯子，但不要碰旁边的碗"），系统结合视觉感知与语义推理输出最优抓取策略。

## 技术架构

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   用户输入层     │────▶│   LLM推理层      │────▶│  抓取决策层      │
│  自然语言指令    │     │ 语义解析/目标识别 │     │  位姿优化/避障   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                                               │
         └───────────────────────────────────────────────┘
                              ↑
                    ┌─────────────────┐
                    │  GraspNet视觉感知 │
                    │  RGB-D点云/深度图 │
                    └─────────────────┘
```

## 核心特性

- **CVPR 2020 基线**：基于 GraspNet-1Billion 大规模抓取数据集，支持 Seen / Similar / Novel 三类物体的泛化抓取
- **LLM 语义融合**：通过大语言模型解析自然语言指令，实现语义级目标选择与抓取策略推理
- **多模态感知**：融合 RGB-D 图像与点云数据，支持 RealSense / Kinect 双相机配置
- **端到端推理**：从视觉输入到抓取位姿输出的完整 pipeline，包含碰撞检测后处理

## 技术栈

| 模块 | 技术 |
|------|------|
| 视觉感知 | PyTorch, Open3D, PointNet++, KNN CUDA |
| 抓取检测 | GraspNet Baseline (CVPR 2020) |
| 语义推理 | 大语言模型 (LLM) |
| 前端交互 | Python Web Frontend |
| 部署环境 | CUDA, Python 3 |

## 快速开始

### 环境要求

- Python 3
- PyTorch 1.6+
- CUDA (用于 GPU 加速)
- Open3D >= 0.8

### 安装

```bash
git clone https://github.com/cypress2621/pipergrasp.git
cd pipergrasp/graspnet
pip install -r requirements.txt

# 编译 PointNet++ 算子
cd pointnet2
python setup.py install

# 编译 KNN CUDA 算子
cd ../knn
python setup.py install
```

### 预训练权重

| 模型 | 相机 | 下载 |
|------|------|------|
| checkpoint-rs.tar | RealSense | [Google Drive](https://drive.google.com/file/d/1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk/view?usp=sharing) |
| checkpoint-kn.tar | Kinect | [Google Drive](https://drive.google.com/file/d/1vK-d0yxwyJwXHYWOtH1bDMoe--uZ2oLX/view?usp=sharing) |

### 运行 Demo

```bash
# 视觉抓取检测
cd graspnet
python demo.py --checkpoint_path checkpoint-rs.tar

# LLM 交互前端
python graspnet_llm.py
```

## 实验结果

在 GraspNet-1Billion 数据集上的评估（RealSense 相机）：

| 场景 | AP | AP@0.8 | AP@0.4 |
|------|-----|--------|--------|
| Seen | 47.47 | 55.90 | 41.33 |
| Similar | 42.27 | 51.01 | 35.40 |
| Novel | 16.61 | 20.84 | 8.30 |

## 项目结构

```
pipergrasp/
├── graspnet/
│   ├── demo.py              # 抓取检测 Demo
│   ├── graspnet_llm.py      # LLM 交互前端 ⭐
│   ├── models/              # 抓取检测网络
│   ├── pointnet2/           # PointNet++ 算子
│   ├── knn/                 # KNN CUDA 算子
│   ├── dataset/             # 数据加载与预处理
│   └── doc/                 # 文档与示例数据
└── README.md
```

## 未来方向

- [ ] 多模态大模型融合（视觉-语言联合推理）
- [ ] 强化学习优化抓取策略
- [ ] 实时抓取规划与机械臂控制集成
- [ ] 仿真到真实（Sim-to-Real）迁移

## 引用

本项目基于 GraspNet-1Billion 开源代码：

```bibtex
@inproceedings{fang2020graspnet,
  title={GraspNet-1Billion: A Large-Scale Benchmark for General Object Grasping},
  author={Fang, Hao-Shu and Wang, Chenxi and Gou, Minghao and Lu, Cewu},
  booktitle={CVPR},
  pages={11444--11453},
  year={2020}
}
```

## 许可证

非商业用途自由使用，遵循 GraspNet 原始许可证。

