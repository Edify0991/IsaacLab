# FMMP Research Prototype (IsaacLab Humanoid AMP)

本目录提供一个**可复现实验骨架**，用于验证：在训练数据缺少技能间过渡片段时，随机技能切换下仍能实现稳定平滑过渡。

## 目录

- `parts.py`: 人体分部件定义、特征抽取、接触特征。
- `manifold_encoders.py`: 分部件流形编码器（相位 + 幅值/偏置）。
- `priors.py`: 分部件判别器 + 全身耦合判别器与 AMP 风格 reward。
- `transition_generator.py`: 过渡补全生成器（inpainting/in-betweening）。
- `metrics.py`: 切换评测指标。
- `configs/*.yaml`: 训练与评测配置样例。

## 1) 构建无过渡数据集

```bash
python scripts/build_no_transition_dataset.py \
  --motions source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/motions/humanoid_run.npz \
           source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/motions/humanoid_dance.npz \
           source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/motions/humanoid_walk.npz \
  --skills RunLegs ArmDribble StopAndShootUpperBody \
  --trim_start 0.2 --trim_end 0.8 \
  --output scripts/fmmp/no_transition_motion_list.json
```

## 2) 训练分部件流形编码器

```bash
python scripts/train_part_manifold_encoder.py \
  --dataset <prepared_part_window_tensor.pt> \
  --output checkpoints/fmmp/part_encoders.pt \
  --epochs 20
```

## 3) 训练策略（保留 AMP 基线 + FMMP 模式）

AMP baseline:
```bash
python scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-Humanoid-AMP-Run-Direct-v0 --prior amp
```

FMMP:
```bash
python scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-Humanoid-AMP-Run-Direct-v0 --prior fmmp
```

## 4) 生成随机技能切换指令（30~60秒，1~3秒切换）

```bash
python scripts/fmmp/generate_skill_commands.py \
  --skills RunLegs ArmDribble StopAndShootUpperBody \
  --duration 45 --min_interval 1.0 --max_interval 3.0 \
  --output outputs/fmmp/random_switch_45s.json
```

## 5) 一键评测

```bash
python eval_switching.py \
  --checkpoint <policy_checkpoint.pt> \
  --commands outputs/fmmp/random_switch_45s.json \
  --output outputs/eval/fmmp
```

输出：`metrics.csv` 与 `metrics.png`（可扩展视频录制钩子）。

## 对比实验建议

用同一 `eval_switching.py` 跑：

1. Baseline AMP（`--prior amp`）
2. 简化 PMP（分部件判别，无 Ek/G）
3. FMMP 完整版（Ek + Dk + Dg + G）
4. Ablation: `-Dg`, `-Ek`, `-G`

> 当前提交是研究原型骨架：关键函数/模块已可运行，部分 IsaacLab 深度集成点标注为 TODO，便于后续插入真实 rollout 与训练循环。
