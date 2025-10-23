# rl-finetune-comparisons  

本仓库用于比较梯度奖励策略优化（GRPO）、直接偏好优化（DPO）和标量化奖励策略优化（SRPO）在多模态微调场景下的效果。基准模型是 flux1.0 kontext，我们将在该模型基础上继续训练，评估三种算法带来的提升。奖励模型使用我们已有的 AutoQC 模型，它可以对图片质量进行评分，作为额外奖励信号。  

## 数学概述  

### GRPO  
GRPO 通过沿期望奖励的梯度上升来优化策略。对于参数化策略 $\theta$，目标函数为 $J(\theta)=\mathbb{E}_{\pi_\theta}[R]$。梯度可以通过 REINFORCE 估计：  
$$  
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\big[\nabla_\theta \log \pi_\theta(a|s)\,(R - b)\big],  
$$  
其中 $R$ 是奖励（这里是任务奖励与 AutoQC 图像质量分数的组合），$b$ 是降低方差的基线。  

### DPO  
直接偏好优化直接从偏好对中学习，通过最大化偏好结果的似然来训练。在给定三元组 $(x, y^+, y^-)$，其中 $y^+$ 优于 $y^-$，DPO 最小化偏好损失：  
$$  
\mathcal{L}(x) = -\mathbb{E}_{\{(x,y^+,y^-)\}}\left[\log\sigma\big(f_\theta(x,y^+) - f_\theta(x,y^-)\big)\right],  
$$  
其中 $f_\theta$ 是模型的评分函数，$\sigma$ 是 sigmoid 函数。梯度更新遵循该损失的标准监督学习。  

### SRPO  
标量化奖励策略优化将多个奖励信号组合成单一的标量奖励。如果奖励 $r_1,\dots,r_k$ 以权重 $w_i$ 组合，整体奖励为 $R = \sum_i w_i\,r_i$。然后  
$$  
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\big[\nabla_\theta \log \pi_\theta(a|s)\,(R - b)\big],  
$$  
类似 GRPO，但使用标量化奖励。SRPO 适用于同时考虑准确率、F1 值和 AutoQC 评分的场景。  

## 梯度更新伪代码  

```pseudo  
# 初始化策略参数 θ  
for each batch of examples:  
    # 根据当前策略采样动作或生成输出  
    outputs = policy(x; θ)  
    # 计算奖励（任务指标 + AutoQC 分数）  
    rewards = compute_rewards(outputs)  
    # 估计梯度  
    grads = ∇_θ log π_θ(outputs | x) * (rewards - baseline)  
    # 更新参数  
    θ = θ + α * grads  
```  

对于 DPO，其伪代码将偏好损失作为梯度计算；SRPO 则改变奖励计算部分。  

## 评价计划  

我们计划通过以下指标对比三种算法在多模态微调中的效果：  

- **准确率**：输出与参考答案匹配的比例。  
- **F1 值**：在分类任务中综合考虑精确率与召回率的指标。  
- **奖励信号**：来自 AutoQC 的图片质量分数以及其它任务奖励的加权组合。  

实验将分阶段进行：首先在 flux1.0 kontext 上分别使用 GRPO、DPO 和 SRPO 进行后训练；然后在验证集上计算上述指标，并绘制学习曲线。对比不同算法的收敛速度、最终性能和稳定性。  

## Colab  

您可以通过以下按钮在 Google Colab 中查看并运行本仓库中的笔记本：  

[![在 Colab 上打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OneBoxCream/rl-finetune-comparisons/blob/main/notebooks/grpo_dpo_srpo_comparison.ipynb)
