# rl-finetune-comparisons

This repository compares Gradient Reward Policy Optimization (GRPO), Direct Preference Optimization (DPO) and Scalarized Reward Policy Optimization (SRPO) for multimodal fine ‑tuning on top of the flux1.0 kontext model. We will evaluate how these algorithms improve the base model after training. For reward modeling we use the existing AutoQC model that scores image quality.

## Mathematical Outline

### GRPO
GRPO optimizes a policy by ascending the gradient of expected reward. For a policy parameterized by $\theta$ the objective is $J(\theta)=\mathbb{E}_{\pi_\theta}[R]$ and the gradient is estimated via REINFORCE:
\[\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\big[\nabla_\theta \log \pi_\theta(a|s)\,(R - b)\big],\n\]
where $R$ is the reward (here, a combination of task‑specific reward and image quality score from AutoQC) and $b$ is a baseline to reduce variance.

### DPO
Direct Preference Optimization learns directly from preference pairs by maximizing the likelihood of preferred outputs. Given a triplet $(x, y^{*}, y^{-})$ with $y^{*}$ preferred over $y^{-}$, DPO minimizes the pairwise logistic loss:
\[\
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x,y^{*},y^{-})}\left[\log \sigma\big(f_\theta(x,y^{*}) - f_\theta(x,y^{-})\big)\right],
\]
where $f_\theta$ is the model’s scoring function and $\sigma$ is the sigmoid. Gradient updates follow standard supervised learning on this loss.

### SRPO
Scalarized Reward Policy Optimization combines multiple reward signals into a single scalar reward. If rewards $r_1,\dots,r_k$ are combined with weights $w_i$, the overall reward is $R = \sum_{i} w_i\,r_i$. The policy gradient is then
\[\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\big[\nabla_\theta \log \pi_\theta(a|s)\,(R - b)\big],\n\]
similar to GRPO but with a scalarized reward that may include accuracy, F1 and AutoQC scores.

## Pseudocode for Gradient Updates

```pseudo
initialize policy parameters θ
for each training iteration:
    sample a batch of examples x from training data
    for each example x:
        generate candidate outputs using current policy πθ
        compute rewards:
            - task reward (e.g., accuracy/F1 on multimodal task)
            - quality reward from AutoQC
            - preferences (for DPO only)
        depending on algorithm:
            if GRPO:
                compute log‑probabilities log πθ(a|s)
                estimate gradient ∇θ log πθ(a|s) * (R - baseline)
            if DPO:
                form pairs (preferred, non‑preferred) and compute logistic loss gradient
            if SRPO:
                form scalar reward R = Σ w_i r_i and estimate policy gradient
    update θ using optimizer (e.g., Adam) with aggregated gradients
```

## Evaluation Plan

We will assess how fine ‑tuning with GRPO, DPO and SRPO affects the flux1.0 kontext model. Evaluation metrics include:

- **Accuracy** – fraction of correct predictions on a held‑out multimodal validation set.
- **F1‑score** – harmonic mean of precision and recall for tasks such as classification.
- **Reward signals** – scores from the AutoQC model (image quality) and any task‑specific reward; for SRPO, we will report contributions of each reward component.
- Training curves and convergence behaviour will also be compared.

## Notebook

A Colab‑ready skeleton notebook that sets up datasets, defines placeholders for GRPO/DPO/SRPO algorithms and evaluation hooks can be found in the `notebooks` directory. Use the link below to open it in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OneBoxCream/rl-finetune-comparisons/blob/main/notebooks/grpo_dpo_srpo_comparison.ipynb)
