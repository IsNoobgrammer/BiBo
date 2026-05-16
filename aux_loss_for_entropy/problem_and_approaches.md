# The Confidence Problem: CCE-Only Training Produces Diffuse Predictions

## The Problem

Cross-entropy loss (CCE) minimizes `−log(p_correct)`. This pushes the correct token's probability up, but has **no explicit penalty for spreading mass across wrong tokens**. The loss is satisfied as long as the correct token gets enough probability — it doesn't care if the remaining 80% is spread across 5 tokens or 2000.

### Symptoms
- Loss decreases steadily (model learns the distribution)
- Accuracy oscillates (argmax is unstable when top-K logits are close)
- Model "knows" the answer is in {A, B, C} but can't commit to one
- Particularly bad for tasks with structured outputs (sorting, math, code)

### Why This Happens in MoE Models (BiBo specifically)
1. **Multiple experts contribute** — each expert pushes logits in slightly different directions
2. **Shared expert adds baseline** — Conv1D shared expert provides a "safe" distribution that's inherently diffuse
3. **Top-K routing** — 3 experts each contribute ~33% weight, creating a mixture of 3 distributions
4. **Small hidden size (256)** — lm_head (256→2048) is underdetermined; many weight configurations produce similar loss but different sharpness

### The Sorting Task Specifically
For sorting `[5, 2, 8, 1]` → `[1, 2, 5, 8]`:
- Position 0 output: model knows it's a small number, gives probability to {1, 2, 3} instead of committing to {1}
- CCE is happy if p(1) = 0.4 (loss = 0.92) — but accuracy = 0 if p(2) = 0.35

---

## Approach 1: Entropy Penalty (Auxiliary Loss)

**Idea:** Add a term that penalizes high entropy in the output distribution.

```python
# After computing logits on the valid (non-masked) positions:
probs = F.softmax(logits[mask], dim=-1)  # [n_valid, vocab]
entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
loss = cce_loss + lambda_ent * entropy
```

**Pros:**
- Simple, differentiable, well-understood
- Directly targets the symptom (high entropy = diffuse predictions)

**Cons:**
- Can cause overconfidence on WRONG answers (model becomes confident but wrong)
- Fights with CCE early in training (CCE wants to explore, entropy penalty wants to commit)
- Needs careful scheduling (λ should be 0 early, increase later)
- Uniform penalty — doesn't distinguish "confident and wrong" from "uncertain and right"

**Variant — Conditional Entropy Penalty:**
Only penalize entropy when the model's top-1 prediction is correct:
```python
top1_correct = (logits[mask].argmax(-1) == labels[mask])
entropy_on_correct = entropy[top1_correct].mean()
loss = cce_loss + lambda_ent * entropy_on_correct
```
This says: "when you're right, be MORE right." Doesn't punish uncertainty when wrong.

---

## Approach 2: Margin Loss (Contrastive)

**Idea:** Enforce a minimum gap between the correct token's logit and the runner-up.

```python
logits_valid = logits[mask]  # [n, vocab]
correct_logits = logits_valid.gather(1, labels[mask].unsqueeze(1))  # [n, 1]
# Mask out correct token, find max of remaining
logits_masked = logits_valid.clone()
logits_masked.scatter_(1, labels[mask].unsqueeze(1), float('-inf'))
runner_up = logits_masked.max(dim=1).values  # [n]

margin = correct_logits.squeeze() - runner_up  # positive = correct is higher
margin_loss = F.relu(target_margin - margin).mean()  # penalize when margin < target
loss = cce_loss + lambda_margin * margin_loss
```

**Pros:**
- Directly targets the decision boundary (argmax stability)
- Doesn't care about the tail of the distribution
- More targeted than blanket entropy penalty

**Cons:**
- `target_margin` is a hyperparameter (what's the right gap?)
- Can be unstable early in training when correct token isn't even top-5
- Doesn't help when model is fundamentally uncertain (multiple valid answers)

---

## Approach 3: Label Smoothing in Reverse (Sharpening)

**Idea:** Instead of smoothing labels (spreading mass), use a "peaked" target that's sharper than one-hot.

Standard label smoothing: `target = (1-ε) * one_hot + ε/V`
Reverse: use temperature-scaled soft targets from the model itself (self-distillation with low temperature).

```python
# Compute sharp targets from model's own logits
with torch.no_grad():
    sharp_targets = F.softmax(logits[mask] / temperature, dim=-1)  # T < 1 = sharper
    # Mix with one-hot
    one_hot = F.one_hot(labels[mask], vocab_size).float()
    targets = alpha * one_hot + (1 - alpha) * sharp_targets

loss = -(targets * F.log_softmax(logits[mask], dim=-1)).sum(dim=-1).mean()
```

**Pros:**
- Self-reinforcing: model's own confident predictions become training signal
- No new hyperparameters beyond temperature and alpha

**Cons:**
- Circular: if model is wrong, it reinforces wrong answers
- Only works after model has some accuracy (not from scratch)
- Needs careful scheduling

---

## Approach 4: Heuristic Bias Update (Router-Style, No Gradient)

**Idea:** Inspired by BiBo's router bias update — maintain a non-trainable temperature/bias on the lm_head that's updated heuristically based on prediction confidence.

```python
class ConfidenceRegulator(nn.Module):
    """
    Non-trainable temperature that sharpens lm_head output.
    Updated heuristically: if model is often right but not confident, sharpen.
    If model is often wrong and confident, soften.
    """
    def __init__(self, initial_temp=1.0, update_factor=0.01, threshold=1000):
        super().__init__()
        self.register_buffer('temperature', torch.tensor(initial_temp))
        self.register_buffer('correct_confidence_sum', torch.tensor(0.0))
        self.register_buffer('wrong_confidence_sum', torch.tensor(0.0))
        self.register_buffer('tokens_seen', torch.tensor(0))
        self.update_factor = update_factor
        self.threshold = threshold
    
    def forward(self, logits):
        return logits / self.temperature.clamp(min=0.1)
    
    @torch.no_grad()
    def update(self, logits, labels, mask):
        """Call during training after forward pass."""
        probs = F.softmax(logits[mask], dim=-1)
        top1_probs = probs.max(dim=-1).values
        correct = (logits[mask].argmax(-1) == labels[mask])
        
        self.correct_confidence_sum += top1_probs[correct].sum()
        self.wrong_confidence_sum += top1_probs[~correct].sum()
        self.tokens_seen += mask.sum()
        
        if self.tokens_seen >= self.threshold:
            avg_correct_conf = self.correct_confidence_sum / (correct.sum() + 1)
            avg_wrong_conf = self.wrong_confidence_sum / (~correct).sum().clamp(min=1)
            
            # If correct predictions are low-confidence → sharpen (decrease temp)
            # If wrong predictions are high-confidence → soften (increase temp)
            if avg_correct_conf < 0.5:  # right but not confident
                self.temperature.sub_(self.update_factor)
            elif avg_wrong_conf > 0.7:  # wrong and overconfident
                self.temperature.add_(self.update_factor)
            
            self.temperature.clamp_(min=0.3, max=2.0)
            
            # Reset accumulators
            self.correct_confidence_sum.zero_()
            self.wrong_confidence_sum.zero_()
            self.tokens_seen.zero_()
```

**Pros:**
- No gradient interference with CCE
- Self-regulating: won't over-sharpen if model is wrong
- Mirrors the proven router bias approach
- Simple to implement, no new loss terms
- Can be added/removed without retraining

**Cons:**
- Heuristic thresholds (0.5, 0.7) need tuning
- Slower to adapt than gradient-based approaches
- Global temperature — doesn't distinguish easy vs hard positions
- Might oscillate if update_factor too large

---

## Approach 5: Top-K Focused Loss (Only Penalize Confusion Among Top Candidates)

**Idea:** Standard CCE operates over full vocab (2048). But the confusion is only among top-K candidates. Focus the sharpening signal there.

```python
logits_valid = logits[mask]  # [n, vocab]
# Get top-K logits (where confusion lives)
topk_vals, topk_idx = logits_valid.topk(k=10, dim=-1)  # [n, 10]

# Is correct answer in top-K?
correct_in_topk = (topk_idx == labels[mask].unsqueeze(1)).any(dim=1)  # [n]

# For tokens where correct IS in top-K: penalize entropy of top-K distribution
topk_probs = F.softmax(topk_vals[correct_in_topk], dim=-1)
topk_entropy = -(topk_probs * torch.log(topk_probs + 1e-10)).sum(dim=-1).mean()

loss = cce_loss + lambda_topk * topk_entropy
```

**Pros:**
- Only sharpens where it matters (model already "knows" the answer is in top-K)
- Doesn't waste gradient on the 2000+ tokens that are already near-zero
- More sample-efficient than full entropy penalty
- Natural curriculum: as model improves, more tokens have correct in top-K

**Cons:**
- K is a hyperparameter
- Slightly more compute (topk operation)
- Still a gradient-based approach (can interfere with CCE)

---

## Recommended Implementation Order

1. **Start with Approach 4 (Heuristic)** — zero risk, no gradient interference, can be toggled off
2. **Add Approach 5 (Top-K entropy)** if heuristic alone isn't enough — targeted, efficient
3. **Approach 2 (Margin)** as nuclear option if model is consistently "almost right"

## Key Insight

The router bias heuristic works because it's **decoupled from the main loss**. Same principle applies here: the confidence regulator should observe and adjust, not fight the gradient. CCE handles "what to predict." The regulator handles "how confidently to predict it."

---

## Relevant Literature

- **Confidence Penalty** (Pereyra et al., 2017) — penalize low entropy (opposite of label smoothing)
- **Focal Loss** (Lin et al., 2017) — down-weight easy examples, focus on hard ones
- **Knowledge Distillation** (Hinton et al., 2015) — temperature scaling for soft targets
- **Margin-based losses** (Liu et al., 2016, ArcFace/CosFace) — enforce angular margin
- **Skywork-MoE** (Wei et al., 2024) — heuristic bias for load balancing (our inspiration)
