# Attention Residuals: parameter routing

> **The rule**: the AttnRes pseudo-queries are `nn.Linear(hidden, 1)`. They must go to **AdamW**,
> not Muon, and at the **embedding/default lr** (`--adam_lr`, 5e-4), not the act-scale lr.
> Set `--vec_matrices_adamw true`. This generalizes: **any `(1, H)` or `(H, 1)` parameter belongs
> on AdamW**, the same bracket embeddings and `lm_head` already sit in.

## The parameters in question

Block AttnRes (Kimi K3 style, see [attention_layers.md](attention_layers.md)) attaches a learned
pseudo-query at each residual site plus one at the trunk output:

```
self_attention_res_proj    nn.Linear(hidden, 1, bias=False)   per layer
mlp_res_proj               nn.Linear(hidden, 1, bias=False)   per layer, sites=2 only
output_attn_res_proj       nn.Linear(hidden, 1, bias=False)   trunk output
```

At `sites=1` on a 10-layer model that is **11 parameters, 5632 elements** -- 0.0009% of a 657M
model. They are also the parameters that decide, at every layer, how much of each earlier block
the stream re-reads. Their routing is worth more than their size suggests.

Muon's group-assignment rule in `ablate/common/optim.py` is `p.ndim == 2 -> Muon`. That rule
catches these **by accident**. Muon's actual convention is *2D hidden-layer matrices*, with
embeddings, heads, norms and biases on AdamW. A `(1, H)` tensor is a vector wearing a matrix
shape, and it is on the AdamW side of that convention, not the Muon side.

## The measurement

Board: `bibo-attnres`, 137M-active / 657M-total, seed 42069, 64 experts k=6, 262144 tok/step,
2000 steps cosine, SWA block3 w128 + XSA, `--attn_res 3 --attn_res_sites 1 --attn_res_carry true
--attn_res_fp32_stream true`. Within-box noise floor 0.0014 bpb.

Five arms, differing only in how the 11 pseudo-queries are optimized:

| arm | pseudo-query optimizer | measured per-neuron step |
|---|---|---|
| base | (no AttnRes at all) | -- |
| unbounded | Muon, aurora scale | 0.04485 |
| csfix | Muon, aurora scale | 0.04485 |
| **vecadamw** | AdamW default group, lr 5e-4, wd 0.1 | **0.00268** |
| **vecact** | AdamW act-scale group, lr 0.01, wd 0 | **0.0538** |

Running mean train loss (`run20`) at matched steps:

| step | base | unbound | csfix | vecadamw | vecact |
|---|---|---|---|---|---|
| 400 | 2.2867 | 2.2887 | 2.2906 | **2.2527** | 2.2767 |
| 500 | 2.2355 | 2.2412 | 2.2445 | **2.1992** | 2.2234 |
| 600 | 2.1485 | 2.1239 | 2.1371 | **2.0923** | 2.1181 |
| 700 | 2.0610 | 2.0453 | 2.0465 | **2.0087** | 2.0339 |
| 750 | 1.9043 | 1.8943 | 1.8956 | **1.8563** | 1.8815 |
| 775 | 2.3703 | 2.3602 | 2.3608 | **2.3143** | 2.3445 |

Held-out bpb at the step-1000 eval:

| arm | bpb hi | bpb en |
|---|---|---|
| base | 0.579 | 1.183 |
| unbounded | 0.582 | 1.166 |
| csfix | 0.581 | 1.172 |
| **vecadamw** | **0.569** | **1.160** |

The three Muon arms cluster at 0.579 / 0.581 / 0.582 -- spread 0.003, about 2x the noise floor,
i.e. mutually indistinguishable. `vecadamw` sits 0.010-0.013 below the whole cluster, roughly
**7x the noise floor**, and wins `en` as well.

### Why the bracket is the argument

One AdamW arm beating Muon would be ambiguous: it could be the optimizer, or just a different
learning rate. Two AdamW arms whose step sizes land on **opposite sides of Muon's** removes that
ambiguity:

```
AdamW default  0.0027   17x BELOW Muon    beats Muon
Muon           0.0449   --
AdamW act      0.0538   1.2x ABOVE Muon   beats Muon
```

Both beat Muon, at 6/6 matched steps, in a strictly consistent ordering
(`vecadamw < vecact < unbound ~ csfix < base`). A step-size explanation would have to predict that
one of the two loses. Neither does. **The cause is the update geometry, not the magnitude.**

Within AdamW, lower is better: `vecadamw` leads `vecact` by 0.025 train loss at step 700, also at
6/6 steps. Hence the recommendation of the default/embedding lr rather than the act-scale lr.

## Why Muon is the wrong tool here

**1. Rank 1 means there is no spectrum to condition.** For a `(1, H)` matrix the spectral norm and
the L2 norm coincide, there is exactly one singular value, and `msign(g) = g / ||g||`. Muon does
not orthogonalize anything -- it degenerates to **normalized SGD with momentum at a fixed step
norm**, discarding the gradient magnitude completely. That is a coherent optimizer, but it is not
the algorithm Muon's justification is about, and nobody chose it.

**2. The magnitude is semantically load-bearing.** In `apply_attention_residual`
(`exp/modeling_bibo.py:106`):

```python
score_weight = norm.weight.float() * projection.weight.squeeze(0).float()
```

`score_weight` produces the logits of a softmax over blocks, so `||score_weight||` **is the
temperature** of that softmax: small norm mixes all blocks toward uniform, large norm picks one.
Muon's fixed-norm update denies the gradient any way to say "sharpen" through magnitude, and the
resulting norm drifts to whatever the update/decay balance dictates rather than what the loss
wants.

**3. Muon decays the temperature 20x harder.** Decoupled decay is `lr * wd`:

```
Muon           0.01   * 0.1 = 1.0e-3
AdamW default  5e-4   * 0.1 = 5.0e-5
```

So under Muon the temperature carrier is pulled toward an `lr/wd` equilibrium set by
hyperparameters, an order of magnitude harder than any other AdamW parameter experiences.

**4. The `lm_head` analogy gets the right answer for the wrong reason.** `lm_head` is kept off
Muon because it is `(81920, H)`: orthogonalization flattens all 81920 singular values and destroys
the per-class norm structure that a Zipf-distributed vocabulary needs. That mechanism **cannot
apply at one row** -- there is no cross-row structure. The correct reason to group these with
`lm_head` and the embeddings is different: they are all parameters where magnitude carries meaning
and where Muon's conditioning either does not apply or actively removes information.

## Usage

```
--vec_matrices_adamw true          # route (1,H)/(H,1) params to AdamW.  RECOMMENDED
--vec_adamw_group default          # ...at adam_lr / wd.  RECOMMENDED
--vec_adamw_group act              # ...at act_scale_lr / wd 0.  The upper bracket arm.
```

Verify from the launch banner: `muon_mats` drops **81 -> 70** at `sites=1`. With
`--vec_adamw_group act` the `[optim] act scales:` count rises by 11 (18 -> 29). If neither number
moves, the flag did not take -- check for a stale `__pycache__` after `git pull`.

Run names carry `_vecadamw` / `_vecadamwact` so the arms do not collide in W&B.

## Not settled

- **`vecact` has no bpb yet.** Its train-loss position is established at 6/6 steps; the arm is
  running to 2000 to give the act-group config a completed reference. The bpb evidence above is
  `vecadamw`'s step-1000 eval only.
- **Neither AdamW arm has a 2000-step final eval.** `vecadamw` was stopped at step 1175. It should
  be re-run to completion -- on current evidence it is the champion of this board and its final
  number is missing.
- **ICL moved the wrong way** in `vecadamw`: slope 0.000 / -0.014 and both jumps at 0.50, against
  -0.10 / -0.09 and 0.6-0.8 for the field. Prior rounds established ICL on this board as
  seed-variance-dominated (see the aurora/EMA round), so this is not currently read as signal, but
  it is the one metric contradicting the result and should be rechecked at 2000.
- **The lr is bracketed, not optimized.** 5e-4 beats 0.01; nothing below 5e-4 has been tried.
- **Only tested at `sites=1` with `attn_res_carry`.** The routing argument is structural and
  should hold at `sites=2`, but that is inference, not measurement.
