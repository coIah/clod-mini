# clod-mini
clod_mini.py - chat format + SFT + DPO in 284 lines of pure Python. The algorithmic delta between a language model and Claude, nothing else.

The algorithmic gap between a language model and Claude, nothing else.

---

## What this is

A plain LLM predicts the next token. Claude does three additional things on top of that:

| Step | What it does |
|------|-------------|
| **Chat format** | Role tokens `[H]` and `[A]` structure the conversation. The model learns human turns are input, assistant turns are output. |
| **SFT** | Loss is computed only on assistant tokens. The model learns to *respond*, not just predict everything. |
| **DPO** | Given (chosen, rejected) response pairs, the model learns to prefer the better response — no reward model, no PPO. |
| **Constitutional** | At inference: generate a draft → critique it → revise. Three forward passes, zero extra parameters. |

That's the whole delta. Everything else in Claude is scale and infrastructure on top of this.

---

## Usage

No dependencies. Just Python.

```bash
python clod_mini.py
```

On first run it downloads the Anthropic HH-RLHF dataset (~50MB) — real human preference pairs, the same data used to align Claude. Then it trains through three phases and generates sample outputs.

### Options

```
--n_embd      embedding dimension         (default: 64)
--n_layer     transformer layers          (default: 2)
--n_head      attention heads             (default: 4)
--block_size  context length in tokens    (default: 128)
--sft_steps   supervised fine-tune steps  (default: 500)
--dpo_steps   preference alignment steps  (default: 200)
--lr          learning rate               (default: 3e-3)
--beta        DPO KL penalty β            (default: 0.1)
--max_pairs   HH-RLHF pairs to load       (default: 2000)
--seed                                    (default: 42)
```

---

## How it works

### Phase 1 — SFT

Trains on the *chosen* side of each preference pair. Loss is masked to assistant tokens only — human turns flow through the attention context but contribute no gradient. This is what turns a text predictor into a chat model.

### Phase 2 — DPO

Snapshots the post-SFT weights as a frozen reference model `π_ref`. Then fine-tunes the live model on preference pairs using:

```
loss = -log σ( β · [ log π_θ(chosen)   - log π_ref(chosen)
                   - log π_θ(rejected) + log π_ref(rejected) ] )
```

The model is pushed to assign relatively higher probability to chosen responses over rejected ones, measured against where it started. No reward model. No RL loop.

### Phase 3 — Constitutional inference

```python
draft    = generate(human_input)
critique = generate(f"Review this: {draft}")
revised  = generate(human_input, context=critique)
```

Same model. Three calls. The pattern behind Constitutional AI.

---

## Caveats

**It's slow.** Pure Python autograd builds a full computation graph every step. Expect minutes per hundred steps on CPU. This is a reading project, not a training project.

**The model is tiny.** At default settings it won't produce coherent English — the constitutional loop works architecturally but won't produce meaningful self-critique at this scale. Increase `--n_embd` and `--n_layer` if you want to push it further.

**The point is the code.** Every concept in the Claude training pipeline fits in a single file you can step through in a debugger.

---

## Data

Uses the [Anthropic HH-RLHF dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf) — human helpfulness and harmlessness preference pairs. Downloaded automatically on first run as `hh_rlhf.jsonl.gz`.
