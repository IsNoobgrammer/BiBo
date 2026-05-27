# BiBo Benchmark — Architecture

## System Overview

```mermaid
graph TD
    subgraph DataPipeline["Data Pipeline"]
        HF_Dataset["HF Dataset<br/>tinycompany/Instruct-packed-2K"]
        DataLoader["DataLoader<br/>batch_size × 1024"]
        HF_Dataset -->|"load_dataset()"| DataLoader
    end

    subgraph ModelArch["Model Architecture"]
        BiBoConfig["BiBoConfig<br/>~50M params"]
        BiBoModel["BiBoForCausalLM<br/>BiBoModel + LM Head"]
        BiBoConfig -->|"from_pretrained()"| BiBoModel
    end

    subgraph Training["Training Loop"]
        Forward["Forward Pass"]
        LossCE["Cross-Entropy Loss"]
        Backward["Backward Pass"]
        OptimStep["Optimizer Step<br/>Muon + AdamW"]
        LRSched["LR Scheduler"]
        Forward --> LossCE --> Backward --> OptimStep --> LRSched
        LRSched -->|"next step"| Forward
    end

    subgraph Distributed["Multi-GPU"]
        FSDP2["FSDP2<br/>2×T4"]
        NCCL["NCCL Backend"]
        FSDP2 --> NCCL
    end

    subgraph Logging["Logging & Eval"]
        WandB["Weights & Biases<br/>loss, lr, throughput"]
        Samples["Sample Generation<br/>QTK-81K tokenizer"]
        Checkpoints["Checkpointing<br/>every N steps"]
        WandB --> Samples
    end

    DataLoader -->|"input_ids, labels"| Forward
    BiBoModel -->|"logits"| Forward
    FSDP2 -->|"shard model"| BiBoModel
    OptimStep -->|"log metrics"| WandB
    LRSched -->|"every M steps"| Samples
    OptimStep -->|"every K steps"| Checkpoints
```

---

## MoE Layer Architecture (Baseline BiBo)

```mermaid
graph TD
    subgraph MoELayer["BiBoMoELayer (per layer)"]
        Input["Hidden State<br/>[batch, seq, 256]"]
        
        subgraph Router["BiBoMoERouter (MLP)"]
            R_Linear["Linear(256 → num_experts)"]
            R_Sigmoid["Sigmoid Gate"]
            R_TopK["Top-K Selection<br/>k=2"]
            R_Linear --> R_Sigmoid --> R_TopK
        end

        subgraph Experts["Routed Experts"]
            E1["SwiGLU Expert 0"]
            E2["SwiGLU Expert 1"]
            E3["SwiGLU Expert 2"]
        end

        subgraph Shared["Shared Expert"]
            SE["SwiGLU MLP<br/>(always active)"]
        end

        Combine["Weighted Sum<br/>+ Shared Expert"]
        Output["Output Hidden State"]

        Input --> Router
        Router -->|"route to top-2"| Experts
        Input --> Shared
        Experts --> Combine
        Shared --> Combine
        Combine --> Output
    end
```

---

## Data Flow

```mermaid
sequenceDiagram
    participant D as DataLoader
    participant M as BiBoForCausalLM
    participant O as Optimizer
    participant W as WandB
    participant T as QTK-81K Tokenizer

    D->>M: input_ids [B, 1024], labels [B, 1024]
    M->>M: Forward pass (embedding → layers → lm_head)
    M->>M: Cross-entropy loss (logits vs labels)
    M->>O: loss.backward() + optimizer.step()
    O->>W: Log metrics (loss, lr, grad_norm, tokens/sec)
    
    alt Every M steps
        M->>T: Generate from fixed prompts
        T->>W: Log sample text as WandB Table
    end

    alt Every N steps
        M->>M: Validation loss on held-out split
        M->>W: Log val/loss, val/perplexity
    end
```

---

## File Structure

```
bench/
├── train.py           # Main entry: arg parsing, training loop, FSDP2 setup
├── config.py          # BiBoConfig for ~50M baseline (no PolyGLU)
├── data.py            # Dataset loading, truncation to 1024, train/val split
├── optim.py           # Muon + AdamW setup, LR scheduler
├── eval.py            # Eval loop + sample generation with QTK-81K
├── utils.py           # WandB init, checkpointing, metrics, helpers
├── run.sh             # Kaggle entry point (pip install, launch training)
└── README.md          # Usage, config options, expected results
```

**File responsibilities:**

| File | Owns | Imports From |
|------|------|-------------|
| `train.py` | Training loop, arg parsing, FSDP2 wrapping, torch.compile | config, data, optim, eval, utils |
| `config.py` | BiBoConfig initialization, param count validation | src.configuration_bibo |
| `data.py` | HF dataset loading, truncation, DataLoader creation | datasets, torch |
| `optim.py` | Optimizer creation (Muon/AdamW), LR scheduler | torch.optim, kellerjordan (optional) |
| `eval.py` | Validation loss computation, sample generation | transformers (tokenizer), torch |
| `utils.py` | WandB init/logging, checkpointing, gradient clipping | wandb, torch |

---

## Deployment (Kaggle)

```mermaid
graph LR
    subgraph Kaggle["Kaggle Notebook"]
        Setup["!pip install<br/>hf_transfer wandb torch"]
        Clone["git clone BiBo repo"]
        Run["python bench/train.py"]
        Setup --> Clone --> Run
    end

    subgraph GPU["Hardware"]
        T4_0["T4 #0<br/>cuda:0"]
        T4_1["T4 #1<br/>cuda:1"]
    end

    subgraph External["External"]
        HF["HuggingFace Hub<br/>dataset + tokenizer"]
        WDB["Weights & Biases<br/>logging server"]
    end

    Run -->|"FSDP2"| T4_0
    Run -->|"FSDP2"| T4_1
    Run -->|"hf_transfer"| HF
    Run -->|"wandb"| WDB
```

---

## Key Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| FSDP2 vs DDP | FSDP2 | Shards model across GPUs, lower memory per GPU, better for scaling to larger models |
| torch.compile mode | reduce-overhead | Best throughput for fixed-shape inputs (packed sequences) |
| fullgraph=False | Required | MoE routing is data-dependent (top-k selection breaks fullgraph) |
| Muon + AdamW | Hybrid | Muon for embeddings (proven on NanoGPT), AdamW for rest (stable for MoE) |
| Dataset truncation | 1024 from 2048 | User requirement: cap at 1024 starting tokens |
| Tokenizer usage | Inference only | Data is pre-tokenized, tokenizer only for decode/samples |
| Gradient clipping | max_norm=1.0 | Standard for MoE training, prevents exploding gradients |
| Mixed precision | fp16 (amp) | T4 has no native bf16, fp16 via autocast is stable |
