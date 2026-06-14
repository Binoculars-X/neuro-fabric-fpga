# NeuronFabric FPGA — High-Level Block Diagram

> Phase 1 (XSim verification) configuration: T=4, D=4, heads=1, FF=4, L=2, V=16.
> Weights are BF16 (16-bit); all activations and Adam state are FP32 (32-bit).
> No ARM/PS core used — pure PL (programmable logic), driven by host C# over JTAG/PCIe in production.

## Top-Level Datapath

```mermaid
flowchart TD
    HOST["Host (C#)\ntoken indices [T × int32]\nweight load via write port"]

    subgraph TRANSFORMER["transformer.sv — top-level RTL"]
        direction TB

        EMB["Embedding Table\nRAM [V×D × 32-bit FP32]\ntoken → row lookup\n16 × 4 = 64 FP32 words"]

        PE["+ Positional Encoding\nFP32 [T×D = 4×4 = 16 words]\nadded element-wise"]

        X0["x  FP32 [T×D]\n4×4 = 16 × 32-bit"]

        subgraph LAYER["Transformer Layer  (instantiated L=2 times)"]
            direction TB
            LN1["layernorm.sv  LN1\nγ/β: BF16→FP32 [D=4]\nout: FP32 [T×D]"]
            ATTN["attention_core.sv\n(see detail below)"]
            ADD1["Residual Add\nFP32 [T×D]  element-wise"]
            LN2["layernorm.sv  LN2\nγ/β: BF16→FP32 [D=4]\nout: FP32 [T×D]"]
            MLP["mlp_core.sv\n(see detail below)"]
            ADD2["Residual Add\nFP32 [T×D]  element-wise"]

            LN1 --> ATTN --> ADD1 --> LN2 --> MLP --> ADD2
        end

        OUTPROJ["Output Projection\nfp32_matmul.sv\nFP32 [T×D] × FP32 [D×V]\n→ logits FP32 [T×V = 4×16]"]

        EMB --> PE --> X0 --> LAYER --> OUTPROJ
    end

    HOST -->|"write port: weights (BF16 16-bit)\ntokens (int32)"| TRANSFORMER
    OUTPROJ -->|"logits FP32 [T×V]"| HOST
```

## attention_core.sv — Internal Datapath

```mermaid
flowchart TD
    xin["x_norm  FP32 [T×D]\n4×4 = 16 × 32-bit"]

    subgraph ATTN_CORE["attention_core.sv"]
        direction TB

        WQ["Wq  BF16 [D×DH]\n4×4 = 16 × 16-bit"]
        WK["Wk  BF16 [D×DH]"]
        WV["Wv  BF16 [D×DH]"]
        WO["Wo  BF16 [DH×D]"]

        MQ["bf16w_matmul.sv\nFP32[T×D] × BF16[D×DH]\n→ Q  FP32[T×DH]"]
        MK["bf16w_matmul.sv\n→ K  FP32[T×DH]"]
        MV["bf16w_matmul.sv\n→ V  FP32[T×DH]"]

        SCORE["fp32_matmul.sv\nQ × Kᵀ\nFP32[T×DH] × FP32[DH×T]\n→ scores  FP32[T×T]"]

        SOFTMAX["softmax.sv\ncausal mask + exp_lut + sum + div\n→ A  FP32[T×T]"]

        AV["fp32_matmul.sv\nA × V\nFP32[T×T] × FP32[T×DH]\n→ ctx  FP32[T×DH]"]

        MO["bf16w_matmul.sv\nFP32[T×DH] × BF16[DH×D]\n→ out  FP32[T×D]"]

        WQ --> MQ
        WK --> MK
        WV --> MV
        MQ & MK --> SCORE --> SOFTMAX --> AV
        MV --> AV
        AV --> MO
        WO --> MO
    end

    xin --> ATTN_CORE
    MO -->|"FP32 [T×D]"| out["attn_out"]
```

## softmax.sv — Internal Datapath

```mermaid
flowchart LR
    scores["scores FP32[T]\n(one row, causal masked)"]
    EXPLUT["exp_lut.sv\n256-entry BRAM ROM\n2^(i/255) BF16 init\n→ exp(x)  FP32[T]"]
    ADDTREE["fp32_add_tree.sv\nFP32 reduction sum\n→ sum  FP32"]
    DIV["fp32_div.sv\nNewton–Raphson FP32÷FP32\n→ 1/sum  FP32"]
    MUL["× each exp output\n→ A[row]  FP32[T]"]

    scores --> EXPLUT --> ADDTREE --> DIV --> MUL
    EXPLUT --> MUL
```

## mlp_core.sv — Internal Datapath

```mermaid
flowchart TD
    xnorm2["x_norm2  FP32 [T×D]"]

    subgraph MLP_CORE["mlp_core.sv"]
        WFF1["Wff1  BF16 [D×FF]\n4×4 = 16 × 16-bit"]
        WFF2["Wff2  BF16 [FF×D]"]
        M1["bf16w_matmul.sv\nFP32[T×D] × BF16[D×FF]\n→ H1  FP32[T×FF]"]
        GELU["gelu.sv\n0.5·x·(1+tanh(c·(x+0.044715x³)))\ntanh via LUT-256\n→ G  FP32[T×FF]"]
        M2["bf16w_matmul.sv\nFP32[T×FF] × BF16[FF×D]\n→ Y  FP32[T×D]"]

        WFF1 --> M1 --> GELU --> M2
        WFF2 --> M2
    end

    xnorm2 --> MLP_CORE
    M2 -->|"FP32 [T×D]"| mlp_out["mlp_out"]
```

## layernorm.sv — Internal Datapath

```mermaid
flowchart LR
    xrow["x[D]  FP32\none token row"]
    MEAN["Σx / D\nrunning FP32 adder\n→ μ  FP32"]
    VAR["Σ(x−μ)² / D\nD-entry shift reg buffer\n→ σ²  FP32"]
    RSQRT["fp32_sqrt.sv\nseed ROM 256-entry\n+ 2× Newton–Raphson\n→ 1/√(σ²+ε)  FP32"]
    SCALE["γ[d]·(x[d]−μ)·invStd + β[d]\nγ/β  BF16→FP32 [D]\n→ y[d]  FP32"]

    xrow --> MEAN --> VAR --> RSQRT --> SCALE
    xrow --> SCALE
```

## adam_core.sv / adam_cell.sv — Update Datapath

```mermaid
flowchart TD
    subgraph ADAM_CORE["adam_core.sv  (R×C iterations, one adam_cell shared)"]
        direction TB

        subgraph ADAM_CELL["adam_cell.sv — one scalar parameter"]
            direction LR
            G["g  FP32\ngradient"]
            W["w  BF16 16-bit\ncurrent weight"]
            M["m  FP32\n1st moment"]
            V["v  FP32\n2nd moment"]

            MOMENTS["β₁·m+(1−β₁)·g\nβ₂·v+(1−β₂)·g²\nshortreal mul+add\n→ m_new, v_new  FP32"]
            DIV1["fp32_div.sv\nm / bc1 → m̂  FP32"]
            DIV2["fp32_div.sv\nv / bc2 → v̂  FP32"]
            SQRT["fp32_sqrt.sv\n1/√v̂  FP32\n+ shortreal mul → √v̂"]
            DIV3["fp32_div.sv\nlr·m̂ / (√v̂+ε)  FP32"]
            ENC["BF16 encode\ntruncate bits[15:0]\n→ w_new  BF16 16-bit"]

            G & M --> MOMENTS
            G & V --> MOMENTS
            W -->|"BF16 decode\n{w[15],w[14:7],w[6:0],16'b0}"| DIV3
            MOMENTS --> DIV1 --> DIV3
            MOMENTS --> DIV2 --> SQRT --> DIV3
            DIV3 --> ENC
        end

        STATE["Per-param state RAM\nw BF16 | m FP32 | v FP32\n= 10 bytes / param\nR×C params total"]
    end

    BC["bc1=(1−β₁ᵗ), bc2=(1−β₂ᵗ)\nlr  — computed by caller (host)\nFP32 scalars"]
    GRAD["grad[R×C]  FP32\nfrom backward pass"]

    BC --> ADAM_CORE
    GRAD --> ADAM_CORE
    STATE <-->|"read/write each cycle"| ADAM_CELL
```

## Bit-Width Summary

| Signal class | Width | Format | Notes |
|---|---|---|---|
| Activations (x, Q, K, V, A, H1, G, logits) | 32-bit | IEEE 754 FP32 | All intermediate activations |
| Weights (Wq/Wk/Wv/Wo/Wff1/Wff2/LN γ/β) | 16-bit | BF16 | Stored/updated as BF16; decoded to FP32 on read |
| Adam 1st moment m | 32-bit | FP32 | Full precision maintained between steps |
| Adam 2nd moment v | 32-bit | FP32 | Full precision maintained between steps |
| Token indices | 32-bit | int | Host → Embedding lookup |
| Embedding table entries | 32-bit | FP32 | Weight-tied with output projection |
| exp LUT entries | 16-bit | BF16 | 256-entry BRAM ROM, `2^(i/255)` |
| fp32_sqrt seed ROM | 32-bit | FP32 | 256-entry, indexed by mantissa top 8 bits |

## No ARM/PS Core

All compute is in the **PL (programmable logic)**. The host C# process communicates
over JTAG (XSim simulation: file I/O hex vectors). No Zynq PS core, no AXI bus, no
embedded Linux. This is pure RTL: token indices and weights arrive via a synchronous
write-port FSM; logits and updated weights leave via a read-port FSM.
