# FEAT-002 — RTL Synthesizable Rewrite (Remove shortreal Tech Debt)

## Status
In progress — `fp32_mul.sv` ✅  `fp32_add.sv` ✅  remaining modules queued

## Context

During [FEAT-001](feat-001-fpga-xsim-development-pipeline.md) all RTL modules were written
using `shortreal`, `$bitstoshortreal`, and `$shortrealtobits` as a shortcut to get XSim
simulations running quickly against the C# reference.  These constructs are **simulation-only** —
Vivado synthesis rejects them.  None of the existing RTL is synthesizable for the ZCU102 target.

Reddit user Superb_5194 identified the issue on 2026-06-16.  Full audit confirmed it is
project-wide: every compute module is affected.

Enforcement added as part of this feature:
- CI lint (`rtl-lint.yml`) hard-fails on any of the banned constructs in `rtl/`
- Copilot instructions (`.github/copilot-instructions.md`) prohibit them in all future RTL

---

## Rules (non-negotiable)

All RTL in `rtl/` must comply with the synthesizable subset defined in
`.github/copilot-instructions.md`:

```
Allowed  : logic [N:0], always_ff, always_comb, assign, module/endmodule,
           parameter/localparam, for (genvar ...)
Forbidden: shortreal, real, $bitstoshortreal, $shortrealtobits, $bitstoreal, $realtobits,
           automatic variables inside always_ff, dynamic arrays, queues, classes,
           $display / $finish / $readmemh (RTL only), initial blocks (RTL only)
```

Each FP32 operation must be an instantiated module with explicit `clk`, `rst`,
`valid_in`, `valid_out` ports and a documented fixed latency.

---

## New synthesizable primitives

These are the foundation — everything else is rewired on top of them.

| Module | Operation | Latency | Status |
|---|---|---|---|
| `fp32_mul.sv` | a × b | 3 cycles | ✅ done, 10/10 tests pass |
| `fp32_add.sv` | a + b (subtract via sign) | 4 cycles | ✅ done, tests pass |
| `fp32_sqrt.sv` | 1/√x  (ROM seed + 2× NR) | ~29 cycles | ✅ done |
| `fp32_div.sv` | a / b  (recip × multiply) | rewrite needed | 🔲 |

---

## Module rewrite plan

Dependencies flow top-down.  Fix lower tiers first.

### Tier 1 — Scalar FP32 operators (no dependencies)

#### `fp32_sqrt.sv` — rewrite
Current: `shortreal` NR iterations.
Replace with: ROM seed lookup (keep existing `recipsqrt_rom.hex` + `$readmemh`),
then 2× NR iterations using `fp32_mul` instances.
Latency: 1 (ROM) + 3 (mul1) + 3 (mul2) + 3 (mul3) = ~10 cycles + FSM overhead.
Tests: existing `RecipSqrtTests.cs` + `RecipSqrtVsSoftwareTests.cs`.

#### `fp32_div.sv` — rewrite
Current: `shortreal` division.
Replace with: `fp32_sqrt` (reciprocal of b) + `fp32_mul` (a × recip).
Latency: sqrt_latency + 3 (mul).
Tests: existing `DivTests.cs` + `DivVsSoftwareTests.cs`.

---

### Tier 2 — MAC and add-tree (depend on fp32_mul + fp32_add)

#### `fp32_add_tree.sv` — rewrite
Current: combinatorial `shortreal` 4-input tree.
Replace with: 3× `fp32_add` instances in a 2-stage pipeline.
New port: add `clk`, `rst`, `valid_in`, `valid_out`.
Latency: 4 + 4 = 8 cycles (stage1 two adds in parallel, stage2 one add).
Tests: existing `AddTreeTests.cs`.

#### `bf16_mac.sv` — rewrite
Current: `shortreal` multiply + add in `always_ff`.
Replace with: one `fp32_mul` instance (stage 2) + one `fp32_add` instance (stage 3).
Latency: unchanged externally (3 cycles) but internally driven by submodule `valid_out`.
Tests: existing `Bf16MacTests.cs` + `Bf16MacVsSoftwareTests.cs`.

#### `bf16w_mac.sv` — rewrite (same as bf16_mac)
Same structure, same fix.
Tests: existing tests.

---

### Tier 3 — Matmul (depend on MAC + add-tree)

#### `bf16_matmul.sv` — rewrite
Current: `shortreal` add reduction in `always_ff`.
Replace with: `fp32_add_tree` instances for the reduction stage.
Latency: mac_latency + add_tree_latency.
Tests: `MatMulTests.cs` + `Bf16MatMulVsSoftwareTests.cs`.

#### `bf16w_matmul.sv` — same fix.

#### `fp32_matmul.sv` — rewrite
Current: `shortreal` multiply array.
Replace with: `fp32_mul` instances per element + `fp32_add_tree` reduction.
Tests: `Fp32MatMulTests.cs` + `Fp32MatMulVsSoftwareTests.cs`.

---

### Tier 4 — Activation and normalization functions

#### `exp_lut.sv` — rewrite
Current: `shortreal` interpolation arithmetic.
Replace with: `fp32_mul` + `fp32_add` instances for the LUT interpolation stage.
ROM load via `$readmemh` is allowed (testbench / simulation path).
Tests: `ExpLutTests.cs` + `ExpLutVsSoftwareTests.cs`.

#### `gelu.sv` — rewrite
Current: `shortreal` throughout.
Replace with: `fp32_mul` + `fp32_add` chain + `exp_lut` instance.
Tests: `GeluTests.cs`.

#### `softmax.sv` — rewrite
Current: `shortreal` max-find + subtract + exp + divide.
Replace with: `fp32_add` for max-reduction, `exp_lut`, `fp32_add_tree` for sum, `fp32_div`.
Tests: `SoftmaxTests.cs`.

#### `layernorm.sv` — rewrite
Current: `shortreal` accumulators, mean, variance, scale.
Replace with: `fp32_add`/`fp32_mul` submodule instances, `fp32_sqrt` for invStd.
Tests: `LayerNormTests.cs` + `LayerNormBackwardTests.cs` + VsSoftware variants.

---

### Tier 5 — Optimizer

#### `adam_cell.sv` — rewrite
Current: `shortreal` for entire Adam update (m, v, mHat, vHat, denom, weight update).
Replace with: `fp32_mul`, `fp32_add`, `fp32_sqrt`, `fp32_div` instances.
Note: `localparam shortreal BETA1/BETA2/EPSILON` → replace with `localparam logic [31:0]` bit-pattern constants.
Tests: `AdamCellTests.cs`.

#### `adam_core.sv` — rewrite after adam_cell.

#### `ce_grad.sv` — audit and rewrite if shortreal present.

---

### Tier 6 — Integrators (depend on everything above)

#### `attention_core.sv`
Current: `shortreal` for residual adds + scale.
Replace inline arithmetic with `fp32_add`/`fp32_mul` instances.
Tests: `AttentionCoreTests.cs` + `AttentionCoreVsSoftwareTests.cs` + backward variants.

#### `mlp_core.sv` — same audit.

#### `transformer.sv` / `transformer_train.sv`
Current: `shortreal` for residual adds.
Replace with `fp32_add` instances.
Tests: `TransformerTests.cs` + `TransformerTrainTests.cs` + VsSoftware variants.

---

## Completion checklist

- [x] `fp32_mul.sv` — synthesizable, 10/10 tests pass
- [x] `fp32_add.sv` — synthesizable, tests pass
- [x] CI lint (`rtl-lint.yml`) enforced on every PR
- [x] Copilot instructions updated with synthesizable subset + ZCU102 target
- [x] `fp32_sqrt.sv` rewrite
- [ ] `fp32_div.sv` rewrite
- [ ] `fp32_add_tree.sv` rewrite
- [ ] `bf16_mac.sv` / `bf16w_mac.sv` rewrite
- [ ] `bf16_matmul.sv` / `bf16w_matmul.sv` / `fp32_matmul.sv` rewrite
- [ ] `exp_lut.sv` rewrite
- [ ] `gelu.sv` / `softmax.sv` / `layernorm.sv` rewrite
- [ ] `adam_cell.sv` / `adam_core.sv` / `ce_grad.sv` rewrite
- [ ] `attention_core.sv` / `mlp_core.sv` rewrite
- [ ] `transformer.sv` / `transformer_train.sv` rewrite
- [ ] All existing XSim tests pass after each module rewrite
- [ ] Vivado synthesis run — zero banned-construct errors
