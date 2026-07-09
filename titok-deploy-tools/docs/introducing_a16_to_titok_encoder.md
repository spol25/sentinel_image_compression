# Introducing Selective A16W8 to the TiTok Encoder

This note summarizes the mixed A16W8/A8W8 work for the source-BHLD TiTok-S-128 encoder. The goal was to recover quality lost by the fully A8W8 encoder while keeping the expensive BMMs, linears, and attention matmuls compatible with the Ethos-U target.

## Why We Tried This

The fully A8W8 encoder lowered cleanly, but reconstruction quality was poor. A fully A16W8-style approach looked better on host, but A16W8 BMMs are not supported for the final Ethos-U board path. So the working hypothesis was:

> Keep compute-heavy operators A8W8, but make numerically sensitive boundaries A16W8.

The first targets were deliberately coarse and low-risk:

```text
image prologue/tail
  -> transformer norms
  -> residual boundary adds
  -> small MLP/activation boundary experiments
```

## Attention Variant Context

Before the A16 work, we first tried a source-level BHLD SDPA attention rewrite. That path could be exported and lowered, including with `Dedicated_Sram_384KB`, but it was less useful for the next round of debugging because the attention stayed wrapped as SDPA-style behavior.

We then switched to the source-level BHLD matmul attention variant. That made the attention dataflow explicit enough to reason about BMM delegation, Q/DQ placement, host-vs-board drift, and selective A16 boundaries.

```text
source BHLD SDPA
  -> lowers, but attention is less explicit

source BHLD matmul
  -> explicit attention matmul/BMM structure
  -> 16 BMMs can be tracked through partitioning
  -> better fit for selective A16W8 experiments
```

All results below use `source_matmul_attention` unless explicitly stated otherwise.

## What Changed

The best practical profile so far is:

```text
global A8W8
+ encoder.patch_embed
+ encoder.ln_pre
+ encoder.ln_post
+ encoder.conv_out
+ encoder.transformer.{0..7}.ln_1
+ encoder.transformer.{0..7}.ln_2
+ attn_residual_adds.{0..7}
+ mlp_residual_adds.{0..7}
```

We made residual adds explicit modules so the quantizer could target them by name:

```text
x = attn_residual_add(x, attention_update)
x = mlp_residual_add(x, mlp_update)
```

The important constraint stayed intact:

```text
attention BMMs: A8W8
MLP linears:   A8W8
attention matmuls: A8W8
```

## Quality Progression

These metrics compare the float encoder reconstruction to the quantized encoder reconstruction unless noted otherwise.

| Profile | PSNR | SSIM | Latent cosine | Latent MAE | VQ exact | VQ top5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A8W8 baseline | 14.6199 | 0.5122 | 0.2634 | 0.2500 | 0.0016 | 0.0133 |
| Surface A16W8 | 16.5619 | 0.6036 | 0.3276 | 0.2372 | 0.0078 | 0.0266 |
| Surface + norms A16W8 | 18.1213 | 0.6667 | 0.4482 | 0.2074 | 0.0180 | 0.0477 |
| Surface + norms + residual adds A16W8 | 18.2515 | 0.6946 | 0.5135 | 0.2017 | 0.0180 | 0.0719 |

The residual-add profile is the best balanced result so far: large quality improvement over A8W8, still close to the A8W8 lowering footprint, and no BMM delegation loss.

Visual comparison:

![A8W8 vs residual-add A16W8 triptychs](../outputs/full_bhld_a8w8_vs_residual_add_a16w8_triptychs/a8w8_vs_residual_add_a16w8_eval_triptych_contact_sheet.png)

## Lowering Result

The residual-add A16W8 profile lowered successfully.

| Metric | A8W8 | Residual-add A16W8 | Delta |
| --- | ---: | ---: | ---: |
| PTE size | 26,302,384 B | 26,317,824 B | +15,440 B |
| BMMs delegated | 16 / 16 | 16 / 16 | unchanged |
| Delegate islands | 1 | 1 | unchanged |
| Vela CPU ops | 0 | 0 | unchanged |
| Runtime CPU kernels | input quant + output dequant | input quant + output dequant | unchanged |
| Vela NPU ops | 1,156 | 1,321 | +165 |
| DRAM bandwidth | 4,625.50 MB/inf | 4,695.13 MB/inf | +69.63 |
| MACs/batch | 11.039B | 11.059B | +20.4M |

Board smoke result for one example:

- CM33 invoke status: `0`
- Ethos-U IRQ count: `1`
- `method->execute` time: `11,359.948 ms`
- Output shape: `[1, 12, 1, 128]`
- Output was finite and on the expected dequant grid

Board reconstruction triptych:

![Board latent host VQ reconstruction](../outputs/full_bhld_surface_transformer_norm_residual_a16w8_board_triptych/000_0000_day_far_S1_C07_R2_PICT2057_ref_residual_host_board_triptych.png)

Note: the board output was structurally valid, but did not closely match the host latent for the same image. That remains a separate host-vs-board numerical mismatch to investigate.

## What We Decided Against

### MLP output projection A16W8

Targeting `encoder.transformer.{0..7}.mlp.c_proj` improved host quality:

| Profile | PSNR | SSIM | Latent cosine | VQ exact | VQ top5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Residual-add A16W8 | 18.2515 | 0.6946 | 0.5135 | 0.0180 | 0.0719 |
| + MLP output projection A16W8 | 18.5282 | 0.7539 | 0.5841 | 0.0383 | 0.1211 |

But lowering cost exploded:

| Metric | Residual-add A16W8 | + MLP output projection A16W8 |
| --- | ---: | ---: |
| PTE size | 26.32 MB | 35.27 MB |
| Vela NPU ops | 1,321 | 444,071 |
| DRAM bandwidth | 4,695.13 MB/inf | 5,081.13 MB/inf |

So this is promising for quality, but not acceptable as an all-block lowering strategy.

### Identity MLP boundaries

We tried explicit post-MLP and post-GELU boundary modules without selecting the large projection itself. They were effectively no-ops: same metrics as residual-add A16W8.

### GELU A16W8

Targeting GELU directly improved latent/token metrics slightly but hurt decoded PSNR/SSIM:

| Profile | PSNR | SSIM | Latent cosine | VQ exact | VQ top5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Residual-add A16W8 | 18.2515 | 0.6946 | 0.5135 | 0.0180 | 0.0719 |
| + GELU A16W8 | 17.5894 | 0.6780 | 0.5205 | 0.0242 | 0.0914 |

After adding AlexNet LPIPS, the input-reconstruction comparison points the same way. Lower LPIPS is better:

| Profile | Input LPIPS alex | Input PSNR | Input MAE | LPIPS vs float decoded |
| --- | ---: | ---: | ---: | ---: |
| Residual-add A16W8 | 0.777382 | 18.9580 | 0.091090 | 0.747033 |
| + GELU A16W8 | 0.803789 | 18.1672 | 0.100508 | 0.776762 |

So GELU A16W8 is not preferred: it makes latent/token agreement look slightly better, but the decoded image gets worse against the original input, including on LPIPS.

## Saturation Audit

We audited quantized activations for:

- float min/max before quant
- quantized min/max after quant
- `% at qmin`
- `% at qmax`
- `% outside dequant range`
- quantization MAE/cosine

The main finding was narrow:

```text
Most true clipping is at MLP GELU outputs.
Residual-add A16W8 reduces several non-GELU clipping points.
Final latent output is not clipping outside range.
```

Worst residual-add GELU output clipping:

| Tensor | % outside dequant range |
| --- | ---: |
| `gelu_5` | 22.60% |
| `gelu_6` | 18.39% |
| `gelu_2` | 13.77% |
| `gelu_1` | 12.65% |

We then widened only those GELU output scales:

| Variant | GELU outside range | Float-vs-quant PSNR | Input PSNR | Input SSIM |
| --- | ---: | ---: | ---: | ---: |
| Residual-add baseline | nonzero | 18.2515 | 18.9580 | 0.7393 |
| GELU scale x1.25 | 0% | 17.8380 | 18.1071 | 0.7246 |
| GELU scale x1.5 | 0% | 17.6301 | 18.2425 | 0.7310 |
| GELU scale x2.0 | 0% | 18.1015 | 18.4273 | 0.7318 |

Widening fixed the measured clipping but did not improve input reconstruction. The likely lesson is that GELU clipping is real, but not the dominant quality limiter by itself; resolution loss from widening can hurt more than clipping removal helps.

For reference, the float encoder reconstruction against input is:

| Model | Input PSNR | Input SSIM | Input MAE |
| --- | ---: | ---: | ---: |
| Float encoder | 22.4988 | 0.8492 | 0.061742 |
| Residual-add A16W8 | 18.9580 | 0.7393 | 0.091090 |

## Current Recommendation

Use `int8_surface_transformer_norm_residual_a16w8` as the current working profile.

Why:

- It gives a large quality jump over A8W8.
- All 16 BMMs remain Ethos-U delegated.
- Vela accepts the graph with zero CPU operators.
- Runtime CPU kernels remain only input quantize and output dequantize.
- PTE size and resource estimates stay close to A8W8.

Open items:

- Investigate host-vs-board output mismatch.
- If more quality is needed, explore smaller or per-block MLP projection targeting rather than enabling all `mlp.c_proj` modules.
- Treat GELU range widening as diagnostic, not a preferred model change.

## Detailed Logs

- Full experiment log: `outputs/mixed_a16w8_quality_experiments.md`
- Activation audit: `outputs/activation_saturation_audit_summary.md`
- A8W8 vs residual visuals: `outputs/full_bhld_a8w8_vs_residual_add_a16w8_triptychs/`
- Input reconstruction metrics: `outputs/full_bhld_surface_transformer_norm_residual_a16w8_gelu_range_widening_input_metrics/input_reconstruction_metrics.md`
- Residual vs GELU A16W8 LPIPS metrics: `outputs/full_bhld_residual_vs_gelu_a16w8_lpips_input_metrics/README.md`
