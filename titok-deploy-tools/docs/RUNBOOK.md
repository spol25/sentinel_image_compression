# TiTok Deploy Runbook

This runbook is the operational handoff for running TiTok / fallback image
compression artifacts through the UCM-i.MX93 + Ethos-U65 path.

It covers:

1. PTQ preparation and calibration.
2. Lowering TiTok encoder variants.
3. Building and using the CM33 ExecuTorch runner.
4. Loading `.pte` models into reserved DDR with U-Boot `fatload`.
5. Starting CM33 through Linux `remoteproc`.
6. Capturing inputs/outputs and debugging failures.
7. Running the fallback solution.
8. Local changes made to TiTok deploy tooling, the CM33 runner, and
   ExecuTorch.

## Required Repositories

Use these forks and pinned commits. Upstream checkouts do not include every
change required by this workflow.

| Purpose | Repository | Branch | Pinned commit |
| --- | --- | --- | --- |
| TiTok model and patched attention helpers | [`spol25/1d-tokenizer`](https://github.com/spol25/1d-tokenizer) | `main` | `ba028d08fbce1c7a03f3661b7f1e17b54c03548f` |
| ExecuTorch portable kernel-only build support | [`spol25/executorch`](https://github.com/spol25/executorch) | `main` | `dd873c3e8ccd1d5b0af5693a583d69b8d9ab5bc3` |
| UCM-i.MX93 CM33 runner | [`spol25/Executorch_runner_cm33`](https://github.com/spol25/Executorch_runner_cm33) | `main` | `1842b9b28d014e8fcaf780f7fab193cc4dabe247` |

Clone and pin them before following the commands below:

```bash
git clone https://github.com/spol25/1d-tokenizer.git
git -C 1d-tokenizer checkout ba028d08fbce1c7a03f3661b7f1e17b54c03548f

git clone https://github.com/spol25/executorch.git executorch-main
git -C executorch-main checkout dd873c3e8ccd1d5b0af5693a583d69b8d9ab5bc3

git clone https://github.com/spol25/Executorch_runner_cm33.git
git -C Executorch_runner_cm33 checkout 1842b9b28d014e8fcaf780f7fab193cc4dabe247
```

## Known-Good Local Paths

The paths below record the host used during bring-up. They are examples, not
required installation locations; substitute the paths to your own clones.

Host workspace:

```text
/Users/sruthipolali/Documents/Playground
```

Deploy tools:

```text
/Users/sruthipolali/Documents/Playground/sentinel_image_compression/titok-deploy-tools
```

Patched TiTok fork:

```text
/Users/sruthipolali/Documents/Playground/1d-tokenizer
```

CM33 runner repo:

```text
/Users/sruthipolali/Documents/Playground/Executorch_runner_cm33
```

Local ExecuTorch checkout:

```text
/Users/sruthipolali/Documents/Playground/executorch-main
```

Default Python:

```text
/Users/sruthipolali/Documents/Playground/1d-tokenizer/.venv/bin/python
```

Snapshot Serengeti calibration manifest:

```text
/Volumes/Media/snapshot_serengeti_balanced_ptq_dataset/manifests/calibration_manifest.json
```

## Board Facts

Board:

```text
CompuLab UCM-i.MX93
```

Active DTB used during bring-up:

```text
ucm-imx93-ethosu.dtb
```

Ethos-U device expected in Linux:

```text
/dev/ethosu0
```

Remoteproc path:

```text
/sys/class/remoteproc/remoteproc0
```

Trace path:

```text
/sys/kernel/debug/remoteproc/remoteproc0/trace0
```

Serial console:

```text
/dev/cu.usbserial-02BE3471
```

First FAT boot partition in Linux:

```text
/run/media/boot-mmcblk0p1
```

U-Boot prompt seen on this board:

```text
ucm-imx93=>
```

## PTQ Prep and Calibration

Use the pinned TiTok fork listed in **Required Repositories**. The lowering
flow depends on attention-layout changes and chunked BHLD helpers that are not
available from the upstream repository.

### Build Image Manifests

For a directory of representative images:

```bash
cd /Users/sruthipolali/Documents/Playground/sentinel_image_compression/titok-deploy-tools

/Users/sruthipolali/Documents/Playground/1d-tokenizer/.venv/bin/python \
  scripts/ptq/prepare_image_manifest.py \
  --image-dir /path/to/representative/images \
  --output-dir outputs/ptq
```

Split calibration and eval sets:

```bash
/Users/sruthipolali/Documents/Playground/1d-tokenizer/.venv/bin/python \
  scripts/ptq/split_calibration_eval_manifests.py \
  --manifest outputs/ptq/image_manifest.json \
  --output-dir outputs/ptq \
  --eval-count 32 \
  --shuffle \
  --seed 0
```

### Run Float Baseline Tokens

```bash
/Users/sruthipolali/Documents/Playground/1d-tokenizer/.venv/bin/python \
  scripts/ptq/run_s128_calibration_baseline.py \
  --titok-root /Users/sruthipolali/Documents/Playground/1d-tokenizer \
  --manifest outputs/ptq/calibration_manifest.json \
  --output-dir outputs/ptq \
  --encoder-variant source_matmul_attention
```

The baseline writes:

```text
outputs/ptq/s128_float_baseline_tokens.json
```

### Run PTQ Experiment

```bash
/Users/sruthipolali/Documents/Playground/1d-tokenizer/.venv/bin/python \
  scripts/ptq/run_encoder_ptq_experiment.py \
  --titok-root /Users/sruthipolali/Documents/Playground/1d-tokenizer \
  --manifest outputs/ptq/calibration_manifest.json \
  --output-dir outputs/ptq \
  --encoder-variant source_matmul_attention \
  --per-channel
```

Compare token outputs:

```bash
/Users/sruthipolali/Documents/Playground/1d-tokenizer/.venv/bin/python \
  scripts/ptq/compare_token_outputs.py \
  --reference outputs/ptq/s128_float_baseline_tokens.json \
  --candidate outputs/ptq/s128_encoder_ptq_tokens.json \
  --output-dir outputs/ptq
```

Compare decoded reconstructions:

```bash
/Users/sruthipolali/Documents/Playground/1d-tokenizer/.venv/bin/python \
  scripts/ptq/compare_decoded_reconstructions.py \
  --titok-root /Users/sruthipolali/Documents/Playground/1d-tokenizer \
  --reference outputs/ptq/s128_float_baseline_tokens.json \
  --candidate outputs/ptq/s128_encoder_ptq_tokens.json \
  --output-dir outputs/ptq
```

PTQ acceptance should be based on:

- token exact/top-k agreement against the float wrapper,
- decoded reconstruction quality,
- visual inspection of failure cases,
- whether the lowered graph still has the intended Ethos-U delegation.

## Lowering TiTok Encoder Variants

Main lowering script:

```text
scripts/export_and_lower/lower_ethosu_titok_s128_encoder.py
```

Important encoder variants:

```text
baseline
einsum_attention
reshape_batch
bmm_attention
source_matmul_attention
source_sdpa_attention
```

The current preferred TiTok path is `source_matmul_attention` with mixed
A16W8/A8W8 quantization profile:

```text
int8_surface_transformer_norm_residual_a16w8
```

The current practical recommendation is documented in:

```text
docs/introducing_a16_to_titok_encoder.md
```

### Lower Source-Matmul Attention

```bash
cd /Users/sruthipolali/Documents/Playground/sentinel_image_compression/titok-deploy-tools

/Users/sruthipolali/Documents/Playground/1d-tokenizer/.venv/bin/python \
  scripts/export_and_lower/lower_ethosu_titok_s128_encoder.py \
  --titok-root /Users/sruthipolali/Documents/Playground/1d-tokenizer \
  --manifest /Volumes/Media/snapshot_serengeti_balanced_ptq_dataset/manifests/calibration_manifest.json \
  --calibration-count 500 \
  --output-dir outputs/current_titok_lowering/source_matmul_residual_a16w8 \
  --summary-name lowering_summary.json \
  --artifact-name titok_s128_encoder_source_matmul_residual_a16w8.pte \
  --encoder-variant source_matmul_attention \
  --quantization-profile a16w8 \
  --per-channel \
  --ethos-target ethos-u65-256 \
  --ethos-system-config Ethos_U65_High_End \
  --ethos-memory-mode Dedicated_Sram_384KB \
  --ethos-config-ini Arm/vela.ini \
  --post-partition-qdq-out-fix
```

For a faster smoke test, reduce `--calibration-count` to `4`.

### Lower Source-SDPA Attention

This path was useful historically because it lowered cleanly, but it leaves
attention less explicit than the source-matmul path.

```bash
/Users/sruthipolali/Documents/Playground/1d-tokenizer/.venv/bin/python \
  scripts/export_and_lower/lower_ethosu_titok_s128_encoder.py \
  --titok-root /Users/sruthipolali/Documents/Playground/1d-tokenizer \
  --manifest /Volumes/Media/snapshot_serengeti_balanced_ptq_dataset/manifests/calibration_manifest.json \
  --calibration-count 4 \
  --output-dir outputs/current_titok_lowering/source_sdpa_smoke \
  --summary-name lowering_summary.json \
  --artifact-name titok_s128_encoder_source_sdpa_smoke.pte \
  --encoder-variant source_sdpa_attention \
  --quantization-profile int8 \
  --per-channel \
  --ethos-target ethos-u65-256 \
  --ethos-system-config Ethos_U65_High_End \
  --ethos-memory-mode Dedicated_Sram_384KB \
  --ethos-config-ini Arm/vela.ini
```

### Vela / Compile-Spec Settings Used

Common flags:

```text
--accelerator-config=ethos-u65-256
--config=Arm/vela.ini
--output-format=raw
--debug-force-regor
--system-config=Ethos_U65_High_End
```

Memory modes tested:

```text
Shared_Sram
Dedicated_Sram
Dedicated_Sram_384KB
```

`Dedicated_Sram_384KB` maps the fast-memory/cache planned buffer to the CM33
runner's OCRAM fast-memory region:

```text
0x20480000 size 0x00060000
```

## CM33 Runner Build

Runner repo:

```bash
cd /Users/sruthipolali/Documents/Playground/Executorch_runner_cm33
```

Build environment:

```bash
env \
  ARMGCC_DIR=/Users/sruthipolali/.mcuxpressotools/arm-gnu-toolchain-14.2.rel1-darwin-arm64-arm-none-eabi \
  SdkRootDirPath=/Volumes/Media \
  MCUX_VENV_PATH=/Users/sruthipolali/.mcuxpressotools/.mcux-venv-3.12/bin \
  BOARD=mcimx93evk \
  cmake --preset debug

env \
  ARMGCC_DIR=/Users/sruthipolali/.mcuxpressotools/arm-gnu-toolchain-14.2.rel1-darwin-arm64-arm-none-eabi \
  SdkRootDirPath=/Volumes/Media \
  MCUX_VENV_PATH=/Users/sruthipolali/.mcuxpressotools/.mcux-venv-3.12/bin \
  BOARD=mcimx93evk \
  cmake --build --preset debug
```

Output ELF:

```text
/Users/sruthipolali/Documents/Playground/Executorch_runner_cm33/debug/executorch_runner_cm33.elf
```

### Selective Portable Helper Ops

If TiTok leaves CPU helper ops outside the Ethos-U delegate, configure the
runner with a `.pte` so CMake can generate a selective portable op registry:

```bash
env \
  ARMGCC_DIR=/Users/sruthipolali/.mcuxpressotools/arm-gnu-toolchain-14.2.rel1-darwin-arm64-arm-none-eabi \
  SdkRootDirPath=/Volumes/Media \
  MCUX_VENV_PATH=/Users/sruthipolali/.mcuxpressotools/.mcux-venv-3.12/bin \
  BOARD=mcimx93evk \
  cmake --preset debug \
    -DET_DIR_PATH=/Users/sruthipolali/Documents/Playground/executorch-main \
    -DEXECUTORCH_SELECT_OPS_MODEL=/absolute/path/to/model.pte
```

You can also pass a semicolon-separated explicit op list:

```bash
cmake --preset debug \
  -DET_DIR_PATH=/Users/sruthipolali/Documents/Playground/executorch-main \
  -DEXECUTORCH_SELECT_OPS_LIST='aten::unsqueeze_copy.out'
```

## UCM Memory Map

The live UCM DTB exposes one reserved Ethos-U DDR carveout:

```text
ethosu_region@0xC0000000
base: 0xC0000000
size: 0x10000000 / 256 MB
```

The CM33 runner was retargeted to keep the model and runtime buffers inside
that carveout.

Current runner layout:

| Region | Start | Size | Purpose |
| --- | ---: | ---: | --- |
| model window | `0xC0000000` | `0x08000000` / 128 MB | `.pte` loaded by U-Boot `fatload` |
| optional input blob | `0xC7800000` | small | host/board input tensor blob, inside high end of model window |
| scratch pool | `0xC8000000` | `0x01000000` / 16 MB | temporary allocator / Ethos-U scratch |
| method allocator | `0xC9000000` | `0x03C00000` / 60 MB | ExecuTorch method allocator |
| planned buffers | `0xCCC00000` | `0x03400000` / 52 MB | large memory-planned tensors |
| optional output blob | `0xCF000000` | up to 1 MB | first output tensor capture |
| Ethos-U fast memory | `0x20480000` | `0x00060000` / 384 KB | `Dedicated_Sram_384KB` fast/cache memory |

Important caveats:

- The input blob starts at `0xC7800000`, so a `.pte` must fit below that if an
  input blob is also loaded. That gives about 120 MB of practical model space
  when using external input capture.
- The older runner repo runbook still says the model window is 4 MB. That is
  stale for TiTok. The current source reserves 128 MB.
- The old EVK-style `0xA8000000` working region is not the UCM layout used by
  the current runner.

Compile-time definitions in the runner:

```text
ET_MODEL_PTE_ADDR=0xC0000000
ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE=0x1000000
ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE=0x03C00000
ET_NUM_INFERENCES=1
```

## Transfer Firmware to the Board

Upload the ELF from the host over serial:

```bash
/Volumes/Media/executorch/.venv/bin/python \
  /Volumes/Media/executorch/serial_put_files.py \
  --port /dev/cu.usbserial-02BE3471 \
  /Users/sruthipolali/Documents/Playground/Executorch_runner_cm33/debug/executorch_runner_cm33.elf
```

Install it on the board:

```sh
cp -f /tmp/executorch_runner_cm33.elf /lib/firmware/executorch_runner_cm33.elf
chmod 644 /lib/firmware/executorch_runner_cm33.elf
md5sum /lib/firmware/executorch_runner_cm33.elf
```

## Put the Model on DDR with U-Boot `fatload`

The Arm tutorial's Linux `/dev/mem` model-loading path was blocked on this BSP
with:

```text
PermissionError: [Errno 1] Operation not permitted
```

Use U-Boot instead.

### Copy the `.pte` to the FAT Boot Partition

From Linux on the board:

```sh
cp /path/to/model.pte /run/media/boot-mmcblk0p1/
sync
ls -lh /run/media/boot-mmcblk0p1/model.pte
```

If transferring from the host, use whichever path is currently reliable:
serial upload, `scp`, or removable boot partition access. The important thing
is that the file exists on:

```text
/run/media/boot-mmcblk0p1
```

### Load the Model from U-Boot

Reboot, interrupt autoboot on the serial console, then:

```text
fatload mmc 0:1 0xc0000000 model.pte
boot
```

For example:

```text
fatload mmc 0:1 0xc0000000 titok_s128_encoder_source_matmul_residual_a16w8.pte
boot
```

Check the byte count printed by `fatload`. It must be less than the current
model window, and if using the input blob it should stay below `0x07800000`
bytes so it does not overlap `0xC7800000`.

## Optional Input Blob

The runner can read an input tensor blob from DDR before falling back to its
historical deterministic fill.

Create the blob on the host:

```bash
cd /Users/sruthipolali/Documents/Playground/sentinel_image_compression/titok-deploy-tools

/Users/sruthipolali/Documents/Playground/1d-tokenizer/.venv/bin/python \
  scripts/board/run_board.py make-cm33-input \
  --image outputs/better_inputs/0026_day_near_S2_H08_R3_IMAG0666.jpg \
  --output outputs/cm33_input/input_blob.bin \
  --metadata outputs/cm33_input/input_blob.json
```

Load it with U-Boot:

```text
fatload mmc 0:1 0xc7800000 input_blob.bin
```

Then load the model:

```text
fatload mmc 0:1 0xc0000000 model.pte
boot
```

Expected trace if the input blob was used:

```text
CM33: input blob copied addr=0xc7800000 bytes=...
```

## Start CM33 with Remoteproc

After Linux boots:

```sh
cd /sys/class/remoteproc/remoteproc0

cat state
echo stop > state 2>/dev/null || true
echo executorch_runner_cm33.elf > firmware
echo start > state
sleep 15
cat state
cat /sys/kernel/debug/remoteproc/remoteproc0/trace0
```

Healthy signals:

```text
NPU config match
NPU arch match
bus_status_error 0x0
cmd_end_reached 0x1
1 inferences finished
```

For TiTok, also look for:

```text
external_pte_header: present=1
external_pte_vela_scan: first_header=...
CM33: ethosu_fast_memory=0x20480000 size=393216
CM33: invoke status = 0
CM33_OUTPUT_HEX_BEGIN
CM33_OUTPUT_HEX_END
```

## Parse CM33 Output

The runner writes the first output tensor into a DDR output blob and also dumps
that blob as hex into `trace0`.

If you have a binary output blob locally:

```bash
/Users/sruthipolali/Documents/Playground/1d-tokenizer/.venv/bin/python \
  scripts/board/run_board.py parse-cm33-output \
  --blob output_blob.bin \
  --npz output_blob.npz \
  --metadata output_blob.json
```

For trace-only capture, extract the hex between:

```text
CM33_OUTPUT_HEX_BEGIN bytes=...
CM33_OUTPUT_HEX_END
```

Convert it to a binary blob, then parse with the command above.

## Troubleshooting Board Runs

If `remoteproc` rejects the ELF:

- inspect the ELF program headers,
- confirm the custom linker script was generated,
- confirm heap/stack are still set to `0x300`,
- rebuild the runner.

If `fatload` succeeds but the runner reports bad model data:

- compare the `fatload` byte count against the host `.pte` size,
- confirm the model was loaded to `0xC0000000`,
- confirm the `.pte` fits below the input blob area if using `0xC7800000`,
- check `external_pte_header` and Vela marker logs in `trace0`.

If the run reports a missing kernel/operator:

- rebuild the runner with `EXECUTORCH_SELECT_OPS_MODEL=/path/to/model.pte`,
- if that is too broad, add a narrower `EXECUTORCH_SELECT_OPS_LIST`.

If the NPU starts but stalls:

- compare Vela memory mode against runner memory layout,
- try `Dedicated_Sram_384KB` if the artifact needs base[2]/fast memory,
- inspect `POST_INVOKE STATUS`, `QREAD`, `CURRENT_OP`, and
  `CURRENT_CMD` logs,
- check for command stream errors, bus status errors, and whether
  `cmd_end_reached` becomes `1`.

If output shape is wrong:

- confirm the runner printed exactly one output tensor,
- compare board output metadata against host pre-lowering output metadata,
- check whether the output is float32 or int8 grid-recovered data.

## Running the Fallback Solution

The fallback solution is a separate student encoder, not the TiTok encoder
lowered to ExecuTorch. The historical checkpoint is archived at:

```text
outputs/_archive/2026-06-fallback-visual-evals/fallback_solution
```

Main scripts:

```text
scripts/ptq/export_fallback_tflite_pre_lowered.py
scripts/ptq/export_fallback_ai_edge_pre_lowered.py
scripts/ptq/evaluate_fallback_tflite_step_by_step.py
scripts/ptq/compare_fallback_vs_residual_add.py
```

### Export Fallback Through TFLite / OpenVINO2TF Path

```bash
cd /Users/sruthipolali/Documents/Playground/sentinel_image_compression/titok-deploy-tools

/Users/sruthipolali/Documents/Playground/1d-tokenizer/.venv/bin/python \
  scripts/ptq/export_fallback_tflite_pre_lowered.py \
  --titok-root /Users/sruthipolali/Documents/Playground/1d-tokenizer \
  --distill-repo-root /Users/sruthipolali/Documents/Playground/sentinel-titok-distill \
  --calibration-manifest /Volumes/Media/snapshot_serengeti_balanced_ptq_dataset/manifests/calibration_manifest.json \
  --calibration-count 500 \
  --output-dir outputs/fallback_tflite_pre_lowered_calib500
```

Output:

```text
outputs/fallback_tflite_pre_lowered_calib500/encoder_int8.tflite
outputs/fallback_tflite_pre_lowered_calib500/metadata.json
```

### Export Fallback Through AI Edge Torch

```bash
/Users/sruthipolali/Documents/Playground/1d-tokenizer/.venv/bin/python \
  scripts/ptq/export_fallback_ai_edge_pre_lowered.py \
  --distill-repo-root /Users/sruthipolali/Documents/Playground/sentinel-titok-distill \
  --calibration-manifest /Volumes/Media/snapshot_serengeti_balanced_ptq_dataset/manifests/calibration_manifest.json \
  --calibration-count 500 \
  --output-dir outputs/fallback_ai_edge_pre_lowered_calib500
```

Output:

```text
outputs/fallback_ai_edge_pre_lowered_calib500/encoder_int8.tflite
outputs/fallback_ai_edge_pre_lowered_calib500/metadata.json
```

### Teammate-Provided Already-Lowered Fallback Artifact

If using a teammate-provided already-lowered fallback artifact, treat the
export scripts above as provenance/debug tools, not the current deployment
source of truth.

The six-image and step-by-step comparison artifacts live here:

```text
outputs/2026-07-step-comparison/titok_vs_fallback_step_by_step_comparison
```

That directory contains per-image metrics, reconstructions, board logs, token
JSONs, and grids comparing TiTok and fallback paths.

## What Changed in TiTok Deploy Tools

The deploy tooling was extended to support multiple encoder/lowering flows:

- wrapper variants for `reshape_batch`, BMM attention, source-matmul
  attention, source-SDPA attention, and stock-MHA-style experiments,
- `TiTokTokenEncoderFromModules` composition so encoder variants can share the
  float VQ tokenizer,
- PTQ tooling that can select encoder variants,
- preference for a local `executorch-main` checkout when available,
- A16W8 Ethos-U quantization profile support,
- optional matmul quantization through the Arm composable quantizer API,
- post-partition q/dq `.out` rewrite support,
- graph/runtime summary JSON helpers,
- a narrow Cortex-M BMM rewrite for qualifying `dq -> bmm -> q` fallback
  islands.

Important local files:

```text
src/titok_deploy_tools/wrapper_tools/wrappers.py
src/titok_deploy_tools/ptq_tools/ptq.py
src/titok_deploy_tools/lowering_tools/ethosu_compat.py
src/titok_deploy_tools/lowering_tools/cortex_m_bmm_rewrite.py
src/titok_deploy_tools/lowering_tools/executorch_summary.py
src/titok_deploy_tools/lowering_tools/graph_summary.py
src/titok_deploy_tools/lowering_tools/post_partition_qdq_fix.py
scripts/export_and_lower/lower_ethosu_titok_s128_encoder.py
```

## What Changed in the CM33 Runner

Runner dependency:

```text
repository: https://github.com/spol25/Executorch_runner_cm33
branch: main
commit: 1842b9b28d014e8fcaf780f7fab193cc4dabe247
```

Key changes:

- normalized `SdkRootDirPath` so both `/Volumes/Media` and
  `/Volumes/Media/mcuxsdk` forms work,
- generated a custom linker script that marks `.bss`, `.heap`, and `.stack`
  as `NOLOAD`,
- reduced data RAM span to satisfy the UCM remoteproc loader,
- added explicit PHDRs so the ELF program headers are accepted,
- built runner source with `-Os` in debug to stay inside CM33 text limits,
- added selective portable-op generation from either an op list or `.pte`,
- generated current quantized q/dq bridge registrations from local
  ExecuTorch source,
- added local Cortex-M q/dq bridge kernels,
- linked optional portable op archives exactly once to avoid duplicate static
  kernel registration,
- set heap/stack to `0x300`,
- set scratch to 16 MB and method allocator to 60 MB in DDR,
- set `ET_MODEL_PTE_ADDR=0xC0000000`,
- expanded `trace0` buffer from 3 KB to 36 KB,
- moved `trace0` to `0x20015000`,
- retargeted runtime buffers to the UCM `0xC0000000` reserved DDR carveout,
- reserved 128 MB for externally loaded `.pte` data,
- added external PTE header inspection and Vela marker scans,
- added input blob loading from `0xC7800000`,
- added output blob dump to `0xCF000000` and hex dump through `trace0`,
- added detailed Ethos-U register/status/timing logs,
- initialized Ethos-U with 384 KB fast memory at `0x20480000`.

Important files:

```text
CMakeLists.txt
source/arm_executor_runner.cpp
source/rsc_table.h
patches/ethosu_log.h
source/cortex_m_qdq_ops.yaml
source/quantized_bridge_ops.yaml
```

## What Changed in ExecuTorch

The current CM33 flow has one direct dependency on the ExecuTorch fork:

```text
repository: https://github.com/spol25/executorch
branch: main
commit: dd873c3e8c (Support portable kernel-only CMake builds)
```

That commit changes only:

```text
kernels/portable/CMakeLists.txt
```

It adds `EXECUTORCH_PORTABLE_BUILD_KERNELS_ONLY`, allowing the external CM33
runner to compile portable kernel implementations while generating its own
selective operator registration from a `.pte` or explicit op list. It also
relaxes `-Werror` in that kernel-only mode so unrelated portable-kernel
warnings do not stop the embedded build.

Earlier Linux/Cortex-A, quantized-I/O, upstream runner-script, and portable BMM
experiments are preserved on the `imx93-linux-ethosu-experiments` branch. They
are not required by the current CM33 deployment path.

## Proven State

Known good:

- host can build the CM33 runner,
- UCM board exposes `/dev/ethosu0`,
- UCM board uses the expected Ethos-U DTB,
- UCM board exposes the expected `0xC0000000` / 256 MB Ethos-U reserved DDR
  region,
- Linux `/dev/mem` model load is blocked on this BSP,
- U-Boot `fatload` can place a `.pte` at `0xC0000000`,
- MobileNet ran successfully through CM33 remoteproc and Ethos-U65,
- the runner has been modified for TiTok-sized model windows and richer trace
  output.

Not yet guaranteed:

- a current TiTok `.pte` has run end-to-end on the CM33 runner,
- every selective helper-op set required by TiTok has been validated on board,
- the teammate-provided fallback artifact is fully documented as a board
  deployment path in this repo,
- the CM33 runner still uses the SDK's `mcimx93evk` board target because a
  UCM-specific MCUX board definition was not confirmed locally.

## Quick End-to-End Checklist

1. Lower or obtain the `.pte`.
2. Build the CM33 runner, using `EXECUTORCH_SELECT_OPS_MODEL` if needed.
3. Upload `executorch_runner_cm33.elf` to `/lib/firmware`.
4. Copy the `.pte` to `/run/media/boot-mmcblk0p1`.
5. Reboot and interrupt U-Boot.
6. Run `fatload mmc 0:1 0xc0000000 <model>.pte`.
7. Optionally `fatload` an input blob to `0xc7800000`.
8. Run `boot`.
9. Start remoteproc from Linux.
10. Inspect `trace0`.
11. Parse output if `CM33_OUTPUT_HEX_*` appears.
