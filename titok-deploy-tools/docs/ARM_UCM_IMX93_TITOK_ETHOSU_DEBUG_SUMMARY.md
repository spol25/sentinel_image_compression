# UCM-i.MX93 TiTok Ethos-U / CM33 Debug Summary

## Goal

Run a TiTok S128 encoder `.pte` on the UCM-i.MX93 CM33 firmware using the
i.MX93 Ethos-U65 NPU. The model is an A8W8, source-SDPA attention flow that
lowers to one `EthosUBackend` delegate plus q/dq boundary ops.

The board is CompuLab UCM-i.MX93. The Linux DTB used at boot is
`ucm-imx93-ethosu.dtb`.

## Build Inputs

Workspace:

```bash
/Users/sruthipolali/Documents/Playground
```

TiTok checkout:

```bash
/Users/sruthipolali/Documents/Playground/1d-tokenizer
```

Deploy tools:

```bash
/Users/sruthipolali/Documents/Playground/sentinel_image_compression/titok-deploy-tools
```

Calibration manifest:

```bash
/Volumes/Media/snapshot_serengeti_balanced_ptq_dataset/manifests/calibration_manifest.json
```

Python:

```bash
/Users/sruthipolali/Documents/Playground/1d-tokenizer/.venv/bin/python
```

## Compile Spec / Vela Flags

All flows used:

```text
--accelerator-config=ethos-u65-256
--config=Arm/vela.ini
--output-format=raw
--debug-force-regor
--system-config=Ethos_U65_High_End
```

Verbose/debug flags added:

```text
--enable-debug-db
--verbose-config
--verbose-allocation
--verbose-high-level-command-stream
--verbose-register-command-stream
--verbose-operators
```

Memory modes tested:

```text
Shared_Sram
Dedicated_Sram
Dedicated_Sram_384KB
```

From the installed Vela `Arm/vela.ini`:

```ini
[System_Config.Ethos_U65_High_End]
axi0_port=Sram
axi1_port=Dram

[Memory_Mode.Shared_Sram]
const_mem_area=Axi1
arena_mem_area=Axi0
cache_mem_area=Axi0

[Memory_Mode.Dedicated_Sram]
const_mem_area=Axi1
arena_mem_area=Axi1
cache_mem_area=Axi0

[Memory_Mode.Dedicated_Sram_384KB]
inherit=Memory_Mode.Dedicated_Sram
arena_cache_size=393216
```

Question for Arm/NXP: is `Ethos_U65_High_End` with one of these Arm default
memory modes correct for UCM-i.MX93 Linux+BSP, or does this BSP require a
custom `system_config`/`memory_mode` mapping for the Ethos-U AXI ports?

## Commands Used To Build

Shared SRAM:

```bash
/Users/sruthipolali/Documents/Playground/1d-tokenizer/.venv/bin/python \
  sentinel_image_compression/titok-deploy-tools/scripts/export_and_lower/trace_titok_s128_sdpa_attention_boundaries.py \
  --titok-root /Users/sruthipolali/Documents/Playground/1d-tokenizer \
  --manifest /Volumes/Media/snapshot_serengeti_balanced_ptq_dataset/manifests/calibration_manifest.json \
  --calibration-count 4 \
  --output-dir outputs/rewrite_a8w8_balanced_pipeline/sdpa_attention_ucm_vela_shared_debug \
  --summary-name sdpa_attention_ucm_vela_shared_debug_summary.json \
  --quantization-profile int8 \
  --per-channel \
  --ethos-memory-mode Shared_Sram \
  --dump-vela-intermediates \
  --ethos-extra-flag=--enable-debug-db \
  --ethos-extra-flag=--verbose-config \
  --ethos-extra-flag=--verbose-allocation \
  --ethos-extra-flag=--verbose-high-level-command-stream \
  --ethos-extra-flag=--verbose-register-command-stream \
  --ethos-extra-flag=--verbose-operators \
  > sentinel_image_compression/titok-deploy-tools/outputs/rewrite_a8w8_balanced_pipeline/sdpa_attention_ucm_vela_shared_debug.log 2>&1
```

Dedicated SRAM:

```bash
/Users/sruthipolali/Documents/Playground/1d-tokenizer/.venv/bin/python \
  sentinel_image_compression/titok-deploy-tools/scripts/export_and_lower/trace_titok_s128_sdpa_attention_boundaries.py \
  --titok-root /Users/sruthipolali/Documents/Playground/1d-tokenizer \
  --manifest /Volumes/Media/snapshot_serengeti_balanced_ptq_dataset/manifests/calibration_manifest.json \
  --calibration-count 4 \
  --output-dir outputs/rewrite_a8w8_balanced_pipeline/sdpa_attention_ucm_vela_dedicated_debug \
  --summary-name sdpa_attention_ucm_vela_dedicated_debug_summary.json \
  --quantization-profile int8 \
  --per-channel \
  --ethos-memory-mode Dedicated_Sram \
  --dump-vela-intermediates \
  --ethos-extra-flag=--enable-debug-db \
  --ethos-extra-flag=--verbose-config \
  --ethos-extra-flag=--verbose-allocation \
  --ethos-extra-flag=--verbose-high-level-command-stream \
  --ethos-extra-flag=--verbose-register-command-stream \
  --ethos-extra-flag=--verbose-operators \
  > sentinel_image_compression/titok-deploy-tools/outputs/rewrite_a8w8_balanced_pipeline/sdpa_attention_ucm_vela_dedicated_debug.log 2>&1
```

Dedicated SRAM 384KB:

```bash
/Users/sruthipolali/Documents/Playground/1d-tokenizer/.venv/bin/python \
  sentinel_image_compression/titok-deploy-tools/scripts/export_and_lower/trace_titok_s128_sdpa_attention_boundaries.py \
  --titok-root /Users/sruthipolali/Documents/Playground/1d-tokenizer \
  --manifest /Volumes/Media/snapshot_serengeti_balanced_ptq_dataset/manifests/calibration_manifest.json \
  --calibration-count 4 \
  --output-dir outputs/rewrite_a8w8_balanced_pipeline/sdpa_attention_ucm_vela_dedicated_384kb_debug \
  --summary-name sdpa_attention_ucm_vela_dedicated_384kb_debug_summary.json \
  --quantization-profile int8 \
  --per-channel \
  --ethos-memory-mode Dedicated_Sram_384KB \
  --dump-vela-intermediates \
  --ethos-extra-flag=--enable-debug-db \
  --ethos-extra-flag=--verbose-config \
  --ethos-extra-flag=--verbose-allocation \
  --ethos-extra-flag=--verbose-high-level-command-stream \
  --ethos-extra-flag=--verbose-register-command-stream \
  --ethos-extra-flag=--verbose-operators \
  > sentinel_image_compression/titok-deploy-tools/outputs/rewrite_a8w8_balanced_pipeline/sdpa_attention_ucm_vela_dedicated_384kb_debug.log 2>&1
```

## Build Outputs

Shared SRAM source-SDPA PTE:

```text
Path: /Users/sruthipolali/Documents/Playground/sentinel_image_compression/titok-deploy-tools/outputs/rewrite_a8w8_balanced_pipeline/sdpa_attention_ucm_vela_shared_debug/source_sdpa_attention/source_sdpa_attention.pte
Size: 30032240 bytes
SHA256: d8ec20c33066314cbc7dcee32c7d0ae910d27e46cbe02ffe53bd05b9fed99d0b
```

Dedicated SRAM source-SDPA PTE:

```text
Path: /Users/sruthipolali/Documents/Playground/sentinel_image_compression/titok-deploy-tools/outputs/rewrite_a8w8_balanced_pipeline/sdpa_attention_ucm_vela_dedicated_debug/source_sdpa_attention/source_sdpa_attention.pte
Size: 29568992 bytes
SHA256: 491289fdedb14fd4d190c9025d3e27c39d91bc822ac98f8548c9eccfaa640111
```

Dedicated SRAM 384KB source-SDPA PTE:

```text
Path: /Users/sruthipolali/Documents/Playground/sentinel_image_compression/titok-deploy-tools/outputs/rewrite_a8w8_balanced_pipeline/sdpa_attention_ucm_vela_dedicated_384kb_debug/source_sdpa_attention/source_sdpa_attention.pte
Size: 30545360 bytes
SHA256: ae934be84eccab8a9feee3254aed4d35c054407b1294913777b66ef8017dbee5
```

## Vela Summary: Shared_Sram

Verbose log:

```text
/Users/sruthipolali/Documents/Playground/sentinel_image_compression/titok-deploy-tools/outputs/rewrite_a8w8_balanced_pipeline/sdpa_attention_ucm_vela_shared_debug.log
```

Artifacts:

```text
/Users/sruthipolali/Documents/Playground/sentinel_image_compression/titok-deploy-tools/outputs/rewrite_a8w8_balanced_pipeline/sdpa_attention_ucm_vela_shared_debug/source_sdpa_attention/vela_intermediates
```

Important Vela output:

```text
System Configuration (Ethos_U65_High_End)
Memory Mode (Shared_Sram)
Accelerator configuration: Ethos_U65_256
Memory mode: Shared_Sram
Design peak SRAM bandwidth: 14.90 GB/s
Design peak DRAM bandwidth: 3.49 GB/s
Total SRAM used: 10864.14 KiB
Total DRAM used: 23932.72 KiB
CPU operators = 0 (0.0%)
NPU operators = 1244 (100.0%)
Total SRAM bandwidth per input: 2772.04 MB/inference
Total DRAM bandwidth per input: 144.61 MB/inference
Neural network MACs: 11038833280 MACs/batch
```

NPZ regions:

```text
cmd_data: 5522228 bytes
weight_data: 24507104 bytes
scratch_shape: 11124880 bytes
input_region/input_offset: [1] / [956480]
output_region/output_offset: [1] / [0]
```

Board behavior from the earlier Shared_Sram A8W8 SDPA run:

```text
NPU config match
NPU arch match
handle_command_stream: cmd_stream=0xc00008e0, cms_length 1380548
base[0]=0xc0544c10 size=24507104
base[1]=0xc8000000 size=11124880
base[2]=0x00000000 size=0
```

The command stream started and advanced through several interrupts, then
stalled/hung. It did not reach:

```text
cmd_end_reached 0x1
1 inferences finished
```

## Vela Summary: Dedicated_Sram

Verbose log:

```text
/Users/sruthipolali/Documents/Playground/sentinel_image_compression/titok-deploy-tools/outputs/rewrite_a8w8_balanced_pipeline/sdpa_attention_ucm_vela_dedicated_debug.log
```

Artifacts:

```text
/Users/sruthipolali/Documents/Playground/sentinel_image_compression/titok-deploy-tools/outputs/rewrite_a8w8_balanced_pipeline/sdpa_attention_ucm_vela_dedicated_debug/source_sdpa_attention/vela_intermediates
```

Important Vela output:

```text
System Configuration (Ethos_U65_High_End)
Memory Mode (Dedicated_Sram)
Accelerator configuration: Ethos_U65_256
Memory mode: Dedicated_Sram
Total SRAM used: 10864.14 KiB
Total DRAM used: 28105.97 KiB
CPU operators = 0 (0.0%)
NPU operators = 1244 (100.0%)
Total SRAM bandwidth per input: 454.93 MB/inference
Total DRAM bandwidth per input: 2461.40 MB/inference
Neural network MACs: 11038833280 MACs/batch
```

NPZ regions:

```text
cmd_data: 5541092 bytes
weight_data: 24024992 bytes
scratch_shape: 4755520 bytes
input_region/input_offset: [1] / [0]
output_region/output_offset: [1] / [0]
```

This plain `Dedicated_Sram` PTE was compiled and inspected but was not the one
run on the board. The board run below used `Dedicated_Sram_384KB`, because that
is closer to the 384KB dedicated SRAM/cache assumption in the generic Arm Vela
config.

## Vela Summary: Dedicated_Sram_384KB

Verbose log:

```text
/Users/sruthipolali/Documents/Playground/sentinel_image_compression/titok-deploy-tools/outputs/rewrite_a8w8_balanced_pipeline/sdpa_attention_ucm_vela_dedicated_384kb_debug.log
```

Artifacts:

```text
/Users/sruthipolali/Documents/Playground/sentinel_image_compression/titok-deploy-tools/outputs/rewrite_a8w8_balanced_pipeline/sdpa_attention_ucm_vela_dedicated_384kb_debug/source_sdpa_attention/vela_intermediates
```

Important Vela output:

```text
System Configuration (Ethos_U65_High_End)
Memory Mode (Dedicated_Sram_384KB)
Accelerator configuration: Ethos_U65_256
Memory mode: Dedicated_Sram_384KB
arena_cache_size: 384 KiB
Total SRAM used: 384.0 KiB
Total DRAM used: 30448.48 KiB
CPU operators = 0 (0.0%)
NPU operators = 1244 (100.0%)
Total DRAM bandwidth per input: 4719.43 MB/inference
Neural network MACs: 11038833280 MACs/batch
```

NPZ regions:

```text
cmd_data: 6521056 bytes
weight_data: 24021408 bytes
scratch_shape: 7157840 bytes
input_region/input_offset: [1] / [956480]
output_region/output_offset: [1] / [0]
```

## Commands Used On Board

Copy PTE to FAT partition:

```bash
scp -o BindAddress=169.254.24.39 \
  -o StrictHostKeyChecking=no \
  -i /Users/sruthipolali/Documents/Playground/.ucm-imx93/id_ed25519 \
  /Users/sruthipolali/Documents/Playground/sentinel_image_compression/titok-deploy-tools/outputs/rewrite_a8w8_balanced_pipeline/sdpa_attention_ucm_vela_dedicated_384kb_debug/source_sdpa_attention/source_sdpa_attention.pte \
  root@169.254.15.28:/run/media/boot-mmcblk0p1/titok_ded384.pte
```

Temporary U-Boot command wrapper from Linux, because interrupting autoboot over
this serial session was unreliable:

```bash
fw_setenv bootcmd 'fatload mmc 0:1 0xc0000000 titok_ded384.pte; run bsp_bootcmd; run distro_bootcmd'
sync
reboot
```

U-Boot output:

```text
30545360 bytes read in 662 ms (44 MiB/s)
Running BSP bootcmd ...
```

Restore U-Boot environment after Linux boot:

```bash
fw_setenv bootcmd 'run bsp_bootcmd; run distro_bootcmd'
fw_setenv bootdelay 2
fw_printenv bootcmd bootdelay
```

Start CM33 firmware:

```bash
echo executorch_runner_cm33.elf > /sys/class/remoteproc/remoteproc0/firmware
echo start > /sys/class/remoteproc/remoteproc0/state
cat /sys/kernel/debug/remoteproc/remoteproc0/trace0
```

Stop CM33 firmware:

```bash
echo stop > /sys/class/remoteproc/remoteproc0/state
```

## Board Runtime Log: Dedicated_Sram_384KB

The CM33 trace was in:

```text
/sys/kernel/debug/remoteproc/remoteproc0/trace0
```

Key log:

```text
I [arm_executor_runner.cpp:1809] Starting running 1 inferences...
CM33: execute start inference=0 irq_count=0 manual_irq=0 timeout_count=0
I: Optimizer config. product=1, cmd_stream_version=0, macs_per_cc=8, shram_size=48, custom_dma=0
I: Optimizer config. arch version: 1.0.6
I: Ethos-U config. product=1, cmd_stream_version=0, macs_per_cc=8, shram_size=48, custom_dma=0
I: Ethos-U. arch version=1.0.6
I: Test Case 16: handle_optimizer_config: NPU config match
I: Test Case 17: handle_optimizer_config: NPU arch match
I: handle_command_stream: cmd_stream=0xc0000a60, cms_length 1630256
CM33: inference_begin base_addrs=3 fast_mem=0x00000000 fast_mem_size=0
CM33:   base[0]=0xc0638b40 size=24021408
CM33:   base[1]=0xc8000000 size=7157840
CM33:   base[2]=0x00000000 size=0
CM33: Ethos-U IRQ #1 enter
CM33: irq_enter: STATUS=0x00008006 CMD=0x00000001 QREAD=7348 CURRENT_QREAD=7580 CURRENT_OP=0x00000003 CURRENT_CMD=0x00010005 DEBUG_MISC=0x00000000
CM33: irq_enter: state=0 irq=1 bus=1 reset=0 parse=0 cmd_end=0 wd=0 ecc=0 fault_if=0 fault_ch=8 irq_hist=0x0000
I: Test Case 10: bus_status_error 0x1, status 0x8004
I: Test Case 10: faulting_inference 0x0
I: Test Case 10: faulting_channel 0x8
I: Test Case 11: cmd_parse_error 0x0
I: Test Case 12: wd_fault 0x0
I: Test Case 13: ecc_fault: 0x0
I: Test Case 9: dma read/write to external memory
I: Test Case 14: cmd_end_reached 0x0
I: Test Case 14: get read offset of command stream 7348
CM33: inference_end irq_count=1 manual_irq_count=0 timeout_count=0
CM33: inference_end: STATUS=0x00008004 CMD=0x00000002 QREAD=7348 CURRENT_QREAD=7580 CURRENT_OP=0x00000003 CURRENT_CMD=0x00010005 DEBUG_MISC=0x00000000
CM33: inference_end: state=0 irq=0 bus=1 reset=0 parse=0 cmd_end=0 wd=0 ecc=0 fault_if=0 fault_ch=8 irq_hist=0x0000
```

Interpretation: `Dedicated_Sram_384KB` does not hang; it fails early with an
Ethos-U external memory bus status error. The NPU config and arch match, so the
accelerator target appears to match, but some memory address/AXI region/channel
assumption still appears wrong.

## 4MB Guide Question

The guide says:

```dts
reg = <0 0xc0000000 0 0x400000>; /* 4MB for .pte model */
```

That is not the active UCM setup used here.

The live UCM boot log shows:

```text
OF: reserved mem: initialized node ethosu_region@C0000000, compatible id shared-dma-pool
OF: reserved mem: 0x00000000c0000000..0x00000000cfffffff (262144 KiB) map reusable ethosu_region@C0000000
```

The CM33 runner was changed for UCM to use the 256MB region:

```cpp
kUcmEthosuRegionBase = 0xC0000000
kUcmEthosuRegionSize = 0x10000000 /* 256MB */
kModelReservedSize = 0x08000000   /* 128MB */
kUcmScratchBase = 0xC0000000 + 128MB = 0xC8000000
```

So the workaround for PTEs larger than 4MB is:

1. Load the `.pte` at `0xC0000000`.
2. Reserve the first `128MB` of the 256MB Ethos-U region for the model payload.
3. Put scratch/method/planned buffers after that, starting at `0xC8000000`.

The tested PTEs are around 29-31MB, so they fit within the 128MB model window.
The runtime buffer bases printed by CM33 also show no simple overlap:

```text
Dedicated_Sram_384KB:
base[0]=0xc0638b40 size=24021408
base[1]=0xc8000000 size=7157840
```

## Direct Answers To Arm Checklist

### 1. system_config and memory_config correctness

Current compile spec:

```text
target / accelerator config: ethos-u65-256
system_config: Ethos_U65_High_End
memory_config tried: Shared_Sram, Dedicated_Sram, Dedicated_Sram_384KB
```

Evidence that accelerator target matches:

```text
NPU config match
NPU arch match
product=1, cmd_stream_version=0, macs_per_cc=8, shram_size=48
arch version=1.0.6
```

Open question: whether `Ethos_U65_High_End` and the default Arm memory modes
match the UCM-i.MX93 BSP's actual Ethos-U AXI port/memory-region mapping.

### 2. Enough memory available

At the high level, capacity seems sufficient:

```text
Linux reserved ethosu_region: 256MB at 0xC0000000
CM33 model window: 128MB at 0xC0000000
PTE sizes: about 29-31MB
Scratch sizes from Vela NPZ: 4.8MB to 11.1MB
Scratch base on CM33: 0xC8000000
```

Vela reports:

```text
Shared_Sram: Total SRAM 10864.14 KiB, Total DRAM 23932.72 KiB
Dedicated_Sram: Total SRAM 10864.14 KiB, Total DRAM 28105.97 KiB
Dedicated_Sram_384KB: Total SRAM 384.0 KiB, Total DRAM 30448.48 KiB
```

However, the `Dedicated_Sram_384KB` board run still gets:

```text
bus_status_error 0x1, status 0x8004, faulting_channel 0x8
```

So this does not look like a simple "PTE too large for region" problem. It
looks more like memory-region/AXI-port/addressability/coherency configuration.

### 3. Verbose Vela logs

Verbose logs were generated with:

```text
--enable-debug-db
--verbose-config
--verbose-allocation
--verbose-high-level-command-stream
--verbose-register-command-stream
--verbose-operators
```

Useful artifacts:

```text
out.tosa
output/out_vela.npz
output/out_summary_Ethos_U65_High_End.csv
output/out_debug.xml
full stdout/stderr .log with register command stream dump
```

Paths:

```text
Shared:
/Users/sruthipolali/Documents/Playground/sentinel_image_compression/titok-deploy-tools/outputs/rewrite_a8w8_balanced_pipeline/sdpa_attention_ucm_vela_shared_debug

Dedicated:
/Users/sruthipolali/Documents/Playground/sentinel_image_compression/titok-deploy-tools/outputs/rewrite_a8w8_balanced_pipeline/sdpa_attention_ucm_vela_dedicated_debug

Dedicated 384KB:
/Users/sruthipolali/Documents/Playground/sentinel_image_compression/titok-deploy-tools/outputs/rewrite_a8w8_balanced_pipeline/sdpa_attention_ucm_vela_dedicated_384kb_debug
```

## Current Question For Arm/NXP

Given:

```text
accelerator config = ethos-u65-256
NPU config/arch match at runtime
PTE fits in the 256MB UCM ethosu_region
Dedicated_Sram_384KB fails with bus_status_error on faulting_channel 0x8
Shared_Sram starts but appears to stall/hang before cmd_end_reached
```

What `system_config` and `memory_mode` should Vela use for UCM-i.MX93, and how
should Vela memory areas map to the physical memory passed by the CM33 runner
(`0xC0000000` model/weights and `0xC8000000` scratch)?

## Follow-Up Finding: Dedicated SRAM Fast Memory Was Not Wired

After reviewing the CM33 runner, we found a concrete runtime mismatch for the
`Dedicated_Sram_384KB` build:

```cpp
ethosu_init(&ethosu_drv, (void*)0x4A900000, NULL, 0, 0, 0)
```

This explains the previous runtime trace:

```text
fast_mem=0x00000000 fast_mem_size=0
base[2]=0x00000000 size=0
```

For `Dedicated_Sram_384KB`, Vela's config uses:

```text
const_mem_area=Axi1
arena_mem_area=Axi1
cache_mem_area=Axi0
arena_cache_size=393216
```

So the runtime must provide the Axi0 cache/fast-memory region to the Ethos-U
core driver. The runner has now been patched locally to pass:

```text
fast_memory=0x204C0000
fast_memory_size=393216
```

into `ethosu_init`. This address is intended to represent the rear 384KB OCRAM
window commonly described for i.MX93 Ethos-U cache usage, but it should still be
confirmed against the exact UCM-i.MX93 BSP device tree/memory map before treating
it as final.

The patched firmware rebuilds successfully. The next board test should confirm
that the trace now shows nonzero fast memory and that `base[2]` is rewritten by
the Ethos-U driver to the OCRAM address before running the command stream.

## Follow-Up Board Test: Fast Memory Nonzero, Fault Persists

The rebuilt firmware was copied to the board:

```text
Local ELF SHA256:
3ea36ede33fe7b07eac3a5675cda7b57f3699061af71e2aaa316f66b638c20a5

Board ELF SHA256:
3ea36ede33fe7b07eac3a5675cda7b57f3699061af71e2aaa316f66b638c20a5
```

The board PTE matched the local `Dedicated_Sram_384KB` PTE:

```text
ae934be84eccab8a9feee3254aed4d35c054407b1294913777b66ef8017dbee5
```

The board was rebooted with:

```text
bootcmd=fatload mmc 0:1 0xc0000000 titok_ded384.pte; run bsp_bootcmd; run distro_bootcmd
```

After starting `remoteproc0`, the trace confirmed that fast memory was now
nonzero and that the core driver rewrote `base[2]`:

```text
CM33: inference_begin base_addrs=3 fast_mem=0x204c0000 fast_mem_size=393216
CM33:   base[0]=0xc0638b40 size=24021408
CM33:   base[1]=0xc8000000 size=7157840
CM33:   base[2]=0x204c0000 size=0
```

The failure still occurs at the same command stream point:

```text
CM33: irq_enter: STATUS=0x00008006 CMD=0x00000001 QREAD=7348 CURRENT_QREAD=7580 CURRENT_OP=0x00000003 CURRENT_CMD=0x00010005 DEBUG_MISC=0x00000000
CM33: irq_enter: state=0 irq=1 bus=1 reset=0 parse=0 cmd_end=0 wd=0 ecc=0 fault_if=0 fault_ch=8 irq_hist=0x0000
I: Test Case 10: bus_status_error 0x1, status 0x8004
I: Test Case 10: faulting_inference 0x0
I: Test Case 10: faulting_channel 0x8
I: Test Case 14: cmd_end_reached 0x0
I: Test Case 14: get read offset of command stream 7348
```

Interpretation: the original zero-fast-memory bug is fixed, but the selected
fast-memory address may still be wrong for UCM-i.MX93, may not be reserved or
NPU-accessible in this BSP, or the Vela memory configuration may still not match
the SoC AXI mapping.

`/proc/iomem` on the live board does not show an obvious OCRAM/SRAM reservation
around `0x204c0000`; it does show:

```text
c0000000-cfffffff : reserved
```

for the 256MB Ethos-U DDR region, but no matching `0x204c0000-0x2051ffff`
reserved-memory line. That makes the exact OCRAM/SRAM address a key remaining
question for UCM/NXP/Arm.

After the test, the board was restored to:

```text
bootcmd=run bsp_bootcmd; run distro_bootcmd
remoteproc0 state=offline
```

## Follow-Up Finding: Live UCM DTB Does Not Define OCRAM Fast Memory

After SSH was restored, the live UCM-i.MX93 device tree and `/proc/iomem` were
checked for the correct fast-memory/OCRAM window.

The live Ethos-U node is:

```text
/sys/firmware/devicetree/base/ethosu
compatible = "arm,ethosu"
status = "okay"
memory-region = <&ethosu_mem>
```

The `ethosu_mem` symbol resolves only to:

```text
/reserved-memory/ethosu_region@C0000000
```

The reserved Ethos-U memory region is:

```text
reg = <0x0 0xC0000000 0x0 0x10000000>
```

That is the 256MB DDR/CMA region:

```text
c0000000-cfffffff : reserved
```

No live device-tree node or `/proc/iomem` range was found for `sram`, `ocram`,
`0x20480000`, `0x204c0000`, or `0x20500000`. The copied boot DTB
`ucm-imx93-ethosu.dtb` also contains the `0xC0000000/0x10000000` Ethos-U DDR
reservation, but does not contain a `0x204c0000/0x60000` reservation.

Conclusion: `0x204C0000` is a plausible rear-384KB i.MX93 OCRAM address based
on the generic i.MX93 memory description, but it is **not confirmed or exposed
by this UCM BSP's live device tree**. For this board image, the only confirmed
Ethos-U memory window from Linux DT is the 256MB DDR region at `0xC0000000`.
The correct NPU-visible OCRAM/cache address still needs confirmation from the
UCM/NXP BSP memory map or by adding an explicit reserved SRAM/OCRAM node that
the firmware and Vela memory mode agree on.
