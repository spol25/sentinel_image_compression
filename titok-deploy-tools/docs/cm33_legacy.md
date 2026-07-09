# CM33 Log

> **Historical bring-up log.** This file preserves the investigation sequence
> and includes paths, repository state, and next steps that were accurate at
> the time they were recorded. It is not the current deployment guide. Use
> [`RUNBOOK.md`](RUNBOOK.md) and the
> [`spol25/Executorch_runner_cm33` README](https://github.com/spol25/Executorch_runner_cm33)
> for current instructions. References below to the deleted
> `UCM_IMX93_RUNBOOK.md` are retained only as historical context.

## goal

Bring up the ExecuTorch CM33 runner on the CompuLab UCM-i.MX93 board with
Ethos-U65 acceleration, using the Arm/NXP tutorial flow as a starting point
while adapting it for the UCM board instead of the FRDM/EVK board.

The practical goal is to make the board run `.pte` models through the CM33
runner and Ethos-U65 NPU, with a repeatable workflow for:

- rebuilding the CM33 firmware on macOS
- transferring the firmware to the board
- loading a `.pte` model into the UCM reserved DDR region
- starting the CM33 firmware through Linux `remoteproc`
- checking `trace0` for NPU execution success
- extending the setup from MobileNet-sized models to larger TiTok models

The original tutorial targets FRDM-MIMX93 / EVK assumptions. The actual board
here is UCM-i.MX93, so the key work has been separating tutorial steps that
carry over directly from board-specific details that need adaptation.

## state of the system right now

Host machine:

- Workspace root: `/Users/sruthipolali/Documents/Playground`
- Runner repo: `/Users/sruthipolali/Documents/Playground/Executorch_runner_cm33`
- Current log file:
  `/Users/sruthipolali/Documents/Playground/sentinel_image_compression/titok-deploy-tools/docs/cm33_legacy.md`
- UCM runbook: `/Users/sruthipolali/Documents/Playground/Executorch_runner_cm33/UCM_IMX93_RUNBOOK.md`

Toolchain and SDK:

- Arm GNU toolchain:
  `/Users/sruthipolali/.mcuxpressotools/arm-gnu-toolchain-14.2.rel1-darwin-arm64-arm-none-eabi`
- MCUX Python venv:
  `/Users/sruthipolali/.mcuxpressotools/.mcux-venv-3.12/bin`
- SDK root used by the build:
  `/Volumes/Media`
- Actual MCUX SDK tree:
  `/Volumes/Media/mcuxsdk/mcuxsdk`
- The runner is still built with the available SDK board target
  `mcimx93evk`; there is no confirmed UCM-specific MCUX board definition in
  the local SDK.

Board connection and board-side facts:

- Serial console: `/dev/cu.usbserial-02BE3471`
- Known serial command:
  `picocom -b 115200 /dev/cu.usbserial-02BE3471`
- Last verified active DTB:
  `ucm-imx93-ethosu.dtb`
- Last verified reserved memory:
  `ethosu_region@0xC0000000`, size `0x10000000` / `256 MB`
- Last verified Ethos-U device:
  `/dev/ethosu0`
- First SD / FAT boot partition in Linux:
  `/run/media/boot-mmcblk0p1`

Runner firmware state:

- Firmware output path:
  `/Users/sruthipolali/Documents/Playground/Executorch_runner_cm33/debug/executorch_runner_cm33.elf`
- Last known successful MobileNet run used this CM33 remoteproc flow:
  load model at `0xC0000000`, start `executorch_runner_cm33.elf`, inspect
  `/sys/kernel/debug/remoteproc/remoteproc0/trace0`.
- MobileNet success signals previously seen in `trace0`:
  `NPU config match`, `NPU arch match`, `bus_status_error 0x0`,
  `cmd_end_reached 0x1`, and `1 inferences finished`.

Current UCM memory layout in the runner code:

- Source file:
  `/Users/sruthipolali/Documents/Playground/Executorch_runner_cm33/source/arm_executor_runner.cpp`
- `kUcmEthosuRegionBase = 0xC0000000`
- `kUcmEthosuRegionSize = 0x10000000` / `256 MB`
- `kModelReservedSize = 0x08000000` / `128 MB`
- Scratch pool begins after the 128 MB model window.
- Method allocator begins after scratch.
- Planned DDR buffers begin after the method allocator.
- The runner now includes external PTE inspection/debug logging helpers.

Current compile-time pool sizes in `CMakeLists.txt`:

- File:
  `/Users/sruthipolali/Documents/Playground/Executorch_runner_cm33/CMakeLists.txt`
- `ET_MODEL_PTE_ADDR=0xC0000000`
- `ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE=0x1000000`
  / `16 MB`
- `ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE=0x03C00000`
  / `60 MB`
- Linker heap size override: `0x300`
- Linker stack size override: `0x300`

Important note about the current runbook:

- The selected paragraph in `UCM_IMX93_RUNBOOK.md` says the runner reserves
  only `4 MB` for the model payload.
- That paragraph is stale relative to the current source.
- The current source now reserves `128 MB`, which is intended to fit TiTok
  `.pte` files around `89-91 MB`.
- The runbook should be updated before using it as the final source of truth
  for TiTok.

SDK patch state:

- The SDK patch script was applied.
- Patched linker script:
  `/Volumes/Media/mcuxsdk/mcuxsdk/devices/i.MX/i.MX93/MIMX9352/gcc/MIMX9352xxxxM_ram.ld`
- Patched Ethos-U log header:
  `/Volumes/Media/mcuxsdk/mcuxsdk/middleware/eiq/ethos-u-core-software/core_driver/src/ethosu_log.h`
- The SDK patches were generic runner fixes, not UCM-specific board-support
  patches.
- The linker patch adds GOT sections to `.data` so startup copies them to RAM.
- The Ethos-U log patch routes useful driver/NPU logs into `trace0`.

Current repo status:

- Modified:
  `CMakeLists.txt`
- Modified:
  `source/arm_executor_runner.cpp`
- Untracked:
  `UCM_IMX93_RUNBOOK.md`
- Untracked:
  `source/cortex_m_qdq_ops.yaml`
- Untracked:
  `source/quantized_bridge_ops.yaml`
- This file, `cm33_legacy.md`, now lives in the deploy-tools docs folder so it is
  visible next to the canonical runbook.

Current TiTok artifacts found locally:

- Shared SRAM TiTok PTE:
  `/Users/sruthipolali/Documents/Playground/sentinel_image_compression/titok-deploy-tools/outputs/rewrite_a16w8_balanced_pipeline/lowering_shared_sram/titok_s128_encoder_ethosu_u65_a16w8_shared_sram.pte`
- Size: about `89 MB`
- Dedicated SRAM TiTok PTE:
  `/Users/sruthipolali/Documents/Playground/sentinel_image_compression/titok-deploy-tools/outputs/rewrite_a16w8_balanced_pipeline/lowering_dedicated_sram/titok_s128_encoder_ethosu_u65_a16w8_dedicated_sram.pte`
- Size: about `89 MB`
- Earlier/default TiTok PTE:
  `/Users/sruthipolali/Documents/Playground/sentinel_image_compression/titok-deploy-tools/outputs/rewrite_a16w8_balanced_pipeline/lowering_a16w8_balanced/titok_s128_encoder_ethosu_u65_a16w8.pte`
- Size: about `90 MB`

Sentinel image compression project state:

- Project root:
  `/Users/sruthipolali/Documents/Playground/sentinel_image_compression`
- The TiTok deploy tooling under `titok-deploy-tools/` has uncommitted source
  changes and many generated experiment outputs.
- Modified tracked files:
  `titok-deploy-tools/src/titok_deploy_tools/wrappers.py`,
  `titok-deploy-tools/src/titok_deploy_tools/ptq.py`,
  `titok-deploy-tools/src/titok_deploy_tools/ethosu_compat.py`,
  `titok-deploy-tools/scripts/export_and_lower/lower_ethosu_titok_s128_encoder.py`,
  `titok-deploy-tools/scripts/ptq/run_encoder_ptq_experiment.py`, and
  `titok-deploy-tools/scripts/ptq/run_s128_calibration_baseline.py`.
- New source helpers:
  `titok-deploy-tools/src/titok_deploy_tools/cortex_m_bmm_rewrite.py`,
  `titok-deploy-tools/src/titok_deploy_tools/executorch_summary.py`,
  `titok-deploy-tools/src/titok_deploy_tools/graph_summary.py`, and
  `titok-deploy-tools/src/titok_deploy_tools/post_partition_qdq_fix.py`.
- New lowering/diagnostic scripts include shared-SRAM, no-unsqueeze, BMM
  attention, source-matmul, SDPA, stock-MHA, and boundary-tracing variants
  under `titok-deploy-tools/scripts/export_and_lower/`.
- Large generated output directories exist:
  `outputs/rewrite_a16w8_balanced_pipeline` is about `1.0 GB`,
  `outputs/nxp_arm_handoff_bundle` is about `352 MB`, and
  `outputs/rewrite_int8_pipeline` is about `49 MB`.

Key Sentinel/TiTok source changes:

- Added several encoder wrapper variants in
  `titok-deploy-tools/src/titok_deploy_tools/wrappers.py`:
  `TiTokEncoderOnlyReshapeBatch`, `TiTokEncoderOnlyBmmAttention`,
  `TiTokEncoderOnlySourceMatmulAttention`,
  `TiTokEncoderOnlyStockMhaAttention`, and
  `TiTokEncoderOnlySourceSdpaAttention`.
- Added a BMM-based attention implementation to express attention as rank-3
  `torch.bmm` operations instead of the original higher-rank attention path.
- Added `TiTokTokenEncoderFromModules` so arbitrary encoder variants can be
  composed with the float VQ tokenizer.
- Extended PTQ tooling in `ptq.py` to select encoder variants, prefer a local
  `executorch-main` checkout, support A16W8 Ethos-U quantization flows, and
  optionally request matmul quantization through the newer composable Arm
  quantizer API.
- Changed `EthosUCompatCompileSpec` default memory mode from `Sram_Only` to
  `Dedicated_Sram`, while allowing scripts to pass `Shared_Sram` for the
  generated shared-SRAM TiTok artifacts.
- Updated the main lowering script so it writes `.pte` artifacts directly,
  records FX/runtime summaries, and optionally rewrites surviving
  post-partition q/dq ops to `.out` overloads.
- Added graph/runtime summary helpers so lowering runs can persist compact JSON
  inventories of FX nodes, delegate calls, kernel calls, and execution-plan
  structure.
- Added a narrow Cortex-M BMM rewrite pass that converts qualifying
  `dq -> bmm -> q` fallback islands into Cortex-M quantized batch matmul ops.

What has been proven:

- The host can configure and build the CM33 runner.
- The SDK path and Arm GNU toolchain path are known.
- The UCM board has the expected Ethos-U reserved DDR region at
  `0xC0000000`.
- The Linux `/dev/mem` model-loading method from the tutorial is blocked on
  this BSP with `PermissionError: [Errno 1] Operation not permitted`.
- The U-Boot `fatload` model-loading path works.
- MobileNet can run successfully through CM33 remoteproc and Ethos-U65 on this
  UCM board.

What has not yet been proven:

- A TiTok `.pte` has not yet been successfully run through the CM33 runner on
  the UCM board.
- The current 128 MB model window has not yet been validated with a real TiTok
  run.
- The current selective-op / helper-op setup has not yet been fully validated
  end to end on the board for TiTok.
- The build still uses `mcimx93evk` board support, so UCM-specific board init
  differences remain a known caveat.

## next steps

1. Update `UCM_IMX93_RUNBOOK.md` so its TiTok memory-layout section matches
   the current source: `128 MB` model window, `16 MB` scratch, `60 MB` method
   allocator, and remaining planned-buffer space inside the 256 MB
   `ethosu_region`.

2. Rebuild the runner from
   `/Users/sruthipolali/Documents/Playground/Executorch_runner_cm33`.

   Current standard build environment:

   ```bash
   cd /Users/sruthipolali/Documents/Playground/Executorch_runner_cm33

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

3. If TiTok requires CPU fallback/helper ops that are not already registered,
   configure with selective portable op generation. The current CMake supports
   either an explicit op list or generating from a `.pte`.

   Example using a model:

   ```bash
   cd /Users/sruthipolali/Documents/Playground/Executorch_runner_cm33

   env \
     ARMGCC_DIR=/Users/sruthipolali/.mcuxpressotools/arm-gnu-toolchain-14.2.rel1-darwin-arm64-arm-none-eabi \
     SdkRootDirPath=/Volumes/Media \
     MCUX_VENV_PATH=/Users/sruthipolali/.mcuxpressotools/.mcux-venv-3.12/bin \
     BOARD=mcimx93evk \
     cmake --preset debug \
       -DET_DIR_PATH=/Users/sruthipolali/Documents/Playground/executorch-main \
       -DEXECUTORCH_SELECT_OPS_MODEL=/Users/sruthipolali/Documents/Playground/sentinel_image_compression/titok-deploy-tools/outputs/rewrite_a16w8_balanced_pipeline/lowering_shared_sram/titok_s128_encoder_ethosu_u65_a16w8_shared_sram.pte
   ```

4. Upload the rebuilt ELF to the board.

   Host-side uploader:

   ```bash
   /Volumes/Media/executorch/.venv/bin/python \
     /Volumes/Media/executorch/serial_put_files.py \
     --port /dev/cu.usbserial-02BE3471 \
     /Users/sruthipolali/Documents/Playground/Executorch_runner_cm33/debug/executorch_runner_cm33.elf
   ```

   Board-side install:

   ```sh
   cp -f /tmp/executorch_runner_cm33.elf /lib/firmware/executorch_runner_cm33.elf
   chmod 644 /lib/firmware/executorch_runner_cm33.elf
   md5sum /lib/firmware/executorch_runner_cm33.elf
   ```

5. Put the shared-SRAM TiTok `.pte` on the first SD / FAT boot partition so
   U-Boot can load it.

   Desired board-side destination:

   ```text
   /run/media/boot-mmcblk0p1/titok_s128_encoder_ethosu_u65_a16w8_shared_sram.pte
   ```

6. Reboot the board, interrupt U-Boot, and load TiTok into the runner's model
   address.

   ```text
   fatload mmc 0:1 0xc0000000 titok_s128_encoder_ethosu_u65_a16w8_shared_sram.pte
   boot
   ```

7. Start the CM33 firmware from Linux.

   ```sh
   echo executorch_runner_cm33.elf > /sys/class/remoteproc/remoteproc0/firmware
   echo start > /sys/class/remoteproc/remoteproc0/state
   sleep 15
   cat /sys/class/remoteproc/remoteproc0/state
   cat /sys/kernel/debug/remoteproc/remoteproc0/trace0
   ```

8. Inspect `trace0`.

   Success would look similar to the MobileNet success case:

   ```text
   NPU config match
   NPU arch match
   bus_status_error 0x0
   cmd_end_reached 0x1
   1 inferences finished
   ```

9. If TiTok fails, classify the failure before changing code:

   - If `trace0` reports missing kernel/operator registration, adjust selective
     op generation.
   - If the NPU starts but stalls or reports command stream errors, inspect
     Vela/shared-SRAM configuration and the PTE header/debug logs.
   - If remoteproc rejects the ELF, inspect program headers and CM33 RAM window
     placement.
   - If the board boots but model data is corrupt, re-check the U-Boot
     `fatload` byte count and confirm it fits inside the 128 MB model window.

## all the steps taken so far

1. Started from the Arm Learn tutorial for observing Ethos-U on NXP i.MX93.

2. Confirmed the local runner repository already existed at:
   `/Users/sruthipolali/Documents/Playground/Executorch_runner_cm33`

3. Identified that the tutorial target and the real board were not the same:
   the tutorial is FRDM/EVK-oriented, while the hardware here is UCM-i.MX93.

4. Verified the runner project was configured around the available MCUX SDK
   board target `mcimx93evk`.

5. Looked for a UCM-specific MCUX board definition in the installed SDK and
   did not confirm one.

6. Found the installed MCUX SDK under `/Volumes/Media/mcuxsdk`.

7. Established the build environment:

   - `ARMGCC_DIR=/Users/sruthipolali/.mcuxpressotools/arm-gnu-toolchain-14.2.rel1-darwin-arm64-arm-none-eabi`
   - `SdkRootDirPath=/Volumes/Media`
   - `MCUX_VENV_PATH=/Users/sruthipolali/.mcuxpressotools/.mcux-venv-3.12/bin`
   - `BOARD=mcimx93evk`

8. Confirmed the MCUX SDK had the expected i.MX93 linker script:
   `/Volumes/Media/mcuxsdk/mcuxsdk/devices/i.MX/i.MX93/MIMX9352/gcc/MIMX9352xxxxM_ram.ld`

9. Configured and built the runner on macOS for the available `mcimx93evk`
   SDK target.

10. Applied the SDK patch script from the runner repo.

11. Patched the SDK linker script so GOT sections are copied into RAM at
    startup.

12. Patched the Ethos-U SDK logging header so useful NPU logs appear in the
    remoteproc trace buffer.

13. Used the serial connection to inspect the live UCM board state.

14. Confirmed the board uses `ucm-imx93-ethosu.dtb`.

15. Confirmed the live UCM reserved-memory layout exposes one
    `ethosu_region@0xC0000000` region of `256 MB`.

16. Confirmed `/dev/ethosu0` exists on the board.

17. Compared that UCM memory map to the original runner assumptions.

18. Found that the original runner used an EVK-style scratch/work region at
    `0xA8000000`, which did not match the UCM board.

19. Retargeted the runner memory layout so the model and runtime buffers all
    live inside the UCM `ethosu_region` at `0xC0000000`.

20. Initially used a small MobileNet-oriented model window, then later moved
    the current source to a larger `128 MB` model window for TiTok-sized PTEs.

21. Moved scratch, method allocator, and planned buffers after the model
    window to avoid overlap with externally loaded model data.

22. Added or kept debug logging around the UCM memory layout and external PTE
    inspection so `trace0` can show useful model-size and Vela-stream clues.

23. Reduced linker heap and stack sizes to `0x300` each to keep the CM33 ELF
    within the UCM remoteproc loader's accepted memory window.

24. Added linker-script customization in the runner build so program headers
    are shaped in a way the UCM remoteproc loader accepts.

25. Uploaded the CM33 ELF to the board over USB serial using:
    `/Volumes/Media/executorch/serial_put_files.py`

26. Installed the uploaded ELF on the board as:
    `/lib/firmware/executorch_runner_cm33.elf`

27. Tried the tutorial's Linux `/dev/mem` model load path and found that this
    UCM BSP blocks it with a permission error.

28. Switched to the reliable U-Boot load path:
    `fatload mmc 0:1 0xc0000000 <model>.pte`

29. Loaded `mobilenetv2_u65.pte` into DDR at `0xC0000000` through U-Boot.

30. Booted Linux and started the CM33 firmware with Linux `remoteproc`.

31. Read `/sys/kernel/debug/remoteproc/remoteproc0/trace0`.

32. Confirmed a successful MobileNet run through Ethos-U65 with:

    ```text
    NPU config match
    NPU arch match
    bus_status_error 0x0
    cmd_end_reached 0x1
    1 inferences finished
    ```

33. Restored the board's U-Boot boot settings after the MobileNet run:

    - `bootcmd=run bsp_bootcmd; run distro_bootcmd`
    - `bootdelay=2`

34. Created the UCM-specific runbook:
    `/Users/sruthipolali/Documents/Playground/Executorch_runner_cm33/UCM_IMX93_RUNBOOK.md`

35. Located local TiTok lowered `.pte` artifacts, including a shared-SRAM
    version under:
    `/Users/sruthipolali/Documents/Playground/sentinel_image_compression/titok-deploy-tools/outputs/rewrite_a16w8_balanced_pipeline/lowering_shared_sram/`

36. In `sentinel_image_compression`, added TiTok encoder wrapper variants for
    reshape-only batch handling, explicit BMM attention, source-level matmul
    attention, stock MHA attention, and source-level SDPA attention.

37. Extended the TiTok PTQ scripts so experiments can choose an encoder
    variant and optionally request matmul quantization.

38. Changed the local Ethos-U compile-spec compatibility shim to use
    `Dedicated_Sram` by default, while keeping script-level support for
    explicit `Shared_Sram` runs.

39. Added lowering scripts and boundary-tracing scripts to compare the baseline
    TiTok path against no-unsqueeze, BMM-attention, source-matmul, SDPA, and
    stock-MHA variants.

40. Added summary helpers for FX graph boundaries and ExecuTorch runtime
    programs so each lowering run can write useful JSON diagnostics.

41. Added a post-partition q/dq fix pass that rewrites surviving
    `quantized_decomposed` q/dq nodes to explicit `.out` overloads after
    Ethos-U partitioning.

42. Added a Cortex-M BMM rewrite helper that can turn narrow
    `dequantize -> bmm -> quantize` fallback islands into Cortex-M quantized
    batch matmul.

43. Generated multiple TiTok Ethos-U `.pte` artifacts and diagnostics,
    including A16W8 shared-SRAM, dedicated-SRAM, no-unsqueeze, and source-matmul
    variants.

44. Added or discovered CMake support in the CM33 runner for selective portable
    helper ops using:

    - `EXECUTORCH_SELECT_OPS_LIST`
    - `EXECUTORCH_SELECT_OPS_MODEL`
    - `ET_DIR_PATH`
    - `EXECUTORCH_PYTHON_EXECUTABLE`

45. Left the system ready for the next major test: transfer and run the
    shared-SRAM TiTok `.pte` with the larger CM33/UCM memory layout.
