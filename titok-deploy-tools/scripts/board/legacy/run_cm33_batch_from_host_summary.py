#!/usr/bin/env python3
"""Run a batch of CM33 Ethos-U captures from a host eval summary."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path


def run(cmd: list[str], *, check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, timeout=timeout, text=True, capture_output=True)


def ssh_cmd(args: argparse.Namespace, remote: str, *, check: bool = True, timeout: int | None = 30) -> subprocess.CompletedProcess:
    return run(
        [
            "ssh",
            "-o",
            f"BindAddress={args.bind_address}",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ConnectTimeout=10",
            "-i",
            str(args.identity),
            f"root@{args.board_ip}",
            remote,
        ],
        check=check,
        timeout=timeout,
    )


def scp_to_board(args: argparse.Namespace, local: Path, remote: str) -> None:
    run(
        [
            "scp",
            "-o",
            f"BindAddress={args.bind_address}",
            "-o",
            "StrictHostKeyChecking=no",
            "-i",
            str(args.identity),
            str(local),
            f"root@{args.board_ip}:{remote}",
        ],
        timeout=120,
    )


def wait_for_ssh(args: argparse.Namespace, timeout_s: int = 180) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = ssh_cmd(args, "true", check=False, timeout=15)
        if result.returncode == 0:
            return
        time.sleep(5)
    raise TimeoutError("board SSH did not come back")


def capture_trace(path: Path, port: int, timeout_s: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("", port))
        server.listen(1)
        server.settimeout(max(1.0, deadline - time.monotonic()))
        conn, _ = server.accept()
        with conn, path.open("wb") as f:
            conn.settimeout(2.0)
            while time.monotonic() < deadline:
                try:
                    chunk = conn.recv(4096)
                except socket.timeout:
                    continue
                if not chunk:
                    break
                f.write(chunk)
                f.flush()
                if b"CM33_OUTPUT_HEX_END" in chunk:
                    break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--board-ip", default="169.254.17.191")
    parser.add_argument("--bind-address", default="169.254.24.39")
    parser.add_argument("--identity", default="/Users/sruthipolali/Documents/Playground/.ucm-imx93/id_ed25519", type=Path)
    parser.add_argument("--trace-port", type=int, default=8766)
    parser.add_argument("--capture-timeout", type=int, default=240)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text())
    output_dir = args.output_dir
    traces_dir = output_dir / "board_traces"
    pte = Path(summary["pte_path"])
    pte_name = pte.name
    records = summary["per_image"][: args.count]
    boot_dir = "/run/media/boot-mmcblk0p1"

    ssh_cmd(args, f"mkdir -p {boot_dir}; rm -f {boot_dir}/*.pte {boot_dir}/image_input_blob.bin")
    scp_to_board(args, pte, f"{boot_dir}/{pte_name}")
    board_pte_sha = ssh_cmd(args, f"sha256sum {boot_dir}/{pte_name}").stdout.split()[0]
    if board_pte_sha != summary["pte_sha256"]:
        raise RuntimeError(f"PTE SHA mismatch board={board_pte_sha} host={summary['pte_sha256']}")

    run_records = []
    try:
        for record in records:
            index = int(record["index"])
            stem = record["stem"]
            input_blob = Path(record["board_input_blob"]["path"])
            trace_path = traces_dir / f"{index:03d}_{stem}_board_trace.txt"
            print(f"=== board eval {index}: {stem} ===", flush=True)
            scp_to_board(args, input_blob, f"{boot_dir}/image_input_blob.bin")
            board_input_sha = ssh_cmd(args, f"sha256sum {boot_dir}/image_input_blob.bin").stdout.split()[0]
            if board_input_sha != record["board_input_blob"]["sha256"]:
                raise RuntimeError(f"input SHA mismatch for {stem}: {board_input_sha}")

            ssh_cmd(
                args,
                "fw_setenv bootcmd 'fatload mmc 0:1 0xc0000000 "
                f"{pte_name}; fatload mmc 0:1 0xc7800000 image_input_blob.bin; "
                "run bsp_bootcmd; run distro_bootcmd'; fw_setenv bootdelay 2; reboot",
                check=False,
                timeout=20,
            )
            wait_for_ssh(args)

            capture_error: list[BaseException] = []
            thread = threading.Thread(
                target=lambda: capture_trace(trace_path, args.trace_port, args.capture_timeout),
                daemon=True,
            )
            thread.start()
            time.sleep(1.0)
            ssh_cmd(
                args,
                "echo stop > /sys/class/remoteproc/remoteproc0/state 2>/dev/null || true; "
                "printf 'executorch_runner_cm33.elf' > /sys/class/remoteproc/remoteproc0/firmware; "
                "echo start > /sys/class/remoteproc/remoteproc0/state",
                timeout=30,
            )
            thread.join(args.capture_timeout + 5)
            if thread.is_alive():
                raise TimeoutError(f"trace capture timed out for {stem}")
            text = trace_path.read_text(errors="ignore")
            if "CM33_OUTPUT_HEX_END" not in text or "invoke status = 0" not in text:
                raise RuntimeError(f"incomplete or failed trace for {stem}: {trace_path}")
            run_records.append(
                {
                    "index": index,
                    "stem": stem,
                    "trace": str(trace_path),
                    "board_input_sha256": board_input_sha,
                    "board_pte_sha256": board_pte_sha,
                }
            )
            (output_dir / "board_run_summary.json").write_text(json.dumps({"records": run_records}, indent=2) + "\n")
    finally:
        ssh_cmd(
            args,
            "echo stop > /sys/class/remoteproc/remoteproc0/state 2>/dev/null || true; "
            "fw_setenv bootcmd 'run bsp_bootcmd; run distro_bootcmd'; "
            "fw_setenv bootdelay 2; "
            "cat /sys/class/remoteproc/remoteproc0/state; fw_printenv bootcmd bootdelay",
            check=False,
            timeout=30,
        )

    (output_dir / "board_run_summary.json").write_text(json.dumps({"records": run_records}, indent=2) + "\n")
    print(json.dumps({"records": run_records}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
