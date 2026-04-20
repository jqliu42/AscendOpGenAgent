#!/usr/bin/env python3
"""
Triton IR Extractor - Extract compiler IR from Triton kernel compilation.

This script runs a Triton kernel script with TRITON_DEBUG enabled,
extracts the last pass IR from bishengir-compile output, and saves
the IR files for analysis.
"""

import os
import sys
import subprocess
import re
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple


def find_bishengir_compile() -> Optional[str]:
    """Find bishengir-compile binary in the system."""
    try:
        result = subprocess.run(
            ["which", "bishengir-compile"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    triton_paths = [
        "/usr/local/python3.11.0/lib/python3.11/site-packages/triton/backends/ascend/bishengir/bin/bishengir-compile",
        os.path.expanduser("~/.local/lib/python3.11/site-packages/triton/backends/ascend/bishengir/bin/bishengir-compile"),
    ]

    for path in triton_paths:
        if os.path.exists(path):
            return path

    return None


def run_triton_script(script_path: str, work_dir: str) -> Tuple[str, List[str]]:
    """
    Run Triton script with TRITON_DEBUG enabled to trigger IR dump.

    Returns:
        Tuple of (log_content, dump_directories)
    """
    env = os.environ.copy()
    env["TRITON_DEBUG"] = "1"
    env["TRITON_ALWAYS_COMPILE"] = "1"
    env["TRITON_DISABLE_LINE_INFO"] = "0"
    env["TRITON_DISABLE_FFTS"] = "1"

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
        env=env,
        cwd=work_dir
    )

    log_content = result.stdout + "\n" + result.stderr

    dump_dirs = re.findall(r"Dumping intermediate results to (\S+)", log_content)

    return log_content, dump_dirs


def extract_kernel_name(ttadapter_path: str) -> str:
    """Extract kernel name from ttadapter.mlir file."""
    with open(ttadapter_path, 'r') as f:
        content = f.read()

    patterns = [
        r'func\.func @(\w+)',
        r'tt\.func @(\w+)',
        r'module @(\w+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            return match.group(1)

    return "unknown"


def run_bishengir_compile(
    ttadapter_path: str,
    bishengir_compile: str,
    output_dir: str
) -> Optional[str]:
    """
    Run bishengir-compile with --mlir-print-ir-after-all to extract IR.

    Returns:
        Path to the last pass IR file, or None if failed.
    """
    bishengir_bin = os.path.dirname(bishengir_compile)
    libdevice_bc = os.path.join(
        os.path.dirname(bishengir_bin),
        "lib",
        "libdevice.10.bc"
    )
    #"--target=Ascend950PR_957c",
    # "--disable-ffts",
    # "--enable-vf-fusion",
    # f"--append-bisheng-options=-cce-link-aicore-ll-module {libdevice_bc}",
    # "--enable-vf-merge-level=1",
    cmd = [
        bishengir_compile,
        "--enable-auto-multi-buffer=False",
        "--enable-auto-bind-sub-block=True",
        "--enable-hfusion-compile=true",
        "--enable-hivm-compile=true",
        "--enable-triton-kernel-compile=true",
        "--mlir-print-ir-after-all",
        ttadapter_path
    ]

    env = os.environ.copy()
    env["PATH"] = f"{bishengir_bin}:{env.get('PATH', '')}"

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env
    )

    output = result.stdout + "\n" + result.stderr

    #print(f"=======output = {output}")

    matches = list(re.finditer(r"IR Dump After (\S+)", output))
    if not matches:
        return None

    last_match = matches[-1]
    last_pass_name = last_match.group(1)
    last_pass_start = last_match.start()

    ir_content = output[last_pass_start:]

    output_file = os.path.join(output_dir, f"last_pass_{last_pass_name}.mlir")
    with open(output_file, 'w') as f:
        f.write(ir_content)

    return output_file


def extract_ir(
    script_path: str,
    output_dir: str,
    work_dir: Optional[str] = None
) -> List[dict]:
    """
    Main function to extract IR from a Triton script.

    Args:
        script_path: Path to the Triton kernel script.
        output_dir: Directory to save extracted IR files.
        work_dir: Working directory for running the script.

    Returns:
        List of dicts with kernel_name and ir_path for each extracted kernel.
    """
    if work_dir is None:
        work_dir = os.path.dirname(os.path.abspath(script_path))

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    bishengir_compile = find_bishengir_compile()
    if not bishengir_compile:
        print("ERROR: bishengir-compile not found. Please ensure Triton with Ascend backend is installed.")
        return []

    print(f"Found bishengir-compile: {bishengir_compile}")

    print(f"\n===== Phase 1: Running Triton script to trigger compilation =====")
    log_content, dump_dirs = run_triton_script(script_path, work_dir)
    print(f"Found {len(dump_dirs)} kernel dump directories")

    results = []
    seen_kernels = set()

    for dump_dir in dump_dirs:
        if not os.path.isdir(dump_dir):
            continue

        ttadapter_path = os.path.join(dump_dir, "kernel.ttadapter.mlir")
        if not os.path.isfile(ttadapter_path):
            continue

        kernel_name = extract_kernel_name(ttadapter_path)

        if kernel_name in seen_kernels:
            print(f"Skipping duplicate kernel '{kernel_name}' (autotune config variant)")
            continue

        seen_kernels.add(kernel_name)
        print(f"\nProcessing kernel: {kernel_name}")

        ir_path = run_bishengir_compile(ttadapter_path, bishengir_compile, output_dir)
        #print(f"==========!!!! ir_path = {ir_path}, ttadapter_path = {ttadapter_path}, bishengir_compile = {bishengir_compile}, output_dir = {output_dir}")
        if ir_path:
            final_path = os.path.join(output_dir, f"{kernel_name}_last_pass.mlir")
            shutil.move(ir_path, final_path)
            print(f"Extracted IR: {final_path}")
            results.append({
                "kernel_name": kernel_name,
                "ir_path": final_path
            })
        else:
            print(f"WARNING: Failed to extract IR for {kernel_name}")

    return results


def cleanup_ir_dir(ir_dir: str):
    """Clean up the IR output directory."""
    if os.path.exists(ir_dir):
        shutil.rmtree(ir_dir)
        print(f"Cleaned up IR directory: {ir_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract Triton compiler IR")
    parser.add_argument("script", help="Path to the Triton kernel script")
    parser.add_argument("--output-dir", "-o", default="./ir_output", help="Output directory for IR files")
    parser.add_argument("--work-dir", "-w", default=None, help="Working directory")
    parser.add_argument("--cleanup", action="store_true", help="Clean up IR directory after extraction")

    args = parser.parse_args()

    results = extract_ir(args.script, args.output_dir, args.work_dir)

    print(f"\n===== Extraction Complete =====")
    print(f"Extracted {len(results)} kernel IR files:")
    for r in results:
        print(f"  - {r['kernel_name']}: {r['ir_path']}")

    if args.cleanup:
        cleanup_ir_dir(args.output_dir)
