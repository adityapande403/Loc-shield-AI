"""Command-line interface (CLI) for LocalShield AI."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from localshield.config import LocalShieldConfig
from localshield.schemas import RedactionMode
from localshield.shield import LocalShield
from localshield.taxonomy import ALL_CATEGORIES, TAXONOMY_VERSION

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="localshield")
def cli():
    """LocalShield AI: Privacy-First Local PII Detection & Redaction Engine."""
    pass


@cli.command("detect")
@click.option("--input", "input_file", type=click.Path(exists=True, dir_okay=False), help="Path to input text file.")
@click.option("--stdin", is_flag=True, help="Read input text from standard input.")
@click.option("--categories", default=None, help="Comma-separated PII categories to filter (e.g. PERSON,PHONE_NUMBER).")
@click.option("--min-confidence", default=0.5, type=float, help="Minimum detection confidence threshold.")
def detect_cmd(input_file: Optional[str], stdin: bool, categories: Optional[str], min_confidence: float):
    """Detect PII entities in text without modifying the content."""
    if stdin:
        text = sys.stdin.read()
    elif input_file:
        text = Path(input_file).read_text(encoding="utf-8")
    else:
        console.print("[red]Error:[/red] Specify either --input <file> or --stdin")
        sys.exit(1)

    cat_set = {c.strip().upper() for c in categories.split(",")} if categories else None

    shield = LocalShield()
    start_time = time.perf_counter()
    candidates = shield.detect(text, categories=cat_set, min_confidence=min_confidence)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    table = Table(title=f"Detected PII Entities ({len(candidates)} total, {elapsed_ms:.2f}ms)")
    table.add_column("Category", style="cyan")
    table.add_column("Span [Start:End]", style="magenta")
    table.add_column("Entity Text", style="green")
    table.add_column("Confidence", style="yellow")
    table.add_column("Source", style="blue")

    for c in candidates:
        table.add_row(
            c.entity_type,
            f"[{c.start}:{c.end}]",
            repr(c.text),
            f"{c.confidence:.2f}",
            c.source,
        )

    console.print(table)


@cli.command("redact")
@click.option("--input", "input_file", type=click.Path(exists=True, dir_okay=False), help="Path to input text file.")
@click.option("--output", "output_file", type=click.Path(dir_okay=False), help="Path to save redacted output.")
@click.option("--stdin", is_flag=True, help="Read input text from standard input.")
@click.option(
    "--mode",
    type=click.Choice(["mask", "pseudonymize", "type_preserving", "selective"], case_sensitive=False),
    default="mask",
    help="Redaction strategy mode.",
)
@click.option("--categories", default=None, help="Comma-separated PII categories to redact.")
@click.option("--reversible", is_flag=True, help="Enable encrypted reversible pseudonymization.")
def redact_cmd(
    input_file: Optional[str],
    output_file: Optional[str],
    stdin: bool,
    mode: str,
    categories: Optional[str],
    reversible: bool,
):
    """Redact sensitive PII from input text or file."""
    if stdin:
        text = sys.stdin.read()
    elif input_file:
        text = Path(input_file).read_text(encoding="utf-8")
    else:
        console.print("[red]Error:[/red] Specify either --input <file> or --stdin")
        sys.exit(1)

    # Check overwrite warning if output file already exists
    if output_file and Path(output_file).exists():
        console.print(f"[yellow]Warning:[/yellow] Output file '{output_file}' already exists and will be overwritten.")

    cat_set = {c.strip().upper() for c in categories.split(",")} if categories else None
    redact_mode = RedactionMode(mode.lower())

    shield = LocalShield()
    result = shield.redact(
        text=text,
        categories=cat_set,
        mode=redact_mode,
        enable_reversible=reversible,
    )

    if output_file:
        Path(output_file).write_text(result.redacted_text, encoding="utf-8")
        console.print(
            f"[green]Success:[/green] Redacted {result.redacted_count} entities in {result.processing_time_ms:.2f}ms -> Saved to '{output_file}'"
        )
        if result.session_id:
            console.print(f"[cyan]Reversible Session ID:[/cyan] {result.session_id}")
    else:
        # Standard stdout
        if stdin:
            sys.stdout.write(result.redacted_text)
        else:
            console.print(result.redacted_text)


@cli.command("benchmark")
@click.option(
    "--backend",
    type=click.Choice(["onnx-int8", "onnx-fp32", "pytorch"], case_sensitive=False),
    default="onnx-int8",
    help="Inference engine backend to test.",
)
@click.option("--iterations", default=50, type=int, help="Repetitions for latency measurement.")
def benchmark_cmd(backend: str, iterations: int):
    """Benchmark end-to-end PII detection and redaction throughput and latency."""
    console.print(f"[bold cyan]Benchmarking LocalShield AI Engine (Backend: {backend}, Repetitions: {iterations})...[/bold cyan]")

    prefer_quantized = backend == "onnx-int8"
    engine_backend = "pytorch" if backend == "pytorch" else "onnx"

    cfg = LocalShieldConfig(
        inference_backend=engine_backend,
        prefer_quantized=prefer_quantized,
    )
    shield = LocalShield(config=cfg)

    sample_texts = [
        "Customer Rahul Sharma called regarding loan at HDFC Bank Mumbai. Mobile: 9876543210, PAN: ABCPE1234F.",
        "Kindly dispatch parcel to Priya Patel at Flat 402, Shanti Heights, Sector 62, Noida, UP 201301.",
        "Server alert from host 192.168.1.105 with key AKIAIOSFODNN7EXAMPLE. Contact admin@localshield.internal.",
        "All systems operational, memory usage at 45% with no personal data present.",
    ]

    # Warmup
    for text in sample_texts:
        shield.redact(text)

    latencies_ms = []
    total_chars = 0

    for _ in range(iterations):
        for text in sample_texts:
            total_chars += len(text)
            t0 = time.perf_counter()
            shield.redact(text)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    import numpy as np

    p50 = np.percentile(latencies_ms, 50)
    p95 = np.percentile(latencies_ms, 95)
    p99 = np.percentile(latencies_ms, 99)
    mean_lat = np.mean(latencies_ms)
    throughput = 1000.0 / mean_lat

    table = Table(title=f"LocalShield Benchmark Results ({backend})")
    table.add_column("Metric", style="cyan")
    table.add_column("Measured Value", style="green")

    table.add_row("P50 Latency", f"{p50:.2f} ms")
    table.add_row("P95 Latency", f"{p95:.2f} ms")
    table.add_row("P99 Latency", f"{p99:.2f} ms")
    table.add_row("Mean Latency", f"{mean_lat:.2f} ms")
    table.add_row("Throughput", f"{throughput:.1f} requests/sec")
    table.add_row("Total Inferences Measured", str(len(latencies_ms)))

    console.print(table)


@cli.command("validate-model")
@click.option("--model", default="./models/localshield_multilingual", help="Path to model directory.")
def validate_model_cmd(model: str):
    """Validate presence and integrity of local model weights, tokenizer, and ONNX graphs."""
    model_dir = Path(model)
    console.print(f"Validating model directory: '{model_dir}'...")

    required_files = [
        "config.json",
        "tokenizer.json",
        "model.safetensors",
        "model.onnx",
        "model_quantized.onnx",
    ]

    all_ok = True
    table = Table(title="Model Artifact Validation")
    table.add_column("Artifact", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Size", style="yellow")

    for fname in required_files:
        fpath = model_dir / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            table.add_row(fname, "[green]FOUND[/green]", f"{size_mb:.2f} MB")
        else:
            table.add_row(fname, "[red]MISSING[/red]", "-")
            all_ok = False

    console.print(table)
    if all_ok:
        console.print("[green]All model artifacts are intact and validated![/green]")
    else:
        console.print("[yellow]Some artifacts are missing. Run training or export scripts if needed.[/yellow]")


@cli.command("serve")
@click.option("--host", default="127.0.0.1", help="Bind host address (default 127.0.0.1).")
@click.option("--port", default=8000, type=int, help="Bind port number (default 8000).")
def serve_cmd(host: str, port: int):
    """Run LocalShield FastAPI service with uvicorn."""
    import uvicorn
    from localshield.api.app import create_app

    console.print(f"[bold green]Starting LocalShield AI Service on http://{host}:{port}...[/bold green]")
    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    cli()
