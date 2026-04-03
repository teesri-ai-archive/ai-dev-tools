#!/usr/bin/env python3

import json
from pathlib import Path
import typer
import torch

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

try:
    import whisper
except ImportError:  # pragma: no cover
    whisper = None

try:
    import openai_whisper
except ImportError:  # pragma: no cover
    openai_whisper = None

try:
    import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_fraction = seconds % 60
    if hours > 0:
        return f"{hours:02}:{minutes:02}:{seconds_fraction:06.3f}"
    return f"{minutes:02}:{seconds_fraction:06.3f}"


class _SilentTqdm:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, *args, **kwargs):
        pass

    def __getattr__(self, _):
        return lambda *_, **__: None

app = typer.Typer(add_completion=False)
console = Console()


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_size: str, device: str):
    module = whisper
    load_fn = getattr(module, "load_model", None) if module else None
    if load_fn is None and openai_whisper is not None:
        module = openai_whisper
        load_fn = getattr(module, "load_model", None)
    if load_fn is None:
        raise typer.Exit(
            "Whisper model loader not available. Install or upgrade "
            "openai-whisper (provides `whisper.load_model`)."
        )
    return load_fn(model_size, device=device)


@app.command()
def main(
    input_audio: Path = typer.Argument(..., exists=True, help="Input audio file"),
    output_file: str = typer.Argument(
        ..., help="Output file (.txt/.json) or '-' for stdout"
    ),
    word_timestamps: bool = typer.Option(
        False, "--word-timestamps", help="Enable word-level timestamps"
    ),
    model_size: str = typer.Option(
        "large", "--model", help="Whisper model size (tiny, base, small, medium, large)"
    ),
):
    """
    Generate transcript from audio using Whisper.
    """

    console.print(Panel("Whisper Transcription", expand=False))

    device = get_device()
    if word_timestamps and device == "mps":
        console.print(
            "[yellow]Word timestamps require CPU precision. "
            "Switching to CPU to satisfy --word-timestamps.[/yellow]"
        )
        device = "cpu"
    console.print(f"[bold green]Device:[/bold green] {device}")
    console.print(f"[bold green]Model:[/bold green] {model_size}")

    # Load model with spinner
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading model...", start=True)
        model = load_model(model_size, device=device)
        progress.update(task, description="Model loaded ✔")

    # Transcription
    tqdm_backup = None
    if tqdm is not None:
        tqdm_backup = getattr(tqdm, "tqdm", None)
        tqdm.tqdm = lambda *args, **kwargs: _SilentTqdm()
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Transcribing...", start=True)
            result = model.transcribe(
                str(input_audio),
                word_timestamps=word_timestamps,
                verbose=False,
            )
            progress.update(task, description="Transcription complete ✔")
    finally:
        if tqdm is not None and tqdm_backup is not None:
            tqdm.tqdm = tqdm_backup

    # Write output
    try:
        normalized_output = output_file.strip()
        if normalized_output == "-":
            if word_timestamps and "segments" in result:
                for segment in result["segments"]:
                    start = format_timestamp(segment["start"])
                    end = format_timestamp(segment["end"])
                    text = segment.get("text", "").strip()
                    console.print(f"[bold cyan]{start}[/bold cyan] → [bold cyan]{end}[/bold cyan]: {text}")
                return result
            console.print(result["text"])
            return result
        else:
            path = Path(normalized_output)
            if path.suffix.lower() == ".json":
                with open(path, "w") as f:
                    json.dump(result, f, indent=2)
            else:
                with open(path, "w") as f:
                    f.write(result["text"])
    except Exception as e:
        console.print(f"[red]Error writing file:[/red] {e}")
        raise typer.Exit(1)

    console.print(
        Panel(
            f"[bold green]Saved transcript to:[/bold green]\n{output_file}",
            expand=False,
        )
    )


if __name__ == "__main__":
    app()