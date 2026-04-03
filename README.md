# AI Dev Tools

Collection of small CLI utilities that help inspect and summarize AI-related artifacts within the Flixie workspace.

## Setup

`task dev:bootstrap-env ; rehash`

## Available commands

- `get-transcript INPUT_FILE OUTPUT_FILE` — transcribe audio via Whisper; supports `--model`, `--word-timestamps`, and `-` for OUTPUT_FILE to stream the text to stdout. Word timestamps require CPU precision, so the CLI automatically switches from MPS to CPU when `--word-timestamps` is used on Apple Silicon.

Use `task help` for the full Taskfile command list once dependencies are bootstrapped.
