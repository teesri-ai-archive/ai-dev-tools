"""CLI: Gemini thinking mode with Flixie video-editor preamble and stub transcript tools."""

from __future__ import annotations

import asyncio
import logging
import os
import textwrap
from typing import Annotated

import typer
from google.genai.types import ThinkingConfig
from langchain_core.tools import InjectedToolArg
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from ai_tools.llms.base import tool
from ai_tools.llms.gemini import GeminiLLM
from ai_tools.llms.token_counter import TokenCounter
from ai_tools.prompt_manager import PromptTemplate, get_and_render_prompt
from ai_dev_tools.code_generation.dsl_emitter import emit_dsl_reference
from ai_dev_tools.code_generation.dsl_parser import DSLValidationError, parse_and_validate

app = typer.Typer(add_completion=False, help="Gemini thinking-mode video-edit assistant (dev CLI).")
console = Console()

# Fill in concrete editing goals, assets, and constraints before running.
DEFAULT_VIDEO_EDIT_USER_PROMPT = """
When the user is talking about the rate-card, update the image on the screen to show the new rate-card (image attached).

File Info:
--- File 0 ---
Id: F8e3wK
Name: rate-card.png

--- File 1 ---
Id: Y1v2XQ
Name: video.mp4

""".strip()


class VideoEditAssistantResponse(BaseModel):
    """Structured final reply after optional tool use."""

    dsl_code: str | None = Field(
        ...,
        description="Complete DSL code to edit the video to the user's request.",
    ),
    exception_message: str | None= Field(
        default=None,
        description="An exception message if the DSL code cannot be generated.",
    )


def _print_dsl_code_panel(dsl_code: str, *, title: str, border_style: str) -> None:
    console.print(
        Panel(
            Syntax(dsl_code, "python", word_wrap=True),
            title=title,
            border_style=border_style,
        )
    )


def _print_text_panel(message: str, *, title: str, border_style: str) -> None:
    console.print(
        Panel(
            message,
            title=title,
            border_style=border_style,
        )
    )


def _video_editor_tools():
    @tool
    async def get_full_transcript(
        logger: Annotated[logging.Logger, InjectedToolArg()],
    ) -> str:
        """Return the full spoken transcript for the video under discussion (plain text, one string)."""
        logger.debug("get_full_transcript invoked")
        return textwrap.dedent("""
            Customer usage is unpredictable. Some may hammer your product, others barely touch it, and even for the same customer, usage can swing wildly from one 
            week to the next. So billing for all of this can be a challenge. In this demo series, we'll build a usage-based billing integration from the ground up 
            using Metronome. By the end, you'll have an example that you can extend. Let's dive right in. Let's start with Metronome. It handles the complexity of 
            billing so that you can focus on building your core product. And here's how Metronome works. You send raw usage events, API calls, storage used, whatever 
            your customers actually do. Billable metrics turn those events into what you actually charge for, like counting total API calls or summing up storage. Your
            rate card is the single source of truth for pricing. Update it once and it applies everywhere. You can even schedule price changes in advance. So no more 
            manual updates. Contracts inherit from the rate card with any custom terms layered on top. And then finally, you get clean invoices that reflect exactly 
            what happened. So from raw data to revenue automatically, and it all happens in real time. This is Nova, our fictional company for the demo series. It's an
            image generation service. So customers send prompts through an API and get back AI-generated images. The price is $1. The pricing is straightforward. 
            Standard images at $0.02, high res at $0.05, and ultra at $0.10. Over the next few episodes, we'll build the billing system that powers this. We'll build a
            full stack application. Vanilla JavaScript for the front end and Python on the back end. And then the Metronome Python SDK will handle all of our billing 
            operations. And that's it. Metronome is the source of truth for our billing data so we won't need a separate database. That's an overview of what we'll be 
            doing next. In the next video, we'll cover authentication and API basics so make sure to subscribe so you can stay up to date. Thanks for watching and see 
            you next time.
        """)

    @tool
    async def get_summary_of_transcript(
        logger: Annotated[logging.Logger, InjectedToolArg()],
    ) -> str:
        """Return a concise summary of what happens in the video (themes, sections, intent)."""
        logger.debug("get_summary_of_transcript invoked")
        return (
            "The speaker opens by framing a common SaaS challenge: customer usage is uneven and can fluctuate significantly, "
            "which makes billing difficult. The video introduces a demo series focused on building a usage-based billing integration "
            "with Metronome and promises a practical, extensible example. It explains the core flow from raw usage events to billable "
            "metrics, then to rate cards, contracts, and final invoices, emphasizing real-time automation. A fictional company called "
            "Nova is presented as the running example, where customers generate images through an API. The pricing model is described "
            "as tiered by image quality, with separate rates for standard, high-res, and ultra outputs. The segment closes by previewing "
            "the upcoming implementation stack (vanilla JavaScript frontend, Python backend, Metronome SDK) and the next episode on "
            "authentication and API basics."
        )

    @tool
    async def get_sectioning_info(
        logger: Annotated[logging.Logger, InjectedToolArg()],
    ) -> list[tuple[str, str]]:
        """Return a list of sections in the video. Each element is a tuple of:
           - human readable section title
           - serialized 'PointInTime' object"""
        logger.debug("get_sectioning_info invoked")
        return [
            ("Simplify billing with Metronome", "ts0:00:00.000"),
            ("How Metronome works", "ts0:00:27.060"),
            ("Nova implementation example", "ts0:01:15.740"),
            ("Next episode on authentication and API basics", "ts0:01:33.560"),
        ]

    @tool
    async def get_time_range_for_spoken_content(
        logger: Annotated[logging.Logger, InjectedToolArg()],
        detailed_description_of_content_to_look_for: str,
    ) -> str:
        """Return a (serialized string form of) thetime range for the spoken content in the video. This will be used by an AI agent to find the spoken content in the video.
        Please be clear and as specific as possible so that the AI agent can locate exactly the spoken content in the video."""
        logger.debug("get_time_range_for_spoken_content invoked with description: {}".format(detailed_description_of_content_to_look_for))
        return "serialized_time_range"

    return [get_full_transcript, get_summary_of_transcript, get_sectioning_info, get_time_range_for_spoken_content]


def _build_system_instruction(logger: logging.Logger) -> str:
    preamble = get_and_render_prompt(PromptTemplate.PROMPT_PREAMBLE, {}, logger)
    dsl_schema = emit_dsl_reference()
    return get_and_render_prompt(
        PromptTemplate.CODING_DSL_RULES_SYSTEM_PROMPT,
        {
            "prompt_preamble": preamble.strip(),
            "dsl_schema": dsl_schema,
        },
        logger,
    )


async def _run(
    user_prompt: str,
    *,
    debug: bool,
) -> None:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        console.print(
            "[red]GEMINI_API_KEY is not set. Export it before running this command.[/red]"
        )
        raise typer.Exit(1)

    logger = logging.getLogger("code_generation")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d:%(funcName)s - - %(message)s",
        force=True,
    )
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    llm = GeminiLLM(api_key=api_key, logger=logger)
    model = llm.get_model(cheap=False, reasoning=True, multimodal=False)
    system_instruction = _build_system_instruction(logger)
    user_message = user_prompt.strip()
    # Use list-based message format so Gemini tool-calling can append function call turns.
    messages = [{"user": user_message}]
    token_counter = TokenCounter()
    tools = _video_editor_tools()

    if debug:
        console.print(
            Panel(system_instruction, title="System prompt", border_style="blue")
        )
        console.print(Panel(user_message, title="User prompt", border_style="blue"))

    with console.status("[bold green]Calling Gemini (thinking mode)…"):
        typed, reasoning = await llm.generate_typed(
            model=model,
            system_instruction=system_instruction,
            messages=messages,
            response_type=VideoEditAssistantResponse,
            token_counter=token_counter,
            tools=tools,
            thinking_config=ThinkingConfig(include_thoughts=True),
        )

    if debug:
        if reasoning:
            console.print(Panel(reasoning, title="Reasoning", border_style="dim"))
        else:
            console.print(
                Panel(
                    "[dim]No reasoning trace returned by the API.[/dim]",
                    title="Reasoning",
                    border_style="dim",
                )
            )

    has_dsl_code = bool(typed.dsl_code)
    has_exception = bool(typed.exception_message)

    if has_dsl_code and has_exception:
        _print_text_panel(
            "AI response error: returned both `dsl_code` and `exception_message`.",
            title="Response Error",
            border_style="red",
        )
        _print_dsl_code_panel(
            typed.dsl_code,
            title="Response (dsl_code)",
            border_style="yellow",
        )
        _print_text_panel(
            typed.exception_message,
            title="Response (exception_message)",
            border_style="yellow",
        )
        return

    if has_dsl_code:
        try:
            parse_and_validate(typed.dsl_code)
        except DSLValidationError as exc:
            _print_dsl_code_panel(
                typed.dsl_code,
                title="Response (invalid dsl_code)",
                border_style="yellow",
            )
            _print_text_panel(
                f"AI response error: generated `dsl_code` failed DSL validation.\n{exc}",
                title="Response Error",
                border_style="red",
            )
            return
        _print_dsl_code_panel(typed.dsl_code, title="Response", border_style="green")
        return

    if has_exception:
        _print_text_panel(typed.exception_message, title="Exception", border_style="red")
        return

    _print_text_panel(
        "Model response did not contain `dsl_code` or `exception_message`.",
        title="Response Error",
        border_style="red",
    )


@app.command()
def main(
    prompt_text: str | None = typer.Option(
        None,
        "--prompt",
        "-p",
        help="Inline video editing user prompt (overridden by --prompt-file if both are set).",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help=(
            "Print system and user prompts before the request; print reasoning after the response."
        ),
    ),
) -> None:
    """
    Load the Flixie prompt preamble, attach stub video-analysis tools, and ask Gemini (thinking
    mode) how to approach the editing request. Set GEMINI_API_KEY in the environment.
    """
    if prompt_text is not None:
        user_prompt = prompt_text
    else:
        user_prompt = DEFAULT_VIDEO_EDIT_USER_PROMPT

    console.print(Panel("[bold]code-generation[/bold] — Gemini + Flixie preamble", expand=False))
    asyncio.run(_run(user_prompt, debug=debug))


if __name__ == "__main__":
    app()
