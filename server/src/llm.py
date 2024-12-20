import os
import asyncio
import aiohttp
from ollama import AsyncClient
from langcodes import Language


INSTRUCTIONS = """
I'm gonna give you a news story.

- Summarize it in <<LANGUAGE>>.
- Talk as if YOU are reporting the news on <<LANGUAGE>> TV. DON'T mention "The article ...".
- State the news agency (not the literal link) before you start summarizing. (e.g. This story is from ...)
- DO NOT use AI phrases or commentary.
- ONLY USE PARAGRAPH FORMATTING - no lists, bolding, or italics. ONLY PLAIN TEXT.
- Summarize the main points in ONLY 1-2 paragraphs. Be clear and informative.

Here's the story:
"""


LLM_SERVER = os.environ.get("LLM_SERVER")
assert LLM_SERVER, "LLM_SERVER env var not set!"


client = AsyncClient(host=LLM_SERVER)


async def summarize_news(content: str, lang="en"):
    prompt = "\n".join(
        [
            # converts language code to display name before replacing
            INSTRUCTIONS.replace("<<LANGUAGE>>", Language.get(lang).display_name()),
            '"""',
            content,
            '"""',
        ]
    )

    # call llm service
    # example call:
    # curl localhost:11000/api/generate?prompt=tell+me+a+story+that+is+peak+fiction.
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"http://{LLM_SERVER}/api/generate?prompt={prompt}"
        ) as response:
            if response.status != 200:
                raise Exception(f"LLM server returned {response.status} on {prompt}")

            async for chunk in response.content:
                yield chunk
