import asyncio
import re
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
)
from semantic_kernel.functions.function_result import FunctionResult
from semantic_kernel.prompt_template import PromptTemplateConfig

# ================= Type Checking =================
if TYPE_CHECKING:
    from semantic_kernel.connectors.ai.prompt_execution_settings import (
        PromptExecutionSettings,
    )
    from semantic_kernel.functions.kernel_function import KernelFunction
    from semantic_kernel.functions.kernel_plugin import KernelPlugin


load_dotenv()

kernel: Kernel = Kernel()
service_id: str = "yt-dlp"

kernel.add_service(OpenAIChatCompletion(service_id=service_id))

request_settings: "PromptExecutionSettings" = kernel.get_prompt_execution_settings_from_service_id(
    service_id
)
prompt = """
You are a AI for user to setup yt-dlp commandline arguments to download a video from youtube.
Your task is to generate a yt-dlp command based on the given video url or playlist url.
Your can only reply the yt-dlp command and nothing else.
--------------------------------------------------------------------------------

User: {{$arg}}
"""

prompt_template_config: PromptTemplateConfig = PromptTemplateConfig(
    template=prompt,
    execution_settings=request_settings,
    name="yt-dlp",
)

function: "KernelFunction | KernelPlugin" = kernel.add_function(
    function_name="ytdlp_command_generator",
    plugin_name="ytdlp_command_generator",
    prompt_template_config=prompt_template_config,
)


async def main():
    # get input from user and check if the user input contains a valid youtube video URL
    valid_url: bool = False
    yt_url_pattern: str = r"(https?\:\/\/)?(www\.youtube\.com|youtu\.?be)\/.+$"
    while not valid_url:
        user_prompt: str = input("Enter the youtube video URL or playlist URL: ")
        if re.search(yt_url_pattern, user_prompt):
            valid_url = True
        else:
            print("Invalid youtube URL, please enter a valid youtube video URL or playlist URL")

    result: FunctionResult = await kernel.invoke(function, arg=user_prompt)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
