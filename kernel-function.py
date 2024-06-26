import asyncio
import json
from importlib import import_module
from typing import Any

from dotenv import load_dotenv
from semantic_kernel.connectors.ai.function_call_behavior import FunctionCallBehavior
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
)
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import (  # noqa: E501
    OpenAIChatPromptExecutionSettings,
)
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.functions.kernel_plugin import KernelPlugin
from semantic_kernel.kernel import Kernel
from semantic_kernel.kernel_types import AI_SERVICE_CLIENT_TYPE
from semantic_kernel.services.ai_service_selector import AIServiceSelector

import sk_plugin.KernelPlugins as KernelPlugins

load_dotenv()

SYSTEM_MESSAGE: str = (
    "As a YouTube Download bot, your task is to download a video from YouTube using yt_dlp package"
    "You will receive a YouTube video URL and user input to guide the download process."
    "Generate YoutubeDL.params (a.k.a kwargs) as a JSON string to pass to the download function"
    "(including any postprocessors if available.)\n"
    "Please don't do anything if user task is not related to the video."
).strip()


class CustomChatHistory(ChatHistory):
    def to_dict(self) -> list[dict] | dict:
        return [message.to_dict() for message in self.messages]


class CustomKernel(Kernel):
    history: ChatHistory = CustomChatHistory()

    def __init__(
        self,
        plugins: KernelPlugin | dict[str, KernelPlugin] | list[KernelPlugin] | None = None,
        services: (
            AI_SERVICE_CLIENT_TYPE
            | list[AI_SERVICE_CLIENT_TYPE]
            | dict[str, AI_SERVICE_CLIENT_TYPE]
            | None
        ) = None,
        ai_service_selector: AIServiceSelector | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            plugins=plugins,
            services=services,
            ai_service_selector=ai_service_selector,
            **kwargs,
        )

        # initialize system prompt
        self.history.add_system_message(SYSTEM_MESSAGE)

    def handle_builtin_command(self, command: str) -> tuple[Kernel | bool]:
        """Builtin behavior for the command.

        1. exit
        2. reset
        3. change video.
        """
        match command:
            case "exit" | "quit" | "bye" | "goodbye":
                print("Goodbye!")
                exit()
            case "reset" | "change video":
                new_url = input("Please provide a youtube video URL: ")
                if command == "reset":
                    self.history = ChatHistory()
                    self.history.add_system_message(SYSTEM_MESSAGE)
                    self.history.add_user_message(new_url)
                else:
                    # Assuming there's a method to update a message in history
                    # This line might need adjustment based on how ChatHistory is implemented
                    self.history.update_user_message(new_url)
                return True
            case _:
                return False


def package_str_to_kernel_plugin(plugin_name: str) -> KernelPlugin:
    classes = import_module(f".KernelPlugins.{plugin_name}", package="sk_plugin")
    return KernelPlugin.from_object(plugin_name=classes.__name__, plugin_instance=classes())


# define the plugins and services
services: list[AI_SERVICE_CLIENT_TYPE] = [OpenAIChatCompletion(service_id="youtube_dl")]
plugins: list[KernelPlugin] = list(map(package_str_to_kernel_plugin, KernelPlugins.__all__))
kernel = CustomKernel(services=services, plugins=plugins)


async def main():
    service_id = "youtube_dl"

    # get chat completion object
    chat_completion: OpenAIChatCompletion = kernel.get_service(service_id=service_id)

    # get the prompt execution setting from service id
    execution_setting = OpenAIChatPromptExecutionSettings(service_id=service_id, tool_choice="auto")

    # apply plugin function call behavior
    execution_setting.function_call_behavior = FunctionCallBehavior.EnableFunctions(
        auto_invoke=True, filters={"included_plugins": KernelPlugins.__all__}
    )

    # invoke the kernel and get the result from user input
    url = input("Please provide a youtube video URL: ")
    kernel.history.add_user_message(url)
    while True:
        user_input = input("What do you want to do for this video?: ")
        if kernel.handle_builtin_command(user_input) is True:
            continue

        kernel.history.add_user_message(user_input)
        result = (
            await chat_completion.get_chat_message_contents(
                chat_history=kernel.history,
                settings=execution_setting,
                kernel=kernel,
                arguments=KernelArguments(),
            )
        )[0]
        usage_text = "Usage: {text}".format(
            text=", ".join(
                f"{key}: {value}" for key, value in dict(result.inner_content.usage).items()
            )
        )
        print(f"AI: {result!s}", f"usage: {usage_text}", sep="\n", end="\n\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"An error occurred: {exc!r}")
    finally:
        print('Please check the "chat_history.json" file for the chat history')
        with open("chat_history.json", "w") as f:
            json.dump(kernel.history.to_dict(), f, indent=4, ensure_ascii=False)
