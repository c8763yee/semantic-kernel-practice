import asyncio
import json
from importlib import import_module
from pathlib import Path
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
from semantic_kernel.planners import SequentialPlanner
from semantic_kernel.services.ai_service_selector import AIServiceSelector

import loggers
import sk_plugin.kernel_plugins as kernel_plugins

loggers.setup_package_logger("sk_plugin", file_level=loggers.NOTSET, console_level=loggers.INFO)
semantic_plugins_dir: Path = Path(__file__).parent / "sk_plugin" / "semantic_plugins"
semantic_plugins_str: list[str] = [
    dirs.name for dirs in semantic_plugins_dir.iterdir() if dirs.is_dir()
]

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

    def update_user_message(self, new_message: str, index: int = 0) -> None:
        self.messages[index].content = new_message


class CustomKernel(Kernel):
    history: ChatHistory = CustomChatHistory(messages=[])

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

        # import semantic plugin

    def handle_builtin_command(self, command: str) -> bool:
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
                    self.history.update_user_message(new_url, 1)
                return True
            case _:
                return False

    def import_semantic_plugin(self, plugin_dir: Path) -> dict[str, KernelPlugin] | None:
        plugins = {}
        for dirs in plugin_dir.iterdir():
            if dirs.is_dir() is False:
                continue

            plugin_name = dirs.name

            plugin = kernel.add_plugin(parent_directory=str(plugin_dir), plugin_name=plugin_name)
            plugins[plugin_name] = plugin
        return plugins

    def setup_planner(
        self, planner_cls: type[SequentialPlanner], service_id: str, **kwargs: Any
    ) -> SequentialPlanner:
        self.planner = planner_cls(kernel=self, service_id=service_id, **kwargs)
        return self.planner


def package_str_to_kernel_plugin(plugin_name: str) -> KernelPlugin:
    kernel_modules = import_module(".kernel_plugins", package="sk_plugin")
    classes = getattr(kernel_modules, plugin_name, None)
    if classes is None:
        raise ValueError(f"Plugin {plugin_name} not found")
    return KernelPlugin.from_object(plugin_name=classes.__name__, plugin_instance=classes())


# define the plugins and services
services: list = [OpenAIChatCompletion(service_id="main")]
kernel_plugin: list[KernelPlugin] = list(map(package_str_to_kernel_plugin, kernel_plugins.__all__))


async def main(kernel: Kernel):
    service_id = "main"

    # get chat completion object
    chat_completion: OpenAIChatCompletion = kernel.get_service(service_id=service_id)

    # get the prompt execution setting from service id
    execution_setting = OpenAIChatPromptExecutionSettings(service_id=service_id, tool_choice="auto")

    # apply plugin function call behavior
    execution_setting.function_call_behavior = FunctionCallBehavior.EnableFunctions(
        # auto_invoke=True, filters={"included_plugins": kernel_plugins.__all__}
        auto_invoke=True,
        filters={"included_plugins": [*kernel_plugins.__all__, *semantic_plugins_str]},
    )
    included_plugins = ", ".join([*kernel_plugins.__all__, *semantic_plugins_str])
    print(f"included_plugins: {included_plugins}")
    arguments: KernelArguments = KernelArguments(settings=execution_setting)

    # invoke the kernel and get the result from user input
    url = input("Please provide a youtube video URL: ")
    kernel.history.add_user_message(url)
    while True:
        user_input = input("What do you want to do?: ")
        if kernel.handle_builtin_command(user_input) is True:
            continue

        kernel.history.add_user_message(user_input)
        result = (
            await chat_completion.get_chat_message_contents(
                chat_history=kernel.history,
                kernel=kernel,
                settings=execution_setting,
                arguments=arguments,
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
        kernel = CustomKernel(services=services, plugins=kernel_plugin)
        kernel.import_semantic_plugin(Path(__file__).parent / "sk_plugin" / "semantic_plugins")
        asyncio.run(main(kernel=kernel))
    except Exception as exc:
        print(f"An error occurred: {exc!r}")
    finally:
        print('Please check the "chat_history.json" file for the chat history')
        with open("chat_history.json", "w") as f:
            json.dump(kernel.history.to_dict(), f, indent=4, ensure_ascii=False)
