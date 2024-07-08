import json
import os
from typing import Annotated, Any

from semantic_kernel.functions import kernel_function
from yt_dlp import YoutubeDL

__all__ = ["YoutubeDLPlugin", "WeatherPlugin"]


def get_obj_from_key(obj: list | dict, key: str | list[str | int]) -> Any:
    """Get the object (list or dict) by (nested) key.

    Parameters
    ----------
    obj: list | dict
        The object to extract the value from.
    key: str | list[str | int]
        The key to extract the value from the object.

    Returns
    -------
    Any
        The value of the object by the key.

    """
    if not key:
        return obj

    if isinstance(key, str | int):
        key = [key]

    try:
        for k in key:
            match obj:
                case dict():
                    obj = obj.get(k)

                case list():
                    obj = obj[int(k)]

                case _:
                    break

        return obj
    except (IndexError, TypeError, KeyError):
        return None


class WeatherPlugin:
    """A sample plugin that provides weather information for cities."""

    @kernel_function(name="get_weather_for_city", description="Get the weather for a city")
    def get_weather_for_city(
        self, city: Annotated[str, "The input city"]
    ) -> Annotated[str, "The output is a string"]:
        if city == "Boston":
            return "61 and rainy"
        if city == "London":
            return "55 and cloudy"
        if city == "Miami":
            return "80 and sunny"
        if city == "Paris":
            return "60 and rainy"
        if city == "Tokyo":
            return "50 and sunny"
        if city == "Sydney":
            return "75 and sunny"
        if city == "Tel Aviv":
            return "80 and sunny"
        return "31 and snowing"


class YoutubeDLPlugin:
    @staticmethod
    def _download(
        video_url: Annotated[str, "The youtube video URL"],
        ydl: Annotated[YoutubeDL, "YoutubeDL instance"],
    ) -> Annotated[str, "The path to the downloaded file"]:
        ydl.download([video_url])
        return ydl.prepare_filename(ydl.extract_info(video_url, download=False))

    @staticmethod
    def _extract_info(
        video_url: Annotated[str, "The youtube video URL"],
        ydl: Annotated[YoutubeDL, "YoutubeDL instance"],
    ) -> Annotated[dict, "The information of the video"]:
        video_info = ydl.extract_info(video_url, download=False)
        os.makedirs("url_info", exist_ok=True)
        with open(f'url_info/{video_info["id"]}.json', "w", encoding="utf-8") as f:
            json.dump(video_info, f, indent=2, ensure_ascii=False)

        return ydl.sanitize_info(video_info)

    @kernel_function(
        name="download",
        description="Download a video or audio from a youtube video URL",
    )
    def download(
        self,
        video_url: Annotated[str, "The youtube video URL"],
        kwargs_json: Annotated[str, "The YoutubeDL.params as JSON string"],
    ) -> Annotated[str, "The path to the downloaded file"]:
        params = json.loads(kwargs_json)
        print(f"params: {params}")

        with YoutubeDL(params) as ydl:
            self._extract_info(video_url, ydl)
            return self._download(video_url, ydl)

    @kernel_function(
        name="get_video_info",
        description="Extract information from a youtube video",
    )
    def get_video_info(
        self,
        video_url: Annotated[str, "The youtube video URL"],
        nested_key: Annotated[
            list[str | int] | str, "The key to extract the information from info (nested) keys"
        ],
    ) -> Annotated[dict | None, "The information of the video"]:
        print(f"key: {nested_key}")
        with YoutubeDL() as ydl:
            video_info = self._extract_info(video_url, ydl)

        return get_obj_from_key(video_info, nested_key)
