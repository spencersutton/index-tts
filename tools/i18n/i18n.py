import json
import locale
from pathlib import Path

I18N_JSON_DIR: Path = Path(__file__).parent / "locale"


def load_language_list(language: str) -> dict[str, str]:
    with (I18N_JSON_DIR / f"{language}.json").open(encoding="utf-8") as f:
        return json.load(f)


def scan_language_list() -> list[str]:
    return [path.stem for path in I18N_JSON_DIR.iterdir() if path.name.endswith(".json")]


class I18nAuto:
    language_map: dict[str, str]

    def __init__(self, language: str | None = None) -> None:
        if language in {"Auto", None}:
            language = locale.getdefaultlocale()[0]
            # getlocale can't identify the system's language ((None, None))
        if not (I18N_JSON_DIR / f"{language}.json").exists():
            language = "en_US"
        self.language = language
        assert language is not None, "Unable to determine language."
        self.language_map = load_language_list(language)

    def __call__(self, key: str) -> str:
        return self.language_map.get(key, key)

    def __repr__(self) -> str:
        return f"Use Language: {self.language}"


if __name__ == "__main__":
    i18n = I18nAuto(language="en_US")
    print(i18n)
