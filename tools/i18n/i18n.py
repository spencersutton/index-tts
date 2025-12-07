import json
import locale
from pathlib import Path

I18N_JSON_DIR = Path(__file__).parent / "locale"


def load_language_list(language):
    with (I18N_JSON_DIR / f"{language}.json").open(encoding="utf-8") as f:
        return json.load(f)


def scan_language_list():
    return [p.stem for p in I18N_JSON_DIR.iterdir() if p.suffix == ".json"]


class I18nAuto:
    def __init__(self, language=None) -> None:
        if language in {"Auto", None}:
            language = locale.getdefaultlocale()[0]
            # getlocale can't identify the system's language ((None, None))
        if not (I18N_JSON_DIR / f"{language}.json").exists():
            language = "en_US"
        self.language = language
        self.language_map = load_language_list(language)

    def __call__(self, key):
        return self.language_map.get(key, key)

    def __repr__(self) -> str:
        return "Use Language: " + self.language


if __name__ == "__main__":
    i18n = I18nAuto(language="en_US")
    print(i18n)
