import re


class TextProcessor:

    def __init__(
        self,
        normalize_persian=True,
        remove_extra_spaces=True
    ):
        self.normalize_persian = normalize_persian
        self.remove_extra_spaces = remove_extra_spaces


    def process(self, text: str) -> str:

        if text is None:
            return ""


        text = str(text).strip()


        if self.normalize_persian:

            text = (
                text
                .replace("ي", "ی")
                .replace("ك", "ک")
            )


        if self.remove_extra_spaces:

            text = re.sub(
                r"\s+",
                " ",
                text
            )


        return text