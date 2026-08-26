import pandas as pd


def build_comment_text(row):

    fields = [
        "title",
        "body",
        "advantages",
        "disadvantages"
    ]

    texts = []

    for field in fields:

        value = row.get(field)

        if pd.notna(value):

            value = str(value).strip()

            if value:
                texts.append(value)


    return " ".join(texts)