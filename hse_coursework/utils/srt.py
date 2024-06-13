import datetime
import textwrap

from hse_coursework.text import DialogWithChannelsSchema


def format_as_srt(dialog: DialogWithChannelsSchema) -> str:
    result = []
    for i, replica in enumerate(dialog.dialog, start=1):
        start_formatted = str(datetime.timedelta(seconds=replica.start))[:-3].replace(
            ".", ","
        )
        end_formatted = str(datetime.timedelta(seconds=replica.end))[:-3].replace(
            ".", ","
        )
        text = textwrap.dedent(f"""
            {i}
            {start_formatted} --> {end_formatted}
            {replica.text}
        """).strip()
        if text[0].islower():
            text = text[0].upper() + text[1:]
        result.append(text)
    return "\n\n".join(result)
