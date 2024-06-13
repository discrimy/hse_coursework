import subprocess
import tempfile
from pathlib import Path

import click


@click.command()
@click.argument(
    "source",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path
    ),
)
@click.argument(
    "destination",
    type=click.Path(file_okay=True, dir_okay=False, writable=True, path_type=Path),
)
def video2text(source: Path, destination: Path) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        if source.suffix.lower() == ".wav":
            source_wav = source
        elif source.suffix.lower() == ".mp4":
            source_wav_raw = Path(temp_dir) / "source_raw.wav"
            subprocess.check_call(
                ["ffmpeg", "-i", str(source.absolute()), str(source_wav_raw.absolute())]
            )
            source_wav = Path(temp_dir) / "source.wav"
            subprocess.check_call(
                [
                    "sox",
                    str(source_wav_raw),
                    "--type",
                    "wav",
                    "-e",
                    "gsm",
                    "-r",
                    "8000",
                    str(source_wav.absolute()),
                ]
            )
        else:
            raise click.UsageError(f"Unknown source file type: {source.name}")

        # Загружаем библиотеки внутри функции, т.к. они долго загружаются
        from hse_coursework.text import prettify_dialog, merge_dialogs
        from hse_coursework.utils.srt import format_as_srt
        from hse_coursework.service import process

        dialog_raw = process(source_wav)
        dialog_pretty = prettify_dialog(merge_dialogs({"1": dialog_raw}))

        if destination.suffix.lower() == ".json":
            destination.write_text(dialog_pretty.model_dump_json(indent=2))
        elif destination.suffix.lower() == ".srt":
            destination.write_text(format_as_srt(dialog_pretty))
        else:
            raise click.UsageError(f"Unknown destination file type: {destination.name}")


if __name__ == "__main__":
    video2text()
