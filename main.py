import functools
import logging
import string
from pathlib import Path
from pprint import pprint
from typing import Tuple, List

import re
from copy import deepcopy
import whisperx  # type: ignore
from transformers import Wav2Vec2ForCTC  # type: ignore
from whisperx.asr import FasterWhisperPipeline  # type: ignore

from hse_coursework.schemas import ReplicaSchema, DialogSchema
from hse_coursework.text import prettify_dialog, merge_dialogs
from hse_coursework.types import Device
from hse_coursework.utils.iterators import window
from hse_coursework.utils.timer import measure_performance

import nltk
nltk.download('punkt')

WhisperxAlignMetadata = dict

DEVICE = Device.CPU


@functools.lru_cache()
def get_model() -> Tuple[FasterWhisperPipeline, Wav2Vec2ForCTC, WhisperxAlignMetadata]:
    """Загрузить модели для конвертации"""
    model_size = "large-v2"
    if DEVICE == Device.CPU:
        compute_type = "float32"
    else:
        compute_type = "float16"

    with measure_performance() as measure:
        model = whisperx.load_model(
            model_size,
            device=DEVICE,
            compute_type=compute_type,
            language="ru",
        )
    print("Loaded whisperx model: %.2fs.", measure.delta)

    with measure_performance() as measure:
        model_a, metadata = whisperx.load_align_model(
            language_code="ru",
            device=DEVICE,
        )
    print("Loaded whisperx align model: %.2fs.", measure.delta)

    return model, model_a, metadata


def clean_text(text: str) -> str:
    """Убрать из текста шум, созданные из-за особенности обучения модели whisper"""
    # Предподготовка текста
    # Сжатие пробелов
    text = re.sub("  +", " ", text)

    regexs_sub = [
        # Удаление фраз и звуков-галлюцинаций
        # Фразы типа ТРЕВОЖНАЯ МУЗЫКА, ГРУСТНАЯ МУЗЫКА
        r"([А-Я]+ )*МУЗЫКА",
        r"ПОДПИШИСЬ",
        r"КОНЕЦ",
        r"ФИЛЬМА",
        r"НА КАНАЛ",
        r"(ты! )?(Ух ты!)+",
        r"Смотрите продолжение в следующей серии(\.\.\.)?",
        r"Продолжение следует(\.\.\.)?",
        r"Играет музыка",
        r"Субтитры делал \w+",
        r"(Редактор|Корректор)( субтитров)? [А-Я]\.[А-Я][а-я]+",
        r"Субтитры (делал|добавил|сделал) [\w.]+",
        # Однобуквенные повторения
        # - через дефис или провел или без разделителя
        # - возможна другая буква в начале (напр. заглавная)
        # - может быть восклицательный знак в конце
        r"\b\w[ -]?(\w)[ -]?(?:\1[ -]?)+\1\b!?",
        # Повторение одного и того же слова или слога 3+ раз
        r"(.+?) (?:\1 )+\1",
        # Странные символы (эмодзи, другие языки кроме русского или английского, т.д.)
        rf'[^a-zA-Zа-яА-ЯёЁ\d\s{re.escape(string.punctuation + "«»")}]',
    ]
    for regex in regexs_sub:
        text = re.sub(regex, "", text)
        # Сжатие пробелов, нужно для корректной работы многословных фильтров после работы предыдущего фильтра
        text = re.sub("  +", " ", text)

    # Постобработка текста
    # Удаление лишних знаков до и после текста
    text = re.sub(r"^[ .]+", "", text)
    text = re.sub(r"[^\w.!?]+$", "", text)
    return text


EMPTY_WAV = (
    b"RIFF4\x00\x00\x00WAVEfmt \x14\x00\x00\x001\x00\x01\x00@\x1f\x00\x00Y\x06\x00\x00A"
    b"\x00\x00\x00\x02\x00@\x01fact\x04\x00\x00\x00\x00\x00\x00\x00data\x00\x00\x00\x00"
)


def _fill_missing_time_labels(replicas: List[ReplicaSchema]) -> List[ReplicaSchema]:
    replicas = deepcopy(replicas)

    if replicas:
        eps = 0.001
        if replicas[0].start < 0:  # pylint:disable=consider-using-max-builtin
            replicas[0].start = 0
        if replicas[0].end < 0:
            replicas[0].end = replicas[0].start + eps
        for prev_replica, replica in window(replicas, 2):
            if replica.start < 0:
                replica.start = prev_replica.end + eps
            if replica.end < 0:
                replica.end = replica.start + eps
    return replicas


def _round_time_labels(replica: ReplicaSchema) -> ReplicaSchema:
    return ReplicaSchema(
        text=replica.text,
        start=round(replica.start, 3),
        end=round(replica.end, 3),
    )


def process(audio_wav: Path) -> DialogSchema:
    """Конвертировать аудио в диалог"""
    model, model_a, metadata = get_model()

    with measure_performance() as measure:
        audio_in = whisperx.load_audio(str(audio_wav))
        segments_in = model.transcribe(
            audio_in,
            language="ru",
            task="transcribe",
        )
        for entry in segments_in["segments"]:
            entry["text"] = clean_text(entry["text"])
        
        segments_in = whisperx.align(
            segments_in["segments"],
            model_a,
            metadata,
            audio_in,
            device=DEVICE,
            return_char_alignments=False,
        )
        text_in = segments_in["word_segments"]
        dialog = [
            ReplicaSchema(
                text=word["word"],
                start=word["start"] if "start" in word else -1,
                end=(word["end"] + word["start"]) / 2 if "start" in word else -1,
            )
            for word in text_in
        ]
        dialog = _fill_missing_time_labels(dialog)
        # Округление необходимо для уменьшения кол-во получаемого JSON
        dialog = [_round_time_labels(replica) for replica in dialog]
    print("Converted for %.2fs.", measure.delta)

    return DialogSchema(dialog=dialog)


dialog = process(audio_wav=Path('audio_result (mp3cut.net).wav'))
for line in dialog.dialog:
    print(f'{line.start} - {line.end}: {line.text}')
pretty = prettify_dialog(merge_dialogs({'1': dialog}))
for line in pretty.dialog:
    print(f'{line.start} - {line.end} ({line.channel}): {line.text}')