import base64
import itertools
from typing import Dict, Optional, List, Iterator

import nltk
from pydantic import BaseModel

from hse_coursework.schemas import DialogSchema


class ReplicaWithChannelSchema(BaseModel):
    """Реплика с привязкой к каналу, по которому её сказали"""

    text: str
    start: float
    end: float
    channel: str


class DialogWithChannelsSchema(BaseModel):
    """Диалог с репликами с привязкой к каналу"""

    dialog: List[ReplicaWithChannelSchema]


def merge_dialogs(
    channel_to_dialog: Dict[str, DialogSchema],
) -> DialogWithChannelsSchema:
    """Склеить диалоги в один диалог с привязкой по времени. Диалог будет отсортировать по времени начала реплики"""
    replicas_with_channels = [
        ReplicaWithChannelSchema(**replica.model_dump(), channel=channel)
        for channel, dialog in channel_to_dialog.items()
        for replica in dialog.dialog
    ]
    return DialogWithChannelsSchema(
        dialog=sorted(
            replicas_with_channels,
            key=lambda named_dialog: named_dialog.start,
        ),
    )


def dialog_to_text(replicas: List[ReplicaWithChannelSchema]) -> str:
    """Преобразует склеенный диалог в сплошной текст (нужен для распознавания записи)"""
    return " ".join(replica.text for replica in replicas)


def _group_replicas_by_channel(
    replicas: List[ReplicaWithChannelSchema],
) -> Iterator[ReplicaWithChannelSchema]:
    # itertools.groupby группирует **соседние** элементы с совпадающим ключом, что нам и надо
    # пример: 1 1 1 2 1 1 2 2 -> [1, 1, 1] [2] [1, 1] [2, 2]
    for channel, replicas_group_iter in itertools.groupby(
        replicas, key=lambda r: r.channel
    ):
        replicas_group = list(replicas_group_iter)
        yield ReplicaWithChannelSchema(
            text=dialog_to_text(replicas_group),
            start=replicas_group[0].start,
            end=replicas_group[-1].end,
            channel=channel,
        )


def _group_replicas_within_channel_by_sentences(
    replicas: List[ReplicaWithChannelSchema],
) -> Iterator[ReplicaWithChannelSchema]:
    text = dialog_to_text(replicas)
    sentences = nltk.tokenize.sent_tokenize(text, language="russian")

    buffer = []
    sentence_index = 0
    for replica in replicas:
        buffer.append(replica)
        buffer_text = dialog_to_text(buffer)
        if sentences[sentence_index] == buffer_text:
            yield ReplicaWithChannelSchema(
                text=buffer_text,
                start=buffer[0].start,
                end=buffer[-1].end,
                channel=buffer[0].channel,
            )
            buffer = []
            sentence_index += 1
    if buffer:
        buffer_text = dialog_to_text(buffer)
        yield ReplicaWithChannelSchema(
            text=buffer_text,
            start=buffer[0].start,
            end=buffer[-1].end,
            channel=buffer[0].channel,
        )


def _prettify_replicas(
    replicas: List[ReplicaWithChannelSchema],
) -> List[ReplicaWithChannelSchema]:
    # Объединяем слова в отдельные предложения
    channels = {r.channel for r in replicas}
    grouped_by_sentences = [
        grouped_replica
        for channel in channels
        for grouped_replica in _group_replicas_within_channel_by_sentences(
            [replica for replica in replicas if replica.channel == channel],
        )
    ]
    # Объединяем соседние предложения от одного канала
    grouped_by_sentences = sorted(
        grouped_by_sentences, key=lambda replica: replica.start
    )
    # grouped_by_channel = list(_group_replicas_by_channel(grouped_by_sentences))
    grouped_by_channel = grouped_by_sentences
    return grouped_by_channel


def prettify_dialog(dialog: DialogWithChannelsSchema) -> DialogWithChannelsSchema:
    """Преобразует диалог в человекочитаемый вид, объединяя слова в полноценные реплики"""
    return DialogWithChannelsSchema(dialog=_prettify_replicas(dialog.dialog))
