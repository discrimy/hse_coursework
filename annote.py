from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.0",
    use_auth_token="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",
)

diarization = pipeline("audio_result (mp3cut.net).wav")
for segment, _, speaker in diarization.itertracks(yield_label=True):
    print(f'Speaker "{speaker}" - "{segment}"')
