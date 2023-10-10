import os
import gradio as gr
import numpy as np
import torch
import torchaudio
from typing import Tuple, Optional

SAMPLE_RATE = 16000
MAX_INPUT_LENGTH = 60  # seconds


def s2st(
        audio_source: str,
        input_audio_mic: Optional[str],
        input_audio_file: Optional[str],
):
    if audio_source == 'file':
        input_path = input_audio_file
    else:
        input_path = input_audio_mic

    if input_path is None:
        gr.Error(f"Input audio is too long. Truncated to {MAX_INPUT_LENGTH} seconds.")
        return (None, None), None

    orig_wav, orig_sr = torchaudio.load(input_path)
    wav = torchaudio.functional.resample(orig_wav, orig_freq=orig_sr, new_freq=SAMPLE_RATE)
    max_length = int(MAX_INPUT_LENGTH * SAMPLE_RATE)
    if wav.shape[1] > max_length:
        wav = wav[:, :max_length]
        gr.Warning(f"Input audio is too long. Truncated to {MAX_INPUT_LENGTH} seconds.")

    wav = wav[0]  # mono

    # TODO: translate wav
    output_path = 'output.wav'
    torchaudio.save(output_path, wav.unsqueeze(0), SAMPLE_RATE)

    return output_path, f'Source: {audio_source}'


def update_audio_ui(audio_source: str) -> Tuple[dict, dict]:
    mic = audio_source == "microphone"
    return (
        gr.update(visible=mic, value=None),  # input_audio_mic
        gr.update(visible=not mic, value=None),  # input_audio_file
    )


def main():
    with gr.Blocks() as demo:
        with gr.Group():
            with gr.Row() as audio_box:
                audio_source = gr.Radio(
                    label="Audio source",
                    choices=["file", "microphone"],
                    value="file",
                )
                input_audio_mic = gr.Audio(
                    label="Input speech",
                    type="filepath",
                    source="microphone",
                    visible=False,
                )
                input_audio_file = gr.Audio(
                    label="Input speech",
                    type="filepath",
                    source="upload",
                    visible=True,
                )

            btn = gr.Button("Translate")

            with gr.Column():
                output_audio = gr.Audio(
                    label="Translated speech",
                    autoplay=False,
                    streaming=False,
                    type="numpy",
                )
                output_text = gr.Textbox(label="Translated text")

        audio_source.change(
            fn=update_audio_ui,
            inputs=audio_source,
            outputs=[
                input_audio_mic,
                input_audio_file,
            ],
            queue=False,
            api_name=False,
        )

        btn.click(
            fn=s2st,
            inputs=[
                audio_source,
                input_audio_mic,
                input_audio_file,
            ],
            outputs=[output_audio, output_text],
            api_name="run",
        )

        demo.queue(max_size=50).launch()


if __name__ == '__main__':
    main()
