import os
import gradio as gr
import numpy as np
import torch
import torchaudio
from typing import Tuple, Optional
import soundfile as sf
from s2st_inference import s2st_inference

SAMPLE_RATE = 16000
MAX_INPUT_LENGTH = 60  # seconds

S2UT_TAG = 'espnet/jiyang_tang_cvss-c_es-en_discrete_unit'
S2UT_DIR = 'model'
VOCODER_TAG = 'espnet/cvss-c_en_wavegan_hubert_vocoder'
VOCODER_DIR = 'vocoder'


def download_model(tag: str, out_dir: str):
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=tag, local_dir=out_dir)


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

    # Download models
    os.makedirs(S2UT_DIR, exist_ok=True)
    os.makedirs(VOCODER_DIR, exist_ok=True)
    s2ut_path = download_model(S2UT_TAG, S2UT_DIR)
    vocoder_path = download_model(VOCODER_TAG, VOCODER_DIR)

    # Temporary change cwd to model dir so that it loads correctly
    cwd = os.getcwd()
    os.chdir(s2ut_path)

    # Translate wav
    out_wav = s2st_inference(
        wav,
        train_config=os.path.join(
            s2ut_path,
            'exp',
            's2st_train_s2st_discrete_unit_raw_fbank_es_en',
            'config.yaml',
        ),
        model_file=os.path.join(
            s2ut_path,
            'exp',
            's2st_train_s2st_discrete_unit_raw_fbank_es_en',
            '500epoch.pth',
        ),
        vocoder_file=os.path.join(
            vocoder_path,
            'checkpoint-450000steps.pkl',
        ),
        vocoder_config=os.path.join(
            vocoder_path,
            'config.yml',
        ),
    )

    # Restore working directory
    os.chdir(cwd)

    # Save result
    output_path = 'output.wav'
    sf.write(
        output_path,
        out_wav,
        16000,
        "PCM_16",
    )

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
