import os
import gradio as gr
import torchaudio
from typing import Tuple, Optional
import soundfile as sf
from s2st_inference import s2st_inference
from utils import download_model

DESCRIPTION = r"""
**Speech-to-Speech Translation from Spanish to English**

- Paper: Direct Speech-to-Speech Translation With Discrete Units
- Dataset: CVSS-C
- Toolkit: [ESPnet](https://github.com/espnet/espnet)
- Pretrained Speech-to-Unit translation model: https://huggingface.co/espnet/jiyang_tang_cvss-c_es-en_discrete_unit
- Pretrained WaveGAN vocoder: https://huggingface.co/espnet/cvss-c_en_wavegan_hubert_vocoder

Part of a CMU MIIS capstone project with [@realzza](https://github.com/realzza)
and [@sophia1488](https://github.com/sophia1488)
"""

SAMPLE_RATE = 16000
MAX_INPUT_LENGTH = 60  # seconds

S2UT_TAG = 'espnet/jiyang_tang_cvss-c_es-en_discrete_unit'
S2UT_DIR = 'model'
VOCODER_TAG = 'espnet/cvss-c_en_wavegan_hubert_vocoder'
VOCODER_DIR = 'vocoder'

NGPU = 0
BEAM_SIZE = 1


class App:
    def __init__(self):
        # Download models
        os.makedirs(S2UT_DIR, exist_ok=True)
        os.makedirs(VOCODER_DIR, exist_ok=True)

        self.s2ut_path = download_model(S2UT_TAG, S2UT_DIR)
        self.vocoder_path = download_model(VOCODER_TAG, VOCODER_DIR)

    def s2st(
            self,
            input_audio: Optional[str],
    ):
        orig_wav, orig_sr = torchaudio.load(input_audio)
        wav = torchaudio.functional.resample(orig_wav, orig_freq=orig_sr, new_freq=SAMPLE_RATE)
        max_length = int(MAX_INPUT_LENGTH * SAMPLE_RATE)
        if wav.shape[1] > max_length:
            wav = wav[:, :max_length]
            gr.Warning(f"Input audio is too long. Truncated to {MAX_INPUT_LENGTH} seconds.")

        wav = wav[0]  # mono

        # Temporary change cwd to model dir so that it loads correctly
        cwd = os.getcwd()
        os.chdir(self.s2ut_path)

        # Translate wav
        out_wav = s2st_inference(
            wav,
            train_config=os.path.join(
                self.s2ut_path,
                'exp',
                's2st_train_s2st_discrete_unit_raw_fbank_es_en',
                'config.yaml',
            ),
            model_file=os.path.join(
                self.s2ut_path,
                'exp',
                's2st_train_s2st_discrete_unit_raw_fbank_es_en',
                '500epoch.pth',
            ),
            vocoder_file=os.path.join(
                self.vocoder_path,
                'checkpoint-450000steps.pkl',
            ),
            vocoder_config=os.path.join(
                self.vocoder_path,
                'config.yml',
            ),
            ngpu=NGPU,
            beam_size=BEAM_SIZE,
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

        return output_path


def main():
    app = App()

    with gr.Blocks() as demo:
        gr.Markdown(DESCRIPTION)
        with gr.Group():
            input_audio = gr.Audio(
                label="Input speech",
                type="filepath",
                sources=["upload", "microphone"],
                format='wav',
                streaming=False,
                visible=True,
            )

            btn = gr.Button("Translate")

            output_audio = gr.Audio(
                label="Translated speech",
                autoplay=False,
                streaming=False,
                type="numpy",
            )

            # Placeholders so that the example section can show these values
            source_text = gr.Text(label='Source Text', visible=False)
            target_text = gr.Text(label='Target Text', visible=False)

        # Examples
        with gr.Row():
            gr.Examples(
                examples=[
                    ["examples/example1.wav",
                     "fue enterrada en el cementerio forest lawn memorial park de hollywood hills",
                     "she was buried at the forest lawn memorial park of hollywood hills"],
                    ["examples/example2.wav",
                     "diversos otros músicos han interpretado esta canción en conciertos en vivo",
                     "many other musicians have played this song in live concerts"],
                    ["examples/example3.wav",
                     "es gómez-moreno el primero en situar su origen en guadalajara, hoy ampliamente aceptado",
                     "gomez moreno was the first person to place its origin in guadalajara which is now broadly accepted"],
                ],
                inputs=[input_audio, source_text, target_text],
                outputs=[output_audio],
            )

        btn.click(
            fn=app.s2st,
            inputs=[input_audio],
            outputs=[output_audio],
            api_name="run",
        )

        demo.queue(max_size=50).launch()


if __name__ == '__main__':
    main()
