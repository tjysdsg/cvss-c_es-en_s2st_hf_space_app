import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import soundfile as sf
import torch
from typeguard import check_argument_types
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.bin.s2st_inference import Speech2Speech


def s2st_inference(
        speech: torch.Tensor,
        ngpu: int = 0,
        seed: int = 2023,
        log_level: Union[int, str] = 'INFO',
        train_config: Optional[str] = None,
        model_file: Optional[str] = None,
        threshold: float = 0.5,
        minlenratio: float = 0,
        maxlenratio: float = 10.0,
        st_subtask_minlenratio: float = 0,
        st_subtask_maxlenratio: float = 1.5,
        use_teacher_forcing: bool = False,
        use_att_constraint: bool = False,
        backward_window: int = 1,
        forward_window: int = 3,
        always_fix_seed: bool = False,
        beam_size: int = 5,
        penalty: float = 0,
        st_subtask_beam_size: int = 5,
        st_subtask_penalty: float = 0,
        st_subtask_token_type: Optional[str] = None,
        st_subtask_bpemodel: Optional[str] = None,
        vocoder_config: Optional[str] = None,
        vocoder_file: Optional[str] = None,
        vocoder_tag: Optional[str] = None,
):
    """Run text-to-speech inference."""
    assert check_argument_types()
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build model
    speech2speech_kwargs = dict(
        train_config=train_config,
        model_file=model_file,
        threshold=threshold,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        st_subtask_maxlenratio=st_subtask_maxlenratio,
        st_subtask_minlenratio=st_subtask_minlenratio,
        use_teacher_forcing=use_teacher_forcing,
        use_att_constraint=use_att_constraint,
        backward_window=backward_window,
        forward_window=forward_window,
        beam_size=beam_size,
        penalty=penalty,
        st_subtask_beam_size=st_subtask_beam_size,
        st_subtask_penalty=st_subtask_penalty,
        st_subtask_token_type=st_subtask_token_type,
        st_subtask_bpemodel=st_subtask_bpemodel,
        vocoder_config=vocoder_config,
        vocoder_file=vocoder_file,
        device=device,
        seed=seed,
        always_fix_seed=always_fix_seed,
    )
    speech2speech = Speech2Speech.from_pretrained(
        vocoder_tag=vocoder_tag,
        **speech2speech_kwargs,
    )

    start_time = time.perf_counter()

    speech_lengths = torch.as_tensor([speech.shape[0]])
    output_dict = speech2speech(speech.unsqueeze(0), speech_lengths)

    insize = speech.size(0) + 1
    # standard speech2mel model case
    feat_gen = output_dict["feat_gen"]
    logging.info(
        f"inference speed = {int(feat_gen.size(0)) / (time.perf_counter() - start_time):.1f} frames / sec."
    )
    logging.info(f"(size:{insize}->{feat_gen.size(0)})")
    if feat_gen.size(0) == insize * maxlenratio:
        logging.warning(f"output length reaches maximum length.")

    feat_gen = output_dict["feat_gen"].cpu().numpy()
    if output_dict.get("feat_gen_denorm") is not None:
        feat_gen_denorm = output_dict["feat_gen_denorm"].cpu().numpy()

    assert 'wav' in output_dict
    wav = output_dict["wav"].cpu().numpy()
    logging.info(f"wav {len(wav)}")

    return wav

    # if output_dict.get("st_subtask_token") is not None:
    #     writer["token"][key] = " ".join(output_dict["st_subtask_token"])
    #     writer["token_int"][key] == " ".join(
    #         map(str, output_dict["st_subtask_token_int"])
    #     )
    #     if output_dict.get("st_subtask_text") is not None:
    #         writer["text"][key] = output_dict["st_subtask_text"]
