---
license: cc-by-4.0
base_model: moondream/parakeet-redux
pipeline_tag: automatic-speech-recognition
language: [en, de, fr, es, it, pt, ru, uk, hr, sl, lv, lt, et, fi, sv, da, nl, pl, cs, sk, hu, ro, bg, el, mt]
tags: [onnx, onnxruntime, parakeet, tdt, ternary, speech-recognition, cpu]
---

# Parakeet Redux (ONNX)

ONNX export of [moondream/parakeet-redux](https://huggingface.co/moondream/parakeet-redux),
the 1.58-bit version of NVIDIA's
[parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), for
inference with ONNX Runtime and no PyTorch. Same architecture, same tokenizer,
same 25 languages. The ternary encoder weights are stored as 4-bit
`MatMulNBits` blocks that reproduce the -1/0/+1 values exactly, so this is the
same model as the source, not a re-quantization of it.

## Files

| File | Role | Size |
| ---- | ---- | ---- |
| `preprocessor.onnx` | 16 kHz waveform → 128-band log-mel features (the recipe Photon uses) | 1.2 MB |
| `encoder-model.onnx` | features → encoder frames, one every 80 ms | 344 MB |
| `decoder_joint-model.onnx` | one prediction-network + joint step of the token-and-duration transducer | 73 MB |
| `vad-model.onnx` | the checkpoint's own voice-activity head, for cutting long audio at pauses | 18 MB |
| `vocab.txt`, `config.json` | tokens and settings, in the layout [onnx-asr](https://github.com/istupakov/onnx-asr) expects | |

Tensor names follow NeMo's exports:

| Model | Inputs | Outputs |
| ----- | ------ | ------- |
| preprocessor | `waveforms` [B, samples] float32, `waveforms_lens` [B] int64 | `features` [B, 128, T], `features_lens` [B] |
| encoder | `audio_signal` [B, 128, T], `length` [B] int64 | `outputs` [B, 1024, T/8], `encoded_lengths` [B] |
| decoder_joint | `encoder_outputs` [B, 1024, 1], `targets` [B, 1] int32, `target_length` [B] int32, `input_states_1`, `input_states_2` [2, B, 640] | `outputs` [B, 1, 1, 8198] (8193 token logits then 5 duration logits), `prednet_lengths`, `output_states_1`, `output_states_2` |
| vad | `audio_signal` [B, 128, T], `length` [B] int64 | `probabilities` [B, T/8], `encoded_lengths` [B] |

**Inference pipeline:** preprocessor → encoder → greedy loop over decoder_joint (durations 0-4 frames per step, blank id 8192). Recordings over 30 s are cut at pauses found by the VAD head.

## Setup

```bash
pip install onnxruntime numpy
```

`ffmpeg` on the PATH lets `transcribe.py` read anything other than 16 kHz wav.

## Transcribe

```bash
python transcribe.py speech.wav
python transcribe.py speech.wav --timestamps segment    # one line per sentence
python transcribe.py speech.wav --timestamps word
python transcribe.py talk.mp3 --json
```

`transcribe.py` mirrors Photon's pipeline: its features, its greedy decode
loop, its word timing and its pause-based segmentation. On clean clips it
produces the same text and word timestamps as Photon.

With onnx-asr, which brings its own NeMo-style feature extractor:

```python
import onnx_asr

model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", "path/to/parakeet-redux-onnx")
print(model.recognize("speech.wav"))
```

## Export

`export_onnx.py` regenerates these files from the source weights. It needs the
Photon runtime, whose PyTorch modules it traces:

```bash
pip install moondream onnx onnxscript onnxruntime
python export_onnx.py                                  # downloads moondream/parakeet-redux
python export_onnx.py --model-path ../moondream/parakeet-redux
python export_onnx.py --bits 2                         # 193 MB encoder, same numbers
python export_onnx.py --bits 0                         # dense float32, plain ONNX, 2.4 GB
python export_onnx.py --accuracy-level 4               # int8 activations, about 2x faster
```

The encoder is exported dense in float32, then every ternary linear layer is
rewritten into ONNX Runtime `MatMulNBits` blocks of 128 weights with the
source's float16 group scales. The script checks every graph against the
PyTorch reference before finishing.

## Accuracy and speed

Checked against the PyTorch float32 reference: encoder outputs agree to
2e-7, the VAD head to 1e-6, and decoder-joint logits to float32 rounding.
End to end, `transcribe.py` matched Photon word for word, timestamps included,
on clean clips and on a four-minute phone call run through Photon's Metal path.

CPU, Apple M2 Max, one utterance at a time:

| Setting | Encoder math | Real time |
| ------- | ------------ | --------- |
| default (`accuracy_level` 0) | float32 over unpacked blocks | ~25x |
| `--accuracy-level 4` | int8 activations | ~50x, changed a few words on test clips |
| Photon, same machine | int8 CPU kernels / Metal | 58x / 124x |

Run `transcribe.py --threads N` with N set to the number of performance cores;
ONNX Runtime otherwise also schedules the efficiency cores.

## Notes

- `MatMulNBits` is an ONNX Runtime contrib operator (`com.microsoft`
  domain). Export with `--bits 0` for a plain-ONNX float32 encoder.
- The model detects the language itself. Forcing a language, translation and
  text prompts are not supported by Parakeet TDT.
- Exported from the source repository at commit `2bf1286` with torch 2.14,
  onnx 1.23 and onnxruntime 1.30.

## License and attribution

This is a converted copy of moondream's
[parakeet-redux](https://huggingface.co/moondream/parakeet-redux), which is
itself derived from NVIDIA's
[parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).
Both are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), and so is this
export. Changes from the source: the weights were converted to ONNX graphs,
the ternary encoder weights were repacked as 4-bit `MatMulNBits` blocks, and the
feature extractor and VAD head were exported as their own graphs. The
conversion and inference scripts are new. Neither moondream nor NVIDIA endorse
this export.
