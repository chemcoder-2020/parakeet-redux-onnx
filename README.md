# parakeet-redux-onnx

**Mirror of [`eschmidbauer/parakeet-redux-onnx`](https://huggingface.co/eschmidbauer/parakeet-redux-onnx)** (Hugging Face).

ONNX export of **[moondream/parakeet-redux](https://huggingface.co/moondream/parakeet-redux)** —
the 1.58-bit (ternary) version of NVIDIA's `parakeet-tdt-0.6b-v3` — for inference with
**ONNX Runtime and no PyTorch**. Same architecture, same tokenizer, same 25 languages.
The ternary encoder weights are stored as 4-bit `MatMulNBits` blocks that reproduce the
−1/0/+1 values exactly, so this is the same model as the source, not a re-quantization of it.

⚠️ The export is **by [eschmidbauer](https://huggingface.co/eschmidbauer)** — this GitHub
repo is a byte-identical mirror with the checksums, license files, and a load test added.
The original model card is preserved verbatim in [SOURCE-CARD.md](SOURCE-CARD.md).

Ships the graphs **and** the code:

- `preprocessor.onnx`, `encoder-model.onnx`, `decoder_joint-model.onnx`, `vad-model.onnx` —
  the inference graphs (mel frontend → encoder → greedy TDT loop; VAD head for cutting long audio at pauses)
- `transcribe.py` — CPU CLI mirroring Photon's pipeline (its features, greedy decode loop,
  word timing, pause-based segmentation)
- `export_onnx.py` — regenerates the export from the source weights (needs the Photon runtime)
- `vocab.txt`, `config.json` — tokens and settings, in the layout `onnx-asr` expects

## Files

| file | bytes | sha256 |
|---|---|---|
| `preprocessor.onnx` | 1224294 | `3de5742f5780fc8615c1d883a2a60e28621ff13d2e20d24d76e94b9e4168445f` |
| `encoder-model.onnx` | 343841943 | `ade2c65c188776b3662db620a553ab82ac7e9ca11a7be6b696bcc5190f75382e` |
| `decoder_joint-model.onnx` | 72552270 | `97ae0cb049f32a5d7a6572e91020b8d87d66baffa0ff9de2cf6de4c0960e92f8` |
| `vad-model.onnx` | 18300687 | `18dfbe344307f9eda4735d4da1fe1cd6e3be27055a6c25b4a4a2e6fe0bffde50` |

The four sha256 values above match the checksums published by the source repo on the
Hugging Face Hub (verified again after download). [checksums.txt](checksums.txt) covers
every mirrored file — re-verify a clone with:

```bash
git lfs pull
shasum -a 256 -c checksums.txt
```

## Usage

```bash
pip install onnxruntime numpy        # + ffmpeg on the PATH for non-16-kHz-wav input
python transcribe.py speech.wav
python transcribe.py speech.wav --timestamps word
python transcribe.py talk.mp3 --json
```

## Verified on this mirror (2026-10-08)

- sha256 of all four ONNX graphs == the source repo's published checksums (table above) —
  byte-identical copy; every other mirrored file is covered by `checksums.txt`.
- Ran `transcribe.py` for real: onnxruntime 1.30.0, macOS arm64 (Apple M1), `--threads 4`:
  - `bcn_weather.wav` (11.0 s) → *"Yesterday it was 35 degrees in Barcelona, but today the
    temperature will go down to minus 20 degrees."* — 12× realtime
  - `en-Alice_woman.wav` (9.3 s) → *"So just to clarify, we've had 19 to 20 year olds on
    our podcast so far, and if you don't mind me asking, how old are you?"* — 13× realtime
- These are short-clip smoke numbers on an M1. The ~25× figure on the card is from an
  Apple M2 Max; this is not a re-run of the exporter's parity testing.

## License & attribution

**CC BY 4.0** — give credit to:

- **NVIDIA Corporation** — `parakeet-tdt-0.6b-v3` base model (CC BY 4.0)
- **Moondream** — `parakeet-redux` ternary retrain (CC BY 4.0)
- **eschmidbauer** — ONNX export, `transcribe.py`, `export_onnx.py` (CC BY 4.0)

As stated on the upstream card, neither moondream nor NVIDIA endorse this export.
See [LICENSE](LICENSE) (verbatim CC BY 4.0) and [NOTICE](NOTICE). Source of truth:
https://huggingface.co/eschmidbauer/parakeet-redux-onnx
