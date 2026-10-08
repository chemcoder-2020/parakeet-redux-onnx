#!/usr/bin/env python3
"""Export moondream/parakeet-redux to ONNX.

Writes a directory that ONNX Runtime and NeMo-style ASR runtimes such as
onnx-asr can load:

    encoder-model.onnx        mel features -> encoder frames (NeMo layout)
    decoder_joint-model.onnx  prediction network + joint, one step
    preprocessor.onnx         waveform -> log-mel features, Photon's recipe
    vad-model.onnx            the checkpoint's voice-activity head
    vocab.txt, config.json    tokens and settings for onnx-asr

The ternary encoder weights are stored as ONNX Runtime ``MatMulNBits`` blocks
(4-bit by default, or 2-bit) that reproduce the -1/0/+1 values exactly, so the
export is numerically the same model as the dequantized float32 one.

The source weights come from moondream/parakeet-redux on the Hugging Face Hub
(or a local clone via --model-path) and need the Photon runtime installed:
``pip install moondream`` pulls in PyTorch and the Kestrel model code.

Examples:
    python export_onnx.py                              # writes the files next to this script
    python export_onnx.py --model-path ../moondream/parakeet-redux
    python export_onnx.py --bits 2 --out ./2bit
    python export_onnx.py --bits 0                     # dense float32, external data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
SOURCE_MODEL = "moondream/parakeet-redux"
SOURCE_FILES = ("config.json", "tokenizer.json", "model.safetensors", "ternary.json")
OPSET = 18
BLOCK = 128
FILES = {
    "encoder": "encoder-model.onnx",
    "decoder_joint": "decoder_joint-model.onnx",
    "preprocessor": "preprocessor.onnx",
    "vad": "vad-model.onnx",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split("Examples:")[1])
    parser.add_argument("--model-path", default=SOURCE_MODEL,
                        help=f"local clone of the source model, or a Hub id to download (default: {SOURCE_MODEL})")
    parser.add_argument("--out", default=str(HERE), help="output directory (default: next to this script)")
    parser.add_argument("--bits", type=int, choices=(0, 2, 4), default=4,
                        help="ternary weight storage: 4 (default) or 2 bits per weight via MatMulNBits, 0 for dense float32")
    parser.add_argument("--accuracy-level", type=int, choices=(0, 1, 2, 3, 4), default=0,
                        help="MatMulNBits accuracy_level: 0/1 float32 math (exact), 4 int8 activations (faster)")
    parser.add_argument("--keep-dense", action="store_true", help="keep the float32 encoder next to the packed one")
    parser.add_argument("--no-verify", action="store_true", help="skip the numerical checks against PyTorch")
    return parser.parse_args(argv)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def resolve_source(value: str) -> Path:
    """A local directory with the source files, or a Hub snapshot of them."""
    path = Path(value).expanduser()
    if path.is_dir():
        return path.resolve()
    from huggingface_hub import snapshot_download

    log(f"downloading {value} from the Hugging Face Hub...")
    return Path(snapshot_download(value, allow_patterns=list(SOURCE_FILES)))


# ----------------------------------------------------------------------------
# The float32 model: Photon's PyTorch modules with the fused kernels swapped for
# their torch references and every ternary layer expanded to a dense nn.Linear.
# ----------------------------------------------------------------------------

def load_dense(root: Path):
    from safetensors.torch import load_file
    import kestrel.models.parakeet_tdt.model as parakeet_model
    from kestrel.models.parakeet_tdt.config import ParakeetTdtConfig
    from kestrel.models.parakeet_tdt.model import ParakeetTdt
    from kestrel.models.parakeet_tdt.vad import VAD_HEAD_PREFIX
    from kestrel.models.parakeet_tdt.weights import _quantized_modules, ternarize, unpack_export
    from kestrel_kernels.conformer_ops import torch_conformer_runtime
    from kestrel_kernels.tdt_ops import torch_tdt_runtime
    from kestrel_kernels.ternary import TernaryLinear, TernaryWeight, unpack_codes

    reference = types.SimpleNamespace(conformer=torch_conformer_runtime("conv1d"), tdt=torch_tdt_runtime())
    parakeet_model.get_runtime = lambda device=None: reference

    config = ParakeetTdtConfig.from_json_file(root / "config.json")
    state = load_file(str(root / "model.safetensors"), device="cpu")
    quantized = _quantized_modules(root / "ternary.json")
    with torch.device("meta"):
        model = ParakeetTdt(config)
        ternarize(model, quantized)
        model.is_ternary = True
        if any(key.startswith(VAD_HEAD_PREFIX) for key in state):
            model.attach_vad_head()
    for entry in quantized:
        key = f"{entry['name']}.qweight"
        state[key] = unpack_export(state[key], entry["in_features"])
    model.load_state_dict(state, strict=True, assign=True)
    model.reset_nonpersistent_buffers()

    ternary: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, module in list(model.named_modules()):
        if not isinstance(module, TernaryLinear):
            continue
        codes = unpack_codes(module.qweight, module.in_features).numpy()  # int8 [out, in] in {-1, 0, 1}
        scales = module.scales.clone().numpy()  # float16 [out, in / 128]
        dense = TernaryWeight(module.qweight, module.scales, module.in_features).dequantized(torch.float32)
        linear = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None)
        with torch.no_grad():
            linear.weight.copy_(dense)
            if module.bias is not None:
                linear.bias.copy_(module.bias)
        model.set_submodule(name, linear)
        ternary[name] = (codes, scales)
    if model.vad_head is None:
        raise SystemExit("error: this checkpoint has no VAD head; the export expects moondream/parakeet-redux")
    return config, model.to(torch.float32).eval(), ternary


# ----------------------------------------------------------------------------
# Export wrappers. Names and layouts follow NeMo's own exports so existing
# runtimes (onnx-asr, sherpa-style loaders) can consume them.
# ----------------------------------------------------------------------------

class EncoderExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder

    def forward(self, audio_signal, length):
        features = audio_signal.transpose(1, 2)  # [B, T, 128]
        mask = torch.arange(features.shape[1], device=features.device)[None] < length[:, None]
        encoded, valid = self.encoder(features, mask)
        return encoded.transpose(1, 2), valid.sum(-1)  # [B, 1024, T'], [B]


class DecoderJointExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder_projector = model.encoder_projector
        self.decoder = model.decoder
        self.joint = model.joint

    def forward(self, encoder_outputs, targets, target_length, input_states_1, input_states_2):
        frame = self.encoder_projector(encoder_outputs.transpose(1, 2))  # [B, 1, 640]
        pred, (hidden, cell) = self.decoder(targets.to(torch.int64), (input_states_1, input_states_2))
        logits = self.joint(frame, pred)  # [B, 1, vocab + durations]
        return logits.unsqueeze(2), target_length, hidden, cell


class VadExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.subsampling = model.encoder.subsampling
        self.vad_head = model.vad_head

    def forward(self, audio_signal, length):
        features = audio_signal.transpose(1, 2)
        mask = torch.arange(features.shape[1], device=features.device)[None] < length[:, None]
        hidden, valid = self.subsampling(features, mask)
        return torch.sigmoid(self.vad_head(hidden)), valid.sum(-1)


class Preprocessor(nn.Module):
    """Photon's Parakeet features: pre-emphasis, 512-point STFT (hann 400, hop 160,
    zero padding), 128 Slaney mel bands, log, per-feature normalization over the
    valid frames. The STFT is a strided convolution with a DFT basis so it exports
    as plain ONNX ops."""

    def __init__(self):
        super().__init__()
        from kestrel.models.asr.features import mel_filters

        n_fft, win, hop = 512, 400, 160
        window = F.pad(torch.hann_window(win, periodic=False), ((n_fft - win) // 2, (n_fft - win) // 2))
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float64)[:, None]
        n = torch.arange(n_fft, dtype=torch.float64)[None]
        angle = 2 * torch.pi * k * n / n_fft
        basis = torch.cat([torch.cos(angle), -torch.sin(angle)]) * window.double()[None]
        self.register_buffer("basis", basis.float()[:, None, :])  # [514, 1, 512]
        self.register_buffer("mel", mel_filters(n_fft, 128, 16_000).clone())  # [128, 257]
        self.hop = hop
        self.pad = n_fft // 2

    def forward(self, waveforms, waveforms_lens):
        x = waveforms
        emphasised = torch.cat((x[:, :1], x[:, 1:] - 0.97 * x[:, :-1]), dim=1)
        inside = torch.arange(x.shape[1], device=x.device)[None] < waveforms_lens[:, None]
        emphasised = torch.where(inside, emphasised, torch.zeros((), dtype=x.dtype))
        spectrum = F.conv1d(F.pad(emphasised, (self.pad, self.pad))[:, None], self.basis, stride=self.hop)
        bins = self.basis.shape[0] // 2
        power = spectrum[:, :bins] ** 2 + spectrum[:, bins:] ** 2
        logmel = torch.log(self.mel @ power + 2.0 ** -24)  # [B, 128, frames]
        valid = waveforms_lens // self.hop
        frame_mask = (torch.arange(logmel.shape[2], device=x.device)[None] < valid[:, None])[:, None].to(logmel.dtype)
        count = valid.to(logmel.dtype)[:, None, None]
        mean = (logmel * frame_mask).sum(2, keepdim=True) / count
        variance = (((logmel - mean) * frame_mask) ** 2).sum(2, keepdim=True) / (count - 1)
        features = ((logmel - mean) / (torch.sqrt(variance) + 1e-5)) * frame_mask
        return features, valid


def export(module, args, names, outputs, dynamic, path: Path, external: bool = False):
    from torch.export import Dim

    dims: dict[str, Dim] = {}

    def dim(name):
        if name not in dims:
            dims[name] = Dim(name, min=16) if name in ("time", "samples") else Dim(name)
        return dims[name]

    shapes = {name: {axis: dim(label) for axis, label in spec.items()} for name, spec in dynamic.items()}
    program = torch.onnx.export(module, args, dynamo=True, opset_version=OPSET, input_names=names,
                                output_names=outputs, dynamic_shapes=shapes, external_data=external)
    program.save(str(path), external_data=external)
    return program


# ----------------------------------------------------------------------------
# Ternary packing: replace each MatMul over a ternary weight with MatMulNBits.
# ----------------------------------------------------------------------------

def pack_nbits(codes_u8: np.ndarray, bits: int, block: int) -> np.ndarray:
    rows, cols = codes_u8.shape
    blocks = -(-cols // block)
    padded = np.zeros((rows, blocks * block), dtype=np.uint8)
    padded[:, :cols] = codes_u8
    per_byte = 8 // bits
    grouped = padded.reshape(rows, blocks, block // per_byte, per_byte)
    shifts = (np.arange(per_byte) * bits).astype(np.uint8)
    return (grouped << shifts).sum(-1).astype(np.uint8)


def pack_encoder(model, ternary, source: Path, target: Path, bits: int, accuracy_level: int) -> int:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    lookup = {}
    for name, module in model.named_modules():
        if name in ternary:
            transposed = np.ascontiguousarray(module.weight.detach().numpy().T)
            lookup[hashlib.sha1(transposed.tobytes()).hexdigest()] = name

    graph_model = onnx.load(str(source))
    graph = graph_model.graph
    consumers: dict[str, list] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    matched = {}
    for init in graph.initializer:
        if len(init.dims) == 2 and init.data_type == TensorProto.FLOAT:
            digest = hashlib.sha1(np.ascontiguousarray(numpy_helper.to_array(init)).tobytes()).hexdigest()
            if digest in lookup:
                matched[init.name] = lookup[digest]
    if len(matched) != len(ternary):
        raise RuntimeError(f"matched {len(matched)} of {len(ternary)} ternary weights in the exported graph")

    zero_point = 1 << (bits - 1)
    replaced = 0
    new_initializers = []
    for init_name, module_name in matched.items():
        codes, scales = ternary[module_name]
        rows, cols = codes.shape
        packed = pack_nbits((codes.astype(np.int16) + zero_point).astype(np.uint8), bits, BLOCK)
        new_initializers.append(numpy_helper.from_array(packed, init_name + "_q"))
        new_initializers.append(numpy_helper.from_array(scales.astype(np.float32).reshape(-1), init_name + "_scales"))
        for node in consumers[init_name]:
            if node.op_type != "MatMul" or node.input[1] != init_name:
                raise RuntimeError(f"unexpected consumer {node.op_type} for ternary weight {module_name}")
            node.op_type = "MatMulNBits"
            node.domain = "com.microsoft"
            node.input[:] = [node.input[0], init_name + "_q", init_name + "_scales"]
            del node.attribute[:]
            node.attribute.extend([
                helper.make_attribute("K", cols), helper.make_attribute("N", rows),
                helper.make_attribute("bits", bits), helper.make_attribute("block_size", BLOCK),
                helper.make_attribute("accuracy_level", accuracy_level),
            ])
            replaced += 1
    kept = [init for init in graph.initializer if init.name not in matched]
    del graph.initializer[:]
    graph.initializer.extend(kept + new_initializers)
    if not any(opset.domain == "com.microsoft" for opset in graph_model.opset_import):
        graph_model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    for init in graph.initializer:  # everything fits in one file now
        init.data_location = TensorProto.DEFAULT
        del init.external_data[:]
    onnx.save(graph_model, str(target))
    return replaced


# ----------------------------------------------------------------------------
# Vocabulary and config for onnx-asr, and the numerical checks.
# ----------------------------------------------------------------------------

def write_vocab(root: Path, out: Path, vocab_size: int, blank_id: int) -> None:
    tokenizer = json.loads((root / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = tokenizer["model"]["vocab"]
    pieces = {i: piece for piece, i in vocab.items()} if isinstance(vocab, dict) else {
        i: piece for i, (piece, *_rest) in enumerate(vocab)}
    for added in tokenizer.get("added_tokens", ()):
        pieces[added["id"]] = added["content"]
    lines = []
    for i in range(vocab_size):
        piece = pieces.get(i, f"<unused{i}>")
        if i == blank_id:
            piece = "<blk>"  # the name onnx-asr looks for
        if " " in piece or "\n" in piece:
            raise RuntimeError(f"token {i!r} contains whitespace and cannot go in vocab.txt")
        lines.append(f"{piece} {i}")
    (out / "vocab.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify(model, config, out: Path, bits: int) -> None:
    import onnxruntime as ort

    torch.manual_seed(0)
    providers = ["CPUExecutionProvider"]
    with torch.inference_mode():
        # preprocessor + encoder + vad on a synthetic two-row batch with padding
        rate = 16_000
        t = torch.arange(2 * rate) / rate
        wave = torch.stack([0.3 * torch.sin(2 * torch.pi * 220 * t) * torch.sin(2 * torch.pi * 3 * t),
                            0.1 * torch.randn(2 * rate)])
        lengths = torch.tensor([2 * rate, rate + 1234])
        wave[1, lengths[1]:] = 0
        pre = ort.InferenceSession(str(out / FILES["preprocessor"]), providers=providers)
        features, feature_lens = pre.run(None, {"waveforms": wave.numpy(), "waveforms_lens": lengths.numpy()})
        from kestrel.models.parakeet_tdt.features import parakeet_features
        worst = 0.0
        for row in range(2):
            reference, mask = parakeet_features(wave[row, : lengths[row]])
            valid = int(mask.sum())
            worst = max(worst, float(np.abs(features[row, :, :valid] - reference[0, :valid].numpy().T).max()))
            assert valid == feature_lens[row], (valid, feature_lens[row])
        log(f"  preprocessor vs torch.stft: max abs diff {worst:.1e}")

        audio_signal = torch.from_numpy(features)
        mask = torch.arange(audio_signal.shape[2])[None] < torch.from_numpy(feature_lens)[:, None]
        reference, valid = model.encoder(audio_signal.transpose(1, 2).contiguous(), mask)
        enc = ort.InferenceSession(str(out / FILES["encoder"]), providers=providers)
        outputs, out_lens = enc.run(None, {"audio_signal": features, "length": feature_lens})
        worst = 0.0
        for row in range(2):
            n = int(out_lens[row])
            assert n == int(valid[row].sum()), (n, int(valid[row].sum()))
            worst = max(worst, float(np.abs(outputs[row, :, :n] - reference[row, :n].numpy().T).max()))
        log(f"  encoder ({'dense fp32' if bits == 0 else f'{bits}-bit MatMulNBits'}) vs torch: max abs diff {worst:.1e}")

        vad = ort.InferenceSession(str(out / FILES["vad"]), providers=providers)
        probabilities, vad_lens = vad.run(None, {"audio_signal": features, "length": feature_lens})
        reference, valid = model.speech_probabilities(audio_signal.transpose(1, 2).contiguous(), mask)
        worst = max(float(np.abs(probabilities[row, : int(vad_lens[row])] - reference[row, : int(vad_lens[row])].numpy()).max())
                    for row in range(2))
        log(f"  vad head vs torch: max abs diff {worst:.1e}")

        joint = ort.InferenceSession(str(out / FILES["decoder_joint"]), providers=providers)
        frame = torch.from_numpy(outputs[:, :, :1].copy())
        targets = torch.tensor([[config.blank_token_id], [17]], dtype=torch.int32)
        hidden = torch.randn(2, 2, config.decoder_hidden_size) * 0.1
        cell = torch.randn(2, 2, config.decoder_hidden_size) * 0.1
        logits, _, new_hidden, new_cell = joint.run(None, {
            "encoder_outputs": frame.numpy(), "targets": targets.numpy(), "target_length": np.ones(2, np.int32),
            "input_states_1": hidden.numpy(), "input_states_2": cell.numpy()})
        pred, (ref_hidden, ref_cell) = model.decoder(targets.long(), (hidden, cell))
        ref_logits = model.joint(model.encoder_projector(frame.transpose(1, 2)), pred)
        diff = float(np.abs(logits[:, :, 0] - ref_logits.numpy()).max())
        state_diff = max(float(np.abs(new_hidden - ref_hidden.numpy()).max()), float(np.abs(new_cell - ref_cell.numpy()).max()))
        log(f"  decoder_joint vs torch: logits max abs diff {diff:.1e} (|logits| up to {ref_logits.abs().max().item():.0f}), state {state_diff:.1e}")


def main(argv=None) -> int:
    args = parse_args(argv)
    root = resolve_source(args.model_path)
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_FILES:
        if not (root / name).exists():
            raise SystemExit(f"error: {root / name} is missing")

    started = time.perf_counter()
    config, model, ternary = load_dense(root)
    log(f"loaded {root.name}: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters, "
        f"{len(ternary)} ternary layers, in {time.perf_counter() - started:.1f}s")

    # Encoder: exported dense (external data), then repacked unless --bits 0.
    started = time.perf_counter()
    dense_path = out / ("encoder-model.onnx" if args.bits == 0 else "encoder-fp32.onnx")
    sample = (torch.randn(2, config.encoder.num_mel_bins, 1100), torch.tensor([1100, 550]))
    export(EncoderExport(model), sample, ["audio_signal", "length"], ["outputs", "encoded_lengths"],
           {"audio_signal": {0: "batch", 2: "time"}, "length": {0: "batch"}}, dense_path, external=True)
    log(f"encoder exported (float32, external data) in {time.perf_counter() - started:.1f}s")
    if args.bits:
        started = time.perf_counter()
        target = out / FILES["encoder"]
        replaced = pack_encoder(model, ternary, dense_path, target, args.bits, args.accuracy_level)
        log(f"encoder repacked: {replaced} MatMul nodes -> {args.bits}-bit MatMulNBits, "
            f"{target.stat().st_size / 1e6:.0f} MB, in {time.perf_counter() - started:.1f}s")
        if not args.keep_dense:
            for stale in out.glob("encoder-fp32.onnx*"):
                stale.unlink()

    started = time.perf_counter()
    batch = 2
    export(DecoderJointExport(model),
           (torch.randn(batch, config.encoder.hidden_size, 1),
            torch.tensor([[config.blank_token_id], [17]], dtype=torch.int32), torch.ones(batch, dtype=torch.int32),
            torch.zeros(config.num_decoder_layers, batch, config.decoder_hidden_size),
            torch.zeros(config.num_decoder_layers, batch, config.decoder_hidden_size)),
           ["encoder_outputs", "targets", "target_length", "input_states_1", "input_states_2"],
           ["outputs", "prednet_lengths", "output_states_1", "output_states_2"],
           {"encoder_outputs": {0: "batch"}, "targets": {0: "batch"}, "target_length": {0: "batch"},
            "input_states_1": {1: "batch"}, "input_states_2": {1: "batch"}},
           out / FILES["decoder_joint"])
    export(Preprocessor().eval(), (torch.randn(2, 16_000), torch.tensor([16_000, 9_000])),
           ["waveforms", "waveforms_lens"], ["features", "features_lens"],
           {"waveforms": {0: "batch", 1: "samples"}, "waveforms_lens": {0: "batch"}}, out / FILES["preprocessor"])
    export(VadExport(model), sample, ["audio_signal", "length"], ["probabilities", "encoded_lengths"],
           {"audio_signal": {0: "batch", 2: "time"}, "length": {0: "batch"}}, out / FILES["vad"])
    log(f"decoder_joint, preprocessor and vad exported in {time.perf_counter() - started:.1f}s")

    write_vocab(root, out, config.vocab_size, config.blank_token_id)
    (out / "config.json").write_text(json.dumps({
        "model_type": "nemo-conformer-tdt",
        "features_size": config.encoder.num_mel_bins,
        "subsampling_factor": config.encoder.subsampling_factor,
        "max_tokens_per_step": config.max_symbols_per_step,
        "vocab_size": config.vocab_size,
        "blank_token_id": config.blank_token_id,
        "durations": list(config.durations),
        "encoder_frame_seconds": 160 / 16_000 * config.encoder.subsampling_factor,
        "sample_rate": 16_000,
        "source": "moondream/parakeet-redux",
        "ternary_weights": "dense float32" if args.bits == 0 else f"MatMulNBits {args.bits}-bit, block {BLOCK}",
    }, indent=2) + "\n")

    if not args.no_verify:
        log("verifying against PyTorch...")
        verify(model, config, out, args.bits)

    total = sum(path.stat().st_size for path in out.iterdir() if path.is_file())
    log(f"done: {out} ({total / 1e6:.0f} MB)")
    for path in sorted(out.iterdir()):
        log(f"  {path.stat().st_size / 1e6:8.1f} MB  {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
