#!/usr/bin/env python3
"""Regenerate row-ID vectors by executing digest-pinned upstream PLE methods on CPU.

See docs/evidence/2026-09-05-ple-reference.md for inputs, commands and proof boundaries.
The reference hash is never reimplemented here; only the embedding lookup is substituted.
"""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/ple-reference.json"
TRANSFORMERS_REVISION = "f62dc9bf2c90353b442a56e74391fbb8c689b55e"
MODEL_REVISION = "34567a4712bc9766c4449e2e98e4468bfa24d915"
TRANSFORMERS_BASE = (
    "https://raw.githubusercontent.com/huggingface/transformers/"
    f"{TRANSFORMERS_REVISION}/src/transformers/models/qwen4_exp/"
)
SOURCES = {
    "modeling_qwen4_exp.py": {
        "url": TRANSFORMERS_BASE + "modeling_qwen4_exp.py",
        "sha256": "2e36ee6a1bc4f43fa0434ae197c2de9eba27a71aec84ee1ff111dc3a812e8283",
    },
    "configuration_qwen4_exp.py": {
        "url": TRANSFORMERS_BASE + "configuration_qwen4_exp.py",
        "sha256": "b78132d8cd935437208ee281fa4569b771a63fcb58ebffe84f3e62f5b86235ca",
    },
    "config.json": {
        "url": (
            "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/"
            f"{MODEL_REVISION}/config.json"
        ),
        "sha256": "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b",
    },
}


class RowIds(torch.nn.Module):
    """Replace the large embedding boundary while retaining its forward output shape."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.register_buffer("weight", torch.empty(0))

    def forward(self, row_ids):
        return row_ids.unsqueeze(-1)


def read_references(directory):
    sources = {}
    for name, provenance in SOURCES.items():
        data = (directory / name).read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != provenance["sha256"]:
            raise ValueError(f"{name}: unexpected SHA-256 {actual}")
        sources[name] = data.decode("utf-8")
    return sources


def check_projection(reduced, official, path="config"):
    if isinstance(reduced, dict):
        if not isinstance(official, dict):
            raise ValueError(f"{path}: official object missing")
        for key, value in reduced.items():
            if key not in official:
                raise ValueError(f"{path}.{key}: missing in official config")
            check_projection(value, official[key], f"{path}.{key}")
    elif type(reduced) is not type(official) or reduced != official:
        raise ValueError(f"{path}: fixture differs from official config")


def reference_class(source):
    names = {
        "_MASK64", "_SPLITMIX_GAMMA", "_SPLITMIX_M1", "_SPLITMIX_M2", "_PRIME_1",
        "_splitmix64", "_build_layer_multipliers", "_is_prime", "_find_nth_prime_after",
        "Qwen4ExpTextNGramEmbedding",
    }
    selected = []
    found = set()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        else:
            continue
        if name in names:
            selected.append(node)
            found.add(name)
    if found != names:
        raise ValueError(f"missing upstream definitions: {names - found}")
    future = ast.parse("from __future__ import annotations").body
    module = ast.Module(body=future + selected, type_ignores=[])
    namespace = {
        "torch": torch,
        "math": math,
        "nn": SimpleNamespace(Module=torch.nn.Module, Buffer=torch.nn.Buffer, Embedding=RowIds),
    }
    exec(compile(module, SOURCES["modeling_qwen4_exp.py"]["url"], "exec"), namespace)
    return namespace["Qwen4ExpTextNGramEmbedding"]


def default_seed(source):
    config = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen4ExpTextConfig"
    )
    field = next(
        node for node in config.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        and node.target.id == "seed"
    )
    return ast.literal_eval(field.value)


def plan_vectors(name, reference, traces):
    records = []
    with torch.no_grad():
        for trace_name, tokens in traces:
            input_ids = torch.tensor([tokens], dtype=torch.int64, device="cpu")
            rows = reference(input_ids, past_key_values=None).squeeze(0)
            if list(rows.shape) != [len(tokens), 16]:
                raise ValueError(f"unexpected upstream output shape: {list(rows.shape)}")
            records.append({"name": trace_name, "tokens": tokens, "rows": rows.tolist()})
    return {
        "name": name,
        "ngram_size": reference.ngram_size,
        "heads_per_ngram": reference.heads_per_ngram,
        "eos_token_id": reference.eos_token_id,
        "multipliers": reference.layer_multipliers.tolist(),
        "vocab_sizes": reference.ngram_heads_vocab_sizes.tolist(),
        "offsets": reference.ngram_heads_offsets.tolist(),
        "traces": records,
    }


def generate(directory):
    if torch.__version__ != "2.8.0+cpu":
        raise ValueError("regeneration requires the recorded torch==2.8.0+cpu version")
    torch.set_num_threads(1)
    sources = read_references(directory)
    config = json.loads(sources["config.json"])
    reduced_bytes = (ROOT / "tests/fixtures/qwen38-flash-next-config.json").read_bytes()
    check_projection(json.loads(reduced_bytes), config)
    text = dict(config["text_config"])
    text.setdefault("seed", default_seed(sources["configuration_qwen4_exp.py"]))
    constructor = reference_class(sources["modeling_qwen4_exp.py"])
    # PLE layer IDs in the model config are one-based; this model has one PLE layer.
    if text["ple_layer_ids"] != [2]:
        raise ValueError("unexpected PLE layer schedule")
    reference = constructor(SimpleNamespace(**text), text["ple_embed_dim"], 1, 0)
    eos = text["eos_token_id"]
    defaults = plan_vectors("upstream_defaults", reference, [
        ("eos_boundaries", [0, 1, 2, 42, text["vocab_size"] - 1, eos, 99, 100, eos, eos, 101, 102, eos, 103]),
        ("draft", [7, 8, 11, eos, 99]),
        ("accepted_prefix", [7, 8, 11, eos, 12, 13]),
        ("aborted_prefix", [7, 8, 12, 13]),
        # Outside the released token vocabulary; exercises the Rust API's u32 input domain.
        ("u32_boundaries", [2**32 - 1, 2**31, 0, eos, 2**32 - 1]),
    ])
    # Synthetic supported i64 metadata forces signed wrap and XOR sign-bit changes.
    # These values are deliberately not described as checkpoint parameters.
    reference.layer_multipliers = torch.tensor([2**62 + 1, 2**61 + 3, 2**60 + 7], dtype=torch.int64)
    overflow = plan_vectors("synthetic_signed_overflow", reference, [
        ("overflow", [0, 1, 2, 3, 4, 7, eos, 42, 2**32 - 1, eos, eos, 11]),
    ])
    return {
        "schema_version": 1,
        "provenance": {
            "sources": SOURCES,
            "torch_version": torch.__version__,
            "device": "cpu",
            "config_projection_sha256": hashlib.sha256(reduced_bytes).hexdigest(),
            "seed": text["seed"],
            "metadata_origin": "upstream constructor defaults, not checkpoint buffer bytes",
            "substitution": "embedding lookup returns its input row IDs; hash methods are unchanged",
        },
        "plans": [defaults, overflow],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--write", action="store_true", help="replace generated fixture; default verifies it")
    args = parser.parse_args()
    encoded = json.dumps(generate(args.reference_dir), indent=2) + "\n"
    if args.write:
        FIXTURE.write_text(encoded, encoding="utf-8")
        print(f"wrote {FIXTURE}")
    elif FIXTURE.read_text(encoding="utf-8") != encoded:
        raise ValueError("stored PLE vectors differ from the upstream reference")
    else:
        print("PLE vectors and config projection match the pinned upstream reference")


if __name__ == "__main__":
    main()
