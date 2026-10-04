"""Check assistant loss masks with each actual pretrained tokenizer."""
import json
from pathlib import Path
from common import MODELS, render_training, label_tokens, verify_bundle, write_json


def main():
    root = Path(__file__).resolve().parent
    verify_bundle(root)
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    report = {}
    for key, (repo, revision, _, _) in MODELS.items():
        path = root / "tokenizer-preflight" / key
        snapshot_download(repo, revision=revision, local_dir=path,
            allow_patterns=["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "config.json"], max_workers=2)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
        messages = [{"role": "user", "content": "Repeat: same α🙂"}, {"role": "assistant", "content": "same α🙂"},
                    {"role": "user", "content": "Second turn"}, {"role": "assistant", "content": "A second answer."}]
        text, spans = render_training(messages)
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        labels = label_tokens(encoded["input_ids"], encoded["offset_mapping"], spans)
        eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if eos is None or eos == tokenizer.unk_token_id or sum(v == eos for v in labels) != 2:
            raise ValueError("User and assistant end tokens were not masked correctly")
        for a, b in spans:
            indices = [i for i, (start, end) in enumerate(encoded["offset_mapping"]) if end > start and start >= a and end <= b]
            if not indices or labels[indices[-1]] != eos:
                raise ValueError("An assistant end token has no training label")
        report[key] = {"status": "passed", "fast_tokenizer": tokenizer.is_fast,
                       "im_end_id": eos, "endoftext_id": tokenizer.convert_tokens_to_ids("<|endoftext|>"),
                       "supervised_tokens": sum(v != -100 for v in labels), "multi_turn_unicode_mask": "passed"}
        write_json(root / "TOKENIZER_PREFLIGHT.json", report)
        print(json.dumps({"model": key, **report[key]}), flush=True)


if __name__ == "__main__": main()
