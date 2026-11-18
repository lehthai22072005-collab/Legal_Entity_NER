"""Vietnamese segmentation with an explicit map to unchanged DOCX characters."""
from __future__ import annotations

import re
from pathlib import Path


def align_segmented_line(raw, segmented, base=0):
    original = [(c, base + i) for i, c in enumerate(raw) if not c.isspace()]
    cursor = 0
    mapping = []
    for char in segmented:
        if char.isspace():
            mapping.append(None)
        elif cursor < len(original) and char == original[cursor][0]:
            mapping.append(original[cursor][1])
            cursor += 1
        elif char == "_":
            # PyVi inserts '_' where source words are joined. The connector
            # maps to an existing source gap; source underscores are matched above.
            previous = original[cursor - 1][1] if cursor else base
            following = original[cursor][1] if cursor < len(original) else previous
            mapping.append(previous + 1 if following > previous + 1 else None)
        else:
            raise ValueError(f"Segmenter changed source character at {base}: {char!r}")
    if cursor != len(original):
        raise ValueError(f"Segmenter dropped source characters at {base}")
    return mapping


def segment_with_offsets(text):
    from pyvi import ViTokenizer
    segmented_parts, mapping, warnings = [], [], []
    base = 0
    for raw_line in text.splitlines(keepends=True):
        raw = raw_line.rstrip("\r\n")
        newline = raw_line[len(raw):]
        if raw.strip():
            segmented = ViTokenizer.tokenize(raw)
            try:
                local_map = align_segmented_line(raw, segmented, base)
            except ValueError as exc:
                # Preserve source literally if a segmenter cannot represent it.
                # This changes no source character or annotation; it is recorded.
                segmented = raw
                local_map = [base + i for i in range(len(raw))]
                warnings.append({"code": "WORD_SEGMENTATION_ALIGNMENT_FALLBACK",
                                 "start": base, "message": str(exc)})
        else:
            segmented = raw
            local_map = [base + i for i in range(len(raw))]
        segmented_parts.append(segmented + newline)
        mapping.extend(local_map + [base + len(raw) + i for i in range(len(newline))])
        base += len(raw_line)
    result = "".join(segmented_parts)
    assert len(result) == len(mapping)
    return result, mapping, warnings


def load_tokenizer(encoder_dir):
    from transformers import PreTrainedTokenizerFast, PhobertTokenizer
    encoder_dir = Path(encoder_dir)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(encoder_dir / "tokenizer.json"),
        bos_token="<s>", eos_token="</s>", pad_token="<pad>", unk_token="<unk>", mask_token="<mask>")
    slow = PhobertTokenizer(str(encoder_dir / "vocab.txt"), str(encoder_dir / "bpe.codes"))
    samples = ["Tôi là sinh_viên .", "THÔNG_TƯ SỐ 9 / 2019 / TT - BTC",
               "ỦY BAN THƯỜNG_VỤ QUỐC_HỘI", "Điều 14 . 1 - Sổ đăng_ký tầu biển ."]
    for sample in samples:
        if tokenizer.encode(sample, add_special_tokens=False) != slow.encode(sample, add_special_tokens=False):
            raise ValueError("Official fast tokenizer differs from PhoBERT's slow tokenizer")
    tokenizer.official_slow_tokenizer = slow
    return tokenizer


def tokenize_source(text, tokenizer):
    segmented, mapping, warnings = segment_with_offsets(text)
    encoded = tokenizer(segmented, add_special_tokens=False, return_offsets_mapping=True,
                        return_attention_mask=False, return_token_type_ids=False)
    # The published fast tokenizer includes intermediate BPE vocabulary entries
    # above the original model's embedding vocabulary. They are unknown tokens
    # in the official slow tokenizer. Remap them ONLY when full-document token
    # IDs are proven identical to the original tokenizer on segmented whitespace.
    slow = tokenizer.official_slow_tokenizer
    encoded["input_ids"] = [index if index < slow.vocab_size else slow.unk_token_id
                            for index in encoded["input_ids"]]
    expected = slow.encode(" ".join(segmented.split()), add_special_tokens=False)
    if encoded["input_ids"] != expected:
        raise ValueError("Document BPE IDs differ from official PhoBERT tokenizer; refusing misaligned inference")
    offsets = []
    for start, end in encoded["offset_mapping"]:
        real = [pos for pos in mapping[start:end] if pos is not None]
        offsets.append((min(real), max(real) + 1) if real else (0, 0))
    return {"input_ids": encoded["input_ids"], "offsets": offsets, "warnings": warnings}
