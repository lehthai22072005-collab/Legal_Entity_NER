"""Run real DOCX inference and validate its output; does not train or score F1."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("docx", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "checkpoints/phobert_span_all369_v1")
    args = parser.parse_args()
    from src.pipeline import predict, FIELDS
    began = time.monotonic()
    result = predict(args.docx, args.checkpoint, device="cpu")
    text = result["full_text"]
    assert result["schema_version"] == "entity_only_1.0"
    assert set(result["document_metadata"]) == set(FIELDS.values())
    assert "relations" not in result
    assert result["external_metadata_retrieved"] is False
    assert result["model"]["relation_head_present"] is False
    with args.docx.open("rb") as stream:
        assert result["docx_sha256"] == hashlib.file_digest(stream, "sha256").hexdigest()
    for entity in result["entities"]:
        assert 0 <= entity["start"] < entity["end"] <= len(text)
        assert text[entity["start"]:entity["end"]] == entity["text"]
    for label, key in FIELDS.items():
        value = result["document_metadata"][key]
        if value is not None:
            assert any(e["label"] == label and e["text"] == value for e in result["entities"])
    assert result["entities"], "No entities extracted from the selected smoke-test document"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "DOCX_INFERENCE_SMOKE_PASS", "file": args.docx.name,
                      "characters": len(text), "entities": len(result["entities"]),
                      "offsets_verified": True, "source_sha256_verified": True,
                      "external_metadata_retrieved": False,
                      "seconds": round(time.monotonic() - began, 2),
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
