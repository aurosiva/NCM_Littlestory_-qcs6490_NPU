"""Download TinyStories, train a small byte-level BPE, and write fixed-length
token arrays: data/train.bin, data/val.bin (uint16, shape [N, L]).
Each story is truncated to L tokens, gets an EOS, then PAD to length L.
"""
import argparse
import os

import numpy as np
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

from config import TokCfg

PAD, EOS = 0, 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_train", type=int, default=1_000_000, help="stories to keep (2.1M available)")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    cfg = TokCfg()
    os.makedirs(args.out, exist_ok=True)

    ds = load_dataset("roneneldan/TinyStories")
    train = ds["train"].shuffle(seed=0).select(range(min(args.max_train, len(ds["train"]))))
    val = ds["validation"]

    tok_path = os.path.join(args.out, "bpe")
    if not os.path.exists(os.path.join(tok_path, "vocab.json")):
        os.makedirs(tok_path, exist_ok=True)
        bpe = ByteLevelBPETokenizer()
        texts = (t for t in train.select(range(min(300_000, len(train))))["text"])
        bpe.train_from_iterator(texts, vocab_size=cfg.vocab, min_frequency=2,
                                special_tokens=["<pad>", "<eos>"])
        bpe.save_model(tok_path)
    bpe = ByteLevelBPETokenizer(os.path.join(tok_path, "vocab.json"), os.path.join(tok_path, "merges.txt"))
    assert bpe.token_to_id("<pad>") == PAD and bpe.token_to_id("<eos>") == EOS

    for name, split in [("train", train), ("val", val)]:
        arr = np.full((len(split), cfg.L), PAD, dtype=np.uint16)
        bs = 10_000
        for i in range(0, len(split), bs):
            encs = bpe.encode_batch([t.strip() for t in split[i:i + bs]["text"]])
            for j, e in enumerate(encs):
                ids = e.ids[: cfg.L - 1] + [EOS]
                arr[i + j, : len(ids)] = ids
            print(f"{name}: {min(i + bs, len(split))}/{len(split)}", end="\r")
        arr.tofile(os.path.join(args.out, f"{name}.bin"))
        print(f"\n{name}: {arr.shape} -> {args.out}/{name}.bin")


if __name__ == "__main__":
    main()
