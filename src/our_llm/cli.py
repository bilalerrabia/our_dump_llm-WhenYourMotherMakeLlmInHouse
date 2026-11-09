"""`our-llm` command line interface.

    our-llm tokenize    # read corpus -> train/load BPE -> encode + report
    our-llm train       # (M4) pretrain the GPT
    our-llm sample      # (M5) generate text from a checkpoint
    our-llm eval        # (M4) loss / perplexity on a checkpoint
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from our_llm.config import DEFAULT_NUM_MERGES, get_preset
from our_llm.data.corpus import Corpus, read_corpus
from our_llm.data.dataset import split_documents
from our_llm.data.tokens import (
    check_compatible,
    encode_corpus,
    load_meta,
    meta_path,
    token_file_path,
    write_tokens,
)
from our_llm.tokenizer.bpe import (
    BPETokenizer,
    corpus_hash,
    load_or_train,
    tokenizer_cache_path,
)

#: Cache left behind by the first tokenizer implementation (merges start at 256).
LEGACY_CACHE = "tokenizer_vocab_merges.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="our-llm",
        description="A GPT built from scratch: corpus -> BPE -> GPT -> training -> sampling.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    tok = sub.add_parser("tokenize", help="train/load the BPE tokenizer and encode the corpus")
    tok.add_argument("--corpus", default="data", help="corpus file or directory (default: data)")
    tok.add_argument(
        "--num-merges",
        type=int,
        default=DEFAULT_NUM_MERGES,
        help=f"merges to learn if retraining (default: {DEFAULT_NUM_MERGES})",
    )
    tok.add_argument("--cache-dir", default="data/cache", help="where to keep vocab caches")
    tok.add_argument(
        "--vocab-cache",
        default=None,
        help=f"load these merges instead of training (tries {LEGACY_CACHE})",
    )
    tok.add_argument("--retrain", action="store_true", help="ignore caches and train again")
    tok.add_argument(
        "--processed-dir", default="data/processed", help="where to write the token file"
    )
    tok.add_argument(
        "--no-tokens", action="store_true", help="only report; do not write the token file"
    )
    tok.add_argument(
        "--force", action="store_true", help="re-encode even if the token file is current"
    )
    tok.add_argument(
        "--granularity",
        default="paragraph",
        choices=["file", "paragraph"],
        help="what counts as one document for the train/val split (default: paragraph)",
    )
    tok.add_argument(
        "--val-ratio", type=float, default=0.10, help="fraction of documents held out"
    )
    tok.set_defaults(func=cmd_tokenize)

    train = sub.add_parser("train", help="pretrain the GPT (not implemented yet: M4)")
    train.add_argument("--preset", default="tiny", choices=["debug", "tiny", "gpt2-small"])
    train.set_defaults(func=cmd_not_implemented)

    sample = sub.add_parser("sample", help="generate text (not implemented yet: M5)")
    sample.add_argument("--ckpt", required=True)
    sample.add_argument("--prompt", default="The monster")
    sample.set_defaults(func=cmd_not_implemented)

    ev = sub.add_parser("eval", help="loss / perplexity of a checkpoint (M4)")
    ev.add_argument("--ckpt", required=True)
    ev.set_defaults(func=cmd_not_implemented)

    return parser


# --- tokenize ---------------------------------------------------------------


def _open_tokenizer(args: argparse.Namespace, corpus: Corpus) -> tuple[BPETokenizer, str]:
    """Pick a tokenizer. Order: explicit -> --retrain -> corpus cache -> legacy -> train."""
    corpus_text = corpus.text
    # The vocabulary depends on the characters only, so key it on the raw files
    # (not on the document layout, which differs between granularities).
    key = corpus.files_hash or corpus_hash(corpus_text)
    path = tokenizer_cache_path(args.cache_dir, key, args.num_merges)

    if args.retrain:
        tokenizer = BPETokenizer().train(corpus_text, args.num_merges)
        tokenizer.save(path)
        return tokenizer, f"trained on this corpus -> {path}"

    if args.vocab_cache:
        return BPETokenizer.load(args.vocab_cache), f"loaded from {args.vocab_cache}"

    # Keyed by (raw-corpus fingerprint, merge count), so it can never go stale.
    if os.path.exists(path):
        return BPETokenizer.load(path), f"cache hit {path}"

    # Older cache from before the layout change: usable, but not corpus-keyed.
    if os.path.exists(LEGACY_CACHE):
        tokenizer = BPETokenizer.load(LEGACY_CACHE)
        return tokenizer, (
            f"loaded from {LEGACY_CACHE} (legacy file, merges shifted to the 258+ "
            f"layout; use --retrain for a corpus-matched vocabulary)"
        )

    tokenizer = load_or_train(corpus_text, args.num_merges, args.cache_dir, key=key)
    return tokenizer, f"trained -> {path}"


def cmd_tokenize(args: argparse.Namespace) -> int:
    corpus = read_corpus(args.corpus, granularity=args.granularity)
    if not corpus:
        print(f"No corpus files found in {args.corpus!r}", file=sys.stderr)
        return 1

    text = corpus.text
    unique_files = sorted(set(corpus.sources))
    print(f"Corpus : {len(corpus.docs)} {args.granularity} document(s) from {len(unique_files)} file(s)")
    for source in unique_files:
        print(f"         - {source}")
    print(f"         {len(text):,} characters, sha256[:16]={corpus.fingerprint}")

    tokenizer, provenance = _open_tokenizer(args, corpus)
    print(f"Vocab  : {tokenizer.vocab_size:,} tokens ({tokenizer.num_merges:,} merges)")
    print(f"         {provenance}")

    # Encode document by document (that is what gives us splittable spans), and
    # reuse the existing file when it is still valid.
    path = token_file_path(
        args.processed_dir, corpus.structure_fingerprint, tokenizer.num_merges
    )
    cached = os.path.exists(path) and os.path.exists(meta_path(path))

    start = time.perf_counter()
    reused = False
    if cached and not args.force:
        meta = load_meta(path)
        check_compatible(meta, tokenizer)  # raises if the vocab changed underneath
        n_tokens = int(meta["n_tokens"])
        reused = True
    elif args.no_tokens:
        encoded = encode_corpus(tokenizer, corpus)
        n_tokens = encoded.n_tokens
        elapsed = time.perf_counter() - start
        print(f"Tokens : {n_tokens:,} in {elapsed:.2f}s (not written: --no-tokens)")
        return 0
    else:
        encoded = encode_corpus(tokenizer, corpus)
        meta = write_tokens(path, encoded, corpus=corpus, tokenizer=tokenizer)
        n_tokens = encoded.n_tokens

    elapsed = time.perf_counter() - start
    ratio = len(text) / max(n_tokens, 1)
    source_note = "reused" if reused else "encoded"
    print(f"Tokens : {n_tokens:,} ({ratio:.2f} chars/token, {source_note} in {elapsed:.2f}s)")
    print(
        f"         dtype {meta['dtype']}, {meta['n_documents']} document(s) of tokens"
        + ("" if args.no_tokens else f" -> {path}")
    )

    train_docs, val_docs = split_documents(len(meta["doc_spans"]), args.val_ratio)
    spans = [(int(s), int(e)) for s, e in meta["doc_spans"]]
    train_tokens = sum(e - s for s, e in (spans[i] for i in train_docs))
    val_tokens = sum(e - s for s, e in (spans[i] for i in val_docs))
    print(
        f"Split  : train {len(train_docs)} docs / {train_tokens:,} tokens, "
        f"val {len(val_docs)} docs / {val_tokens:,} tokens (val_ratio {args.val_ratio})"
    )
    if not val_docs:
        print(
            "         note: only one document -> nothing to validate on "
            "(try --granularity paragraph)",
            file=sys.stderr,
        )
    return 0


# --- stubs ------------------------------------------------------------------


def cmd_not_implemented(args: argparse.Namespace) -> int:
    if args.command == "train":
        cfg = get_preset(args.preset)
        print(
            f"'train' is not implemented yet (ROADMAP M4).\n"
            f"Preset {args.preset!r}: {cfg.num_parameters():,} parameters "
            f"(ctx {cfg.context_length}, emb {cfg.emb_dim}, {cfg.n_layers} layers).",
            file=sys.stderr,
        )
    else:
        print(f"'{args.command}' is not implemented yet (see ROADMAP.md).", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
