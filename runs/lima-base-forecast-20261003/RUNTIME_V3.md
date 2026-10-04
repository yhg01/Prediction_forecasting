# LIMA data correction

Authentication succeeded on 2026-10-03. The pinned official training file contains 1,030 rows. It has 1,000 single-turn examples and 30 multi-turn examples. One example has five turns and ends with a user message. No turn is empty.

The original preparation code required exactly 1,000 rows and an even number of turns. That assumption prevented training. The corrected code retains all 1,030 rows and every original turn. The user message at the end remains in the input. Its tokens have no training loss. Every assistant response and its end token have training loss.

No training job ran before this correction. The model revisions, training parameters, final-epoch selection, forecast prompts, decoding parameters, and paired generation seeds remain those in the original protocol. Three full epochs now cover the complete official file. The original source bundle and submission records are retained. The corrected runtime has a separate hash record. Existing pretrained forecast jobs use the original worker.

Use `remote_v3.py` for data preparation, previews, and submission. Use `remote.py sync` to copy records. All submissions use the shared guard, lock, and ledger.
