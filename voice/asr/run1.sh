#!/bin/bash
# run1: en/hi streaming ASR with speaker turns. 114M cache-aware FastConformer hybrid RNNT+CTC (NVIDIA, English,
# CC-BY-4.0), vocabulary swapped for our joint 4k en/hi SentencePiece BPE (+ <spk1..4>), fine-tuned on the 400 h mix
# (HI ~225 : EN ~175 effective + augmented repeats + ~15% multi-speaker windows). Eval every ~150 audio hours on the
# per-source speaker-held-out val sets (+ <spk> windows) + FLEURS en/hi, logged to W&B project bibo-asr
# (layout: voice/asr/wb_layout.py). run0 (100 h, NeMo finetune script) is in
# git history.
#   bash voice/asr/run1.sh [extra train_asr.py args, e.g. --compile_layers]   (after the setup_repo cell)
set -euo pipefail
W=/home/marimo/work; A=$W/asr; R=$A/run1; E=$A/eval_sets; P=/home/marimo/asrenv/bin/python
cd $W/BiBo && git log --oneline -1

[ -f $R/train.jsonl ] || $P voice/asr/prep_train.py --mix $A/mix400 --out $R
[ -f $R/val_multispk.jsonl ] || $P voice/asr/prep_train.py --mix $A/mix400 --out $R --val_only   # per-source val sets
[ -f $E/fleurs_hi.jsonl ] || $P voice/asr/build_eval.py --out $E

# a resumed run must reuse its tokenizer: pull it (and last.ckpt) from the HF ckpt repo before training a new one
[ -d $R/tok/tokenizer_spe_bpe_v4096 ] || $P voice/asr/hf_sync.py pull run1 $A

# joint tokenizer: balanced text, full Devanagari coverage, byte fallback for unseen characters, no language tags
[ -d $R/tok/tokenizer_spe_bpe_v4096 ] || $P $W/NeMo/scripts/tokenizers/process_asr_text_tokenizer.py \
  --data_file $R/tokenizer.txt --data_root $R/tok --vocab_size 4096 --tokenizer spe --spe_type bpe \
  --spe_character_coverage 1.0 --spe_byte_fallback --no_lower_case \
  --spe_user_defined_symbols "<spk1>" "<spk2>" "<spk3>" "<spk4>"

$P voice/asr/train_asr.py --run run1 --train $R/train.jsonl --tok $R/tok/tokenizer_spe_bpe_v4096 \
  --val $(ls $R/val_*.jsonl | grep -v -e '/val_en.jsonl' -e '/val_hi.jsonl') $E/fleurs_en.jsonl $E/fleurs_hi.jsonl "$@"
echo RUN1_DONE
