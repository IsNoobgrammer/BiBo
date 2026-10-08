#!/bin/bash
# run0: first en/hi streaming ASR. 114M cache-aware FastConformer hybrid TDT/RNNT+CTC (NVIDIA, English, CC-BY-4.0)
# with its vocabulary swapped for our joint 4k en/hi SentencePiece BPE, fine-tuned on the 100 h 55/45 mix.
#   bash voice/asr/run0.sh            (on the ASR box; NeMo repo at $W/NeMo)
set -euo pipefail
W=/home/marimo/work; A=$W/asr; R=$A/run0; P=/tmp/uv-venv/bin/python   # the env NeMo was installed into
cd $W/BiBo && git log --oneline -1
# NeMo's RNNT loss JIT-compiles numba CUDA kernels: needs libnvvm (numba-cuda cu13 wheels); numba-cuda 0.30 breaks on numpy>=2.4
uv pip install -q --python $P "numba-cuda[cu13]" "numpy<2.4"

$P voice/asr/prep_train.py --mix $A/mix100 --out $R

# joint tokenizer: balanced text, full Devanagari coverage, byte fallback for unseen characters, no language tags
$P $W/NeMo/scripts/tokenizers/process_asr_text_tokenizer.py --data_file $R/tokenizer.txt --data_root $R/tok \
  --vocab_size 4096 --tokenizer spe --spe_type bpe --spe_character_coverage 1.0 --spe_byte_fallback --no_lower_case
TOK=$(ls -d $R/tok/tokenizer_spe_bpe_v4096*)

$P $W/NeMo/examples/asr/speech_to_text_finetune.py \
  --config-path=$W/NeMo/examples/asr/conf/asr_finetune --config-name=speech_to_text_finetune \
  +init_from_pretrained_model=stt_en_fastconformer_hybrid_large_streaming_multi \
  model.tokenizer.update_tokenizer=true model.tokenizer.dir=$TOK model.tokenizer.type=bpe \
  model.train_ds.manifest_filepath=$R/train.jsonl model.train_ds.batch_size=64 model.train_ds.max_duration=30 \
  "model.validation_ds.manifest_filepath=[$R/val_en.jsonl,$R/val_hi.jsonl]" model.validation_ds.batch_size=64 \
  model.optim.lr=5e-4 model.optim.sched.warmup_steps=1000 model.optim.sched.min_lr=1e-5 \
  trainer.devices=1 trainer.max_epochs=30 trainer.precision=bf16-mixed trainer.strategy=auto \
  exp_manager.exp_dir=$R/exp exp_manager.name=run0 \
  exp_manager.checkpoint_callback_params.save_top_k=2
echo RUN0_DONE
