#!/bin/bash
# A trained run -> the Q8_0 GGUFs the RT Captions engine (parakeet.cpp) loads: one per default lookahead.
#   la13 = the model as trained (att_context_size[0] = [70,13], 1040 ms lookahead, 1120 ms chunks)
#   laR  = the same weights re-saved with [70,R] first (the converter takes preset [0] as the stream's context):
#          la6 480 ms (560 ms chunks), la1 80 ms (160 ms chunks), la0 0 ms (80 ms chunks)
# Both heads (RNN-T + CTC) are in each file. Only the encoder / joint-projection linears are quantized (see the
# converter's docstring); everything else stays F32. Optionally uploads the .nemo + GGUFs to the HF ckpt repo.
#   bash voice/asr/gguf/export_gguf.sh run2v2 [--push]          LOOK="13 6 1 0" picks the lookaheads (default 13 6)
set -euo pipefail
RUN=$1; PUSH=${2:-}; LOOK=${LOOK:-13 6}
W=/home/marimo/work; A=$W/asr; P=/home/marimo/asrenv/bin/python; G=$A/gguf/$RUN; HERE=$(cd "$(dirname "$0")" && pwd)
NEMO=$A/exp/$RUN/$RUN.nemo
mkdir -p $G
$P -c "import gguf" 2>/dev/null || uv pip install -q --python $P gguf
OUTS=()
for R in $LOOK; do
  SRC=$NEMO
  if [ "$R" != 13 ]; then
    SRC=$G/${RUN}_la$R.nemo
    $P - <<EOF
import nemo.collections.asr as nemo_asr
from omegaconf import open_dict
m = nemo_asr.models.ASRModel.restore_from("$NEMO", map_location="cpu")
ctx = [list(c) for c in m.cfg.encoder.att_context_size]
assert [70, $R] in ctx, ctx
with open_dict(m.cfg):
    m.cfg.encoder.att_context_size = [[70, $R]] + [c for c in ctx if c != [70, $R]]
m.encoder.set_default_att_context_size([70, $R])
m.save_to("$SRC")
print("la$R presets", m.cfg.encoder.att_context_size)
EOF
  fi
  $P $HERE/convert_parakeet_to_gguf.py --model $SRC --dtype q8_0 --output $G/${RUN}_la${R}_q8_0.gguf
  OUTS+=("$G/${RUN}_la${R}_q8_0.gguf")
done
ls -la $G
if [ "$PUSH" = "--push" ]; then
  $P - "$NEMO" "${OUTS[@]}" <<EOF
import os, sys
from huggingface_hub import HfApi
api = HfApi()
for local in sys.argv[1:]:
    remote = "$RUN/" + os.path.basename(local)
    api.upload_file(repo_id="fhai50032/bibo-asr-ckpt", path_or_fileobj=local, path_in_repo=remote,
                    commit_message=f"$RUN {os.path.basename(local)}")
    print("pushed", remote, flush=True)
EOF
fi
echo GGUF_DONE
