#!/bin/bash
# A trained run -> the Q8_0 GGUFs the RT Captions engine (parakeet.cpp) loads: one per default lookahead.
#   la13 = the model as trained (att_context_size[0] = [70,13], 1040 ms lookahead)
#   la6  = the same weights re-saved with [70,6] first (480 ms lookahead; the converter takes preset [0] as default)
# Both heads (RNN-T + CTC) are in each file. Only the encoder / joint-projection linears are quantized (see the
# converter's docstring); everything else stays F32. Optionally uploads the .nemo + GGUFs to the HF ckpt repo.
#   bash voice/asr/gguf/export_gguf.sh run2v2 [--push]
set -euo pipefail
RUN=$1; PUSH=${2:-}
W=/home/marimo/work; A=$W/asr; P=/home/marimo/asrenv/bin/python; G=$A/gguf/$RUN; HERE=$(cd "$(dirname "$0")" && pwd)
NEMO=$A/exp/$RUN/$RUN.nemo
mkdir -p $G
$P -c "import gguf" 2>/dev/null || uv pip install -q --python $P gguf
$P - <<EOF
import nemo.collections.asr as nemo_asr
from omegaconf import open_dict
m = nemo_asr.models.ASRModel.restore_from("$NEMO", map_location="cpu")
ctx = [list(c) for c in m.cfg.encoder.att_context_size]
with open_dict(m.cfg):
    m.cfg.encoder.att_context_size = [[70, 6]] + [c for c in ctx if c != [70, 6]]
m.encoder.set_default_att_context_size([70, 6])
m.save_to("$G/${RUN}_la6.nemo")
print("la6 presets", m.cfg.encoder.att_context_size)
EOF
$P $HERE/convert_parakeet_to_gguf.py --model $NEMO --dtype q8_0 --output $G/${RUN}_la13_q8_0.gguf
$P $HERE/convert_parakeet_to_gguf.py --model $G/${RUN}_la6.nemo --dtype q8_0 --output $G/${RUN}_la6_q8_0.gguf
ls -la $G
if [ "$PUSH" = "--push" ]; then
  $P - <<EOF
from huggingface_hub import HfApi
api = HfApi()
for local, remote in (("$NEMO", "$RUN/$RUN.nemo"), ("$G/${RUN}_la13_q8_0.gguf", "$RUN/${RUN}_la13_q8_0.gguf"),
                      ("$G/${RUN}_la6_q8_0.gguf", "$RUN/${RUN}_la6_q8_0.gguf")):
    api.upload_file(repo_id="fhai50032/bibo-asr-ckpt", path_or_fileobj=local, path_in_repo=remote,
                    commit_message=f"$RUN {remote.split('/')[-1]}")
    print("pushed", remote, flush=True)
EOF
fi
echo GGUF_DONE
