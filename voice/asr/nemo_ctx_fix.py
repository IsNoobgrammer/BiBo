"""Copy a .nemo with its encoder look-ahead set made loadable:  python nemo_ctx_fix.py in.nemo out.nemo

NeMo's ConformerEncoder refuses [left, R] unless left % (R+1) == 0, but train_asr (before 2026-10-10) saved [70, R]
for every trained R ([70, 3] for 240 ms). The training mask used left // (R+1) chunks, so [70 - 70 % (R+1), R] is the
identical mask: [70, 3] -> [68, 3]. Everything else in the archive is copied byte for byte.
"""
import io
import shutil
import sys
import tarfile

import yaml

src, dst = sys.argv[1], sys.argv[2]
with tarfile.open(src) as t:
    name = next(n for n in t.getnames() if n.endswith("model_config.yaml"))
    cfg = yaml.safe_load(t.extractfile(name))
    ctx = cfg["encoder"]["att_context_size"]
    fixed = [[left - left % (r + 1), r] for left, r in ctx] if isinstance(ctx[0], list) else ctx
    print("att_context_size", ctx, "->", fixed, flush=True)
    if fixed == ctx:
        shutil.copy(src, dst)
        sys.exit()
    cfg["encoder"]["att_context_size"] = fixed
    with tarfile.open(dst, "w") as out:
        for mem in t.getmembers():
            if mem.name == name:
                b = yaml.safe_dump(cfg).encode()
                mem.size = len(b)
                out.addfile(mem, io.BytesIO(b))
            else:
                out.addfile(mem, t.extractfile(mem) if mem.isfile() else None)
