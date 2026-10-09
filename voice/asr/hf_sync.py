"""Survive box death: training state lives in a private HF model repo, not only on the box's disk.

    python voice/asr/hf_sync.py pull run1 /home/marimo/work/asr      # fresh box: tokenizer + last.ckpt back in place
    (train_asr.py pushes after every evaluation via push())

Repo layout: <run>/tok/* (tokenizer -- a resumed run MUST reuse it; retraining could change the vocab),
<run>/last.ckpt (Lightning: weights + optimizer + scheduler + step), <run>/wandb_id.
"""
import os
import shutil
import sys
import threading

from huggingface_hub import HfApi, snapshot_download

REPO = "fhai50032/bibo-asr-ckpt"


TOK_FILES = ["tokenizer.model", "tokenizer.vocab", "vocab.txt"]
_PUSH_LOCK = threading.Lock()          # one upload at a time: a slow push must not overlap the next eval's


def push(run, tok_dir=None, ckpt=None, wandb_id=None, block=False):
    # The checkpoint is FROZEN (copied) before the thread starts: ModelCheckpoint rewrites last.ckpt at the next
    # eval, and uploading the live file raced with that ("LFS pointer pointed to a file that does not exist").
    snap = None
    if ckpt and os.path.exists(ckpt):
        snap = ckpt + ".upload"
        shutil.copyfile(ckpt, snap)

    def _go():
        with _PUSH_LOCK:
            api = HfApi()
            api.create_repo(REPO, private=True, exist_ok=True)
            if tok_dir and os.path.isdir(tok_dir):
                # only the tokenizer files themselves (a stray nested tok/ dir was uploaded as <run>/tok/tok/)
                api.upload_folder(repo_id=REPO, folder_path=tok_dir, path_in_repo=f"{run}/tok",
                                  allow_patterns=TOK_FILES, commit_message=f"{run} tok")
            if snap:
                try:
                    api.upload_file(repo_id=REPO, path_or_fileobj=snap, path_in_repo=f"{run}/last.ckpt",
                                    commit_message=f"{run} ckpt")
                finally:
                    os.remove(snap)
            if wandb_id:
                api.upload_file(repo_id=REPO, path_or_fileobj=wandb_id.encode(), path_in_repo=f"{run}/wandb_id",
                                commit_message=f"{run} wandb id")
    if block:
        _go()
    else:
        threading.Thread(target=_go, daemon=False).start()      # never blocks the training loop


def pull(run, asr_dir):
    """-> (tok_dir or None, ckpt_path or None, wandb_id or None); missing pieces are simply None."""
    try:
        d = snapshot_download(REPO, allow_patterns=[f"{run}/*", f"{run}/tok/*"],
                              local_dir=os.path.join(asr_dir, "hf_ckpt"))
    except Exception as e:                                      # repo not created yet = fresh run
        print(f"hf_sync: nothing to pull ({type(e).__name__})", flush=True)
        return None, None, None
    base = os.path.join(d, run)
    tok = os.path.join(base, "tok") if os.path.isdir(os.path.join(base, "tok")) else None
    ckpt = os.path.join(base, "last.ckpt") if os.path.exists(os.path.join(base, "last.ckpt")) else None
    wid = open(os.path.join(base, "wandb_id")).read().strip() if os.path.exists(os.path.join(base, "wandb_id")) else None
    print(f"hf_sync: tok={tok} ckpt={ckpt} wandb_id={wid}", flush=True)
    return tok, ckpt, wid


if __name__ == "__main__":
    if sys.argv[1] == "pull":
        tok, _, _ = pull(sys.argv[2], sys.argv[3])
        if tok:                                                 # put the run's tokenizer where run1.sh expects it
            dst = os.path.join(sys.argv[3], sys.argv[2], "tok", "tokenizer_spe_bpe_v4096")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst):
                os.symlink(tok, dst)
