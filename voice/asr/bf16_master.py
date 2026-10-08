"""bf16 weights + fp32 master weights in the optimizer, instead of autocast (train_asr.py --bf16_master).

Under bf16-mixed autocast the weights are fp32 and every op's output dtype follows the autocast lists: LayerNorm,
softmax and adds come out fp32 and the next linear casts back -- prof_step.py measured 42 ms/step of copy/cast
kernels (20% of GPU time) plus their launches. Here the model itself is bf16 (no autocast), with exceptions:
  preprocessor (STFT / mel)        fp32: audio features need the range; the encoder input is cast after it
  BatchNorm (conformer conv)       fp32 weights + running stats; CUDA batch_norm takes bf16 input with fp32 params
  CTC log-probs                    cast to fp32 before the loss (ctc_loss has no bf16 kernel); RNN-T losses are fp32
The optimizer steps fp32 MASTER copies (AdamW state fp32 too): grads are copied up, clipped on the masters, the step
runs, the masters are copied back down. Built lazily on the first step (after Lightning moved the model to the GPU);
a resumed optimizer state is moved onto the masters. Master weights are not checkpointed: a resume restarts them
from the bf16 weights (one bf16 rounding of the weights, once).

    model = bf16_master.to_bf16(model)                     # before setup_optimization
    bf16_master.wrap(model._optimizer, clip=1.0)           # after it; Trainer(precision="32-true", no gradient clip)
"""
import torch

KEEP_FP32 = (torch.nn.modules.batchnorm._BatchNorm,)


def to_bf16(model):
    model.to(torch.bfloat16)
    model.preprocessor.float()
    for mod in model.modules():
        if isinstance(mod, KEEP_FP32):
            mod.float()

    def enc_in(mod, args, kwargs):
        if "audio_signal" in kwargs:
            kwargs["audio_signal"] = kwargs["audio_signal"].to(torch.bfloat16)
        elif args:
            args = (args[0].to(torch.bfloat16),) + tuple(args[1:])
        return args, kwargs

    model.encoder.register_forward_pre_hook(enc_in, with_kwargs=True)
    model.ctc_decoder.register_forward_hook(lambda mod, i, out: out.float())
    return model


def wrap(opt, clip=None):
    model_params = [p for g in opt.param_groups for p in g["params"]]
    built = []

    def build():
        masters = []
        for g in opt.param_groups:
            ms = []
            for p in g["params"]:
                m = p.detach().float().clone()
                st = opt.state.pop(p, None)                      # a resumed state was loaded onto the bf16 params
                if st:
                    opt.state[m] = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() and v.dim() else v)
                                    for k, v in st.items()}
                ms.append(m)
            g["params"] = ms
            masters += ms
        built.append(masters)

    inner = opt.step

    def step(closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if not built:
            build()
        masters = built[0]
        live = [(m, p) for m, p in zip(masters, model_params) if p.grad is not None]
        for m, p in live:
            if m.grad is None:
                m.grad = torch.empty_like(m)
        torch._foreach_copy_([m.grad for m, _ in live], [p.grad for _, p in live])
        if clip:
            torch.nn.utils.clip_grad_norm_([m for m, _ in live], clip, foreach=True)
        inner()
        torch._foreach_copy_([p.data for _, p in live], [m for m, _ in live])
        return loss

    def zero_grad(set_to_none=True):
        for p in model_params:
            p.grad = None

    opt.step = step
    opt.zero_grad = zero_grad
    return opt
