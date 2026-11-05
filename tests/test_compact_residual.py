"""The compact residual stream ("compact_residual", transformer_flux2): the
blocks run on the active rows only instead of carrying both streams
full-length. A tiny random FLUX.2 transformer runs the same frames on two
spatial caches, one per path; outputs and every cache tensor must match
exactly (on the CPU every op computes a row the same way whatever the other
rows are). On a GPU the compiled kernels may round differently; that is what
scripts/perf_ab.py checks."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from fluxrt.stream_processor import transformer_flux2 as tf  # noqa: E402
from fluxrt.stream_processor.int8_linear import quantize_transformer_blocks  # noqa: E402

TXT, IMG, CH = 8, 24, 16  # text tokens, image tokens (noisy + condition), latent channels
HEADS, HEAD_DIM, DOUBLE, SINGLE = 2, 16, 2, 2
BUCKET = 4  # small, so padded (repeated) rows show up


def tiny_model(dtype, int8=False):
    torch.manual_seed(0)
    model = tf.Flux2Transformer2DModel(
        in_channels=CH,
        num_layers=DOUBLE,
        num_single_layers=SINGLE,
        attention_head_dim=HEAD_DIM,
        num_attention_heads=HEADS,
        joint_attention_dim=24,
        timestep_guidance_channels=32,
        axes_dims_rope=(4, 4, 4, 4),
        guidance_embeds=False,
    )
    # Random weights are small; scale the modulation so the residual stream
    # actually moves (a gate near zero would hide a wrong row).
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(4.0)
    if int8:
        quantize_transformer_blocks(model)
    return model.to(dtype).eval()


def new_cache(dtype):
    return tf.SpatialCache(
        image_seq_len=IMG,
        text_seq_len=TXT,
        output_channels=CH,
        attention_head_dim=HEAD_DIM,
        num_attention_heads=HEADS,
        num_layers=DOUBLE,
        num_single_layers=SINGLE,
        device="cpu",
        dtype=dtype,
    )


def masks():
    """(name, full-sequence mask) per frame, as the pipeline's change mask
    would set them: 2 = execute and update, 0 = take from the cache."""

    def mask(txt=False, img=()):
        m = torch.zeros(TXT + IMG, dtype=torch.int32)
        if txt:
            m[:TXT] = 2
        for i in img:
            m[TXT + i] = 2
        return m

    yield "first frame (everything)", mask(True, range(IMG))
    yield "nothing changed", mask()
    yield "7 image rows (padded to 8)", mask(False, range(3, 10))
    yield "other image rows", mask(False, [0, 5, 11, 12, 13, 23])
    yield "prompt changed + image rows", mask(True, range(4))
    yield "prompt changed, image still", mask(True)
    yield "one image row", mask(False, [17])
    exec_only = mask(False, range(6))
    exec_only[TXT + 6 : TXT + 9] = 1  # manual mask: execute, don't update
    yield "execute-only rows (falls back)", exec_only


def run_frames(model, dtype, compact):
    cache = new_cache(dtype)
    g = torch.Generator().manual_seed(1)
    img_ids = torch.randint(0, 6, (1, IMG, 4), generator=g)
    txt_ids = torch.randint(0, 6, (1, TXT, 4), generator=g)
    timestep = torch.tensor([0.75])
    outputs = []
    for name, m in masks():
        hidden = torch.randn(1, IMG, CH, generator=g).to(dtype)
        text = torch.randn(1, TXT, 24, generator=g).to(dtype)
        sparse = tf.SparseMask(cache.preprocess_mask(m.unsqueeze(0)), TXT, BUCKET)
        with torch.no_grad():
            out = model(
                hidden_states=hidden,
                encoder_hidden_states=text,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=None,
                return_dict=False,
                mask=sparse,
                spatial_cache=cache,
                compact_residual=compact,
            )[0]
        outputs.append((name, out.clone()))
    return outputs, cache


def cache_tensors(cache):
    yield "output_cache", cache.output_cache
    yield "valid", cache.valid
    for kind in ("double", "single"):
        for i in range(DOUBLE if kind == "double" else SINGLE):
            keys, values = cache.kv_tensors(i, kind)
            yield f"{kind}[{i}].keys", keys
            yield f"{kind}[{i}].values", values


@pytest.mark.parametrize(
    "dtype,int8",
    [(torch.float32, False), (torch.bfloat16, False), (torch.bfloat16, True)],
    ids=["fp32", "bf16", "bf16-int8"],
)
def test_compact_residual_is_bit_identical(dtype, int8):
    model = tiny_model(dtype, int8)
    full, full_cache = run_frames(model, dtype, compact=False)
    compact, compact_cache = run_frames(model, dtype, compact=True)
    for (name, a), (_, b) in zip(full, compact):
        assert torch.isfinite(a).all(), name
        assert torch.equal(a, b), f"{name}: max diff {(a.float() - b.float()).abs().max()}"
    for (name, a), (_, b) in zip(cache_tensors(full_cache), cache_tensors(compact_cache)):
        assert torch.equal(a, b), name


def test_the_frames_exercise_the_cache():
    """The comparison above means something only if frames differ from each
    other and the cached rows are actually reused."""
    model = tiny_model(torch.float32)
    outputs, _ = run_frames(model, torch.float32, compact=True)
    by_name = dict(outputs)
    first, nothing = by_name["first frame (everything)"], by_name["nothing changed"]
    assert torch.equal(first, nothing)  # nothing changed: all rows from the cache
    seven = by_name["7 image rows (padded to 8)"]
    changed = (seven != nothing).any(-1).squeeze(0)
    assert changed[3:10].all() and not changed[:3].any() and not changed[10:].any()


def test_the_callers_kwargs_are_left_alone():
    """The compact path must not leave its "compact" flag in the caller's
    joint_attention_kwargs (the pipeline passes the same dict to other calls)."""
    model = tiny_model(torch.float32)
    cache = new_cache(torch.float32)
    kwargs = {}
    m = torch.full((1, TXT + IMG), 2, dtype=torch.int32)
    with torch.no_grad():
        model(
            hidden_states=torch.randn(1, IMG, CH),
            encoder_hidden_states=torch.randn(1, TXT, 24),
            timestep=torch.tensor([0.5]),
            img_ids=torch.zeros(1, IMG, 4),
            txt_ids=torch.zeros(1, TXT, 4),
            joint_attention_kwargs=kwargs,
            return_dict=False,
            mask=tf.SparseMask(cache.preprocess_mask(m), TXT, BUCKET),
            spatial_cache=cache,
            compact_residual=True,
        )
    assert "compact" not in kwargs
