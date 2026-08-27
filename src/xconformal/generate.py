"""ArchesWeatherGen inference: model loading, batch construction, t2m extraction.

Kept out of scripts/01_generate.py so that the benchmark and the smoke test
import the same code paths the production run uses. Nothing here decides
anything; all parameters come from config.

CWD. The archesweathergen config lists its four deterministic backbones as
RELATIVE paths (``modelstore/archesweather-m-seed0`` and friends), and
``geoarches.lightning_modules.load_module`` resolves them against the current
working directory. The repo carries a ``modelstore`` symlink to ``$MODEL_ROOT``
for exactly this reason, so :func:`load_model` chdirs to the repo root before
loading and restores the caller's CWD afterwards.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from . import config

__all__ = [
    "REPO_ROOT",
    "load_model",
    "build_dataset",
    "load_batch",
    "member_seed",
    "extract_t2m",
    "sample_members",
]

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_model(device: str = "auto"):
    """Load the ArchesWeatherGen lightning module from config.GEN_MODEL_DIR.

    in: device ("auto"/"cuda"/"cpu"); out: (module, cfg).

    geoarches is imported lazily: it drags in torch and lightning, and the CLI
    --help paths must stay fast and importable without a GPU.
    """
    from geoarches.lightning_modules import load_module

    previous = Path.cwd()
    os.chdir(REPO_ROOT)
    try:
        symlink = REPO_ROOT / "modelstore"
        if not symlink.exists():
            raise FileNotFoundError(
                f"{symlink} is missing. The model config references its deterministic "
                f"backbones as relative 'modelstore/...' paths, so the repo needs that "
                f"symlink: ln -s {config.MODEL_ROOT} {symlink}"
            )
        module, cfg = load_module(str(config.GEN_MODEL_DIR), device=device)
    finally:
        os.chdir(previous)

    # This config instantiates DiffusionModule, whose det_model is a single
    # AvgModule wrapping the four deterministic backbones (EnsembleDiffusionModule
    # would instead expose det_models, a ModuleList). Accept either, but insist
    # the right NUMBER of backbones is live: load_module restores relative
    # 'modelstore/...' paths silently, so a wrong CWD yields an uninitialised
    # backbone and garbage samples rather than an exception.
    expected = len(cfg.module.module.load_deterministic_model)
    det = getattr(module, "det_model", None)
    n_det = (
        len(det.core) if det is not None and hasattr(det, "core")
        else len(getattr(module, "det_models", []) or [])
    )
    if n_det != expected:
        raise RuntimeError(
            f"config lists {expected} deterministic models but {n_det} are live; "
            f"sampling would run on uninitialised backbones"
        )
    return module, cfg


def build_dataset(years: tuple[int, ...] = (config.CALIBRATION_YEAR, config.VERIFICATION_YEAR)):
    """Build one Era5Forecast over the 0Z files for the given years.

    in: years; out: Era5Forecast. EXPENSIVE -- it opens every matching netcdf on
    construction, so build it once and reuse it across inits.

    Constructed with an explicit filename_filter rather than a domain string:
    Era5Forecast silently re-selects a single hard-coded year when domain is
    "val" or "test", which would quietly drop half the window.
    """
    from geoarches.dataloaders.era5 import Era5Forecast

    tags = tuple(str(y) for y in years)
    hour_tag = f"_{config.INIT_HOUR}h"

    def filename_filter(name: str) -> bool:
        return any(t in name for t in tags) and hour_tag in name

    return Era5Forecast(
        path=str(config.ERA5_FULL_DIR),
        filename_filter=filename_filter,
        timedelta_hours=24,      # consecutive 0Z files are 24h apart
        lead_time_hours=config.LEAD_TIME_HOURS,
        multistep=1,             # 0 would zero out lead_time_hours in the batch
        load_prev=True,          # the model is conditioned on state and prev_state
        norm_scheme="pangu",
    )


def _init_index(dataset) -> dict:
    """Map init datetime64 -> dataset index, accounting for the load_prev shift."""
    offset = int(dataset.load_prev) * config.LEAD_TIME_HOURS // dataset.timedelta
    out = {}
    for i in range(len(dataset)):
        stamp = dataset.id2pt[i + offset][-1].astype("datetime64[h]")
        out[stamp] = i
    return out


def init_dates(dataset) -> list:
    """Every init date this dataset can serve, ascending."""
    if not hasattr(dataset, "_xconformal_index"):
        dataset._xconformal_index = _init_index(dataset)
    return sorted(dataset._xconformal_index)


def load_batch(init: np.datetime64, dataset=None, device: str = "auto"):
    """Build the model input batch for one init date.

    in: init datetime64, optional prebuilt dataset, device;
    out: (batch, dataset) where batch has state / prev_state / timestamp /
    lead_time_hours with a leading batch dimension of 1, on the device.
    """
    from geoarches.evaluation.eval_multistep import _custom_collate_fn

    if dataset is None:
        dataset = build_dataset()
    if not hasattr(dataset, "_xconformal_index"):
        dataset._xconformal_index = _init_index(dataset)

    key = np.datetime64(init, "h")
    if key not in dataset._xconformal_index:
        available = sorted(dataset._xconformal_index)
        raise KeyError(
            f"init {key} not in the dataset; it spans {available[0]} .. {available[-1]}"
        )
    batch = _custom_collate_fn([dataset[dataset._xconformal_index[key]]])

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = {
        k: (v.to(device) if hasattr(v, "to") else v)
        for k, v in batch.items()
        if "next" not in k and "future" not in k
    }
    # geoarches' collate builds each TensorDict without a batch_size, so it
    # comes back as torch.Size([]). The backbone reads state.shape[0] to get
    # the batch dimension and raises IndexError on an empty shape, so stamp the
    # batch size on explicitly.
    return _set_batch_size(batch, 1), dataset


def _set_batch_size(batch: dict, n: int) -> dict:
    """Stamp a leading batch dimension of n onto every TensorDict in a batch."""
    for value in batch.values():
        if hasattr(value, "batch_size"):
            value.batch_size = [n]
    return batch


def init_batch_nb(init: np.datetime64) -> int:
    """Reproducible batch_nb for one init date: days since 1970-01-01.

    geoarches composes the per-step seed itself, as
    ``seed_i = member + 1000 * i + batch_nb * 10**6``. Feeding it a
    DATE-derived batch_nb (rather than a position in the to-do list) is what
    makes a resumed run bit-identical to an uninterrupted one: an init
    regenerated after a preemption draws exactly the noise it would have drawn
    the first time, whatever order the driver happens to visit inits in.

    The composition is collision-free for our grid: members are < 1000 and
    rollout steps are < 1000, so distinct (init, member, step) triples map to
    distinct seeds.
    """
    return int(np.datetime64(init, "D").astype("int64"))


def member_seed(init: np.datetime64, member: int, step: int = 0) -> int:
    """The seed geoarches will actually use for (init, member, rollout step)."""
    return int(member) + 1000 * int(step) + init_batch_nb(init) * 10**6


def extract_t2m(rollout, dataset) -> np.ndarray:
    """Denormalize a rollout and pull out 2m temperature.

    in: rollout TensorDict (batch, iterations, var, lev, lat, lon) from
        sample_rollout, plus the dataset that normalized the inputs;
    out: float32 ndarray (batch, iterations, lat, lon) in Kelvin.
    """
    denorm = dataset.denormalize(rollout)
    surface = denorm["surface"]  # (batch, iterations, var, lev, lat, lon)
    t2m = surface[..., config.TARGET_SURFACE_INDEX, 0, :, :]
    return t2m.detach().to(torch.float32).cpu().numpy()


def replicate(batch: dict, n: int) -> dict:
    """Replicate a batch-of-1 along the batch dimension n times.

    in: batch with leading dim 1, n; out: a new batch with leading dim n.

    Every leading-dim-1 tensor is expanded, including the scalar-ish
    ``timestamp`` and ``lead_time_hours``: the conditioning embedder indexes
    them per batch element, so leaving them at length 1 silently conditions
    every replicated member on element 0's time.
    """
    out = {}
    for key, value in batch.items():
        if hasattr(value, "batch_size"):          # TensorDict
            out[key] = value.expand(n, *value.batch_size[1:]).clone()
            out[key].batch_size = [n]
        elif hasattr(value, "repeat") and getattr(value, "ndim", 0) >= 1:
            out[key] = value.repeat(n, *([1] * (value.ndim - 1)))
        else:
            out[key] = value
    return out


def sample_members(module, batch, dataset, init, members, batch_size: int = 1) -> np.ndarray:
    """Sample the given members for one init and return physical t2m.

    in: module, batch (leading dim 1), dataset, init, member indices, batch_size;
    out: float32 ndarray (n_members, n_leads, lat, lon) in Kelvin.

    batch_size > 1 replicates the conditioning state so several members denoise
    in one pass. NOTE: geoarches draws the initial noise for the whole batch
    from ONE generator in a single stream, so a member's noise depends on how
    many members share its pass. Batched results are reproducible for a fixed
    batch_size, but are NOT bit-identical to the serial run. Keep batch_size
    pinned in config for any archive you intend to resume.
    """
    chunks = [members[i : i + batch_size] for i in range(0, len(members), batch_size)]
    out = []
    for chunk in chunks:
        wide = batch if len(chunk) == 1 else replicate(batch, len(chunk))
        rollout = module.sample_rollout(
            wide,
            batch_nb=init_batch_nb(init),
            iterations=config.ROLLOUT_ITERATIONS,
            member=chunk[0],
            disable_tqdm=True,
        )
        out.append(extract_t2m(rollout, dataset))
    return np.concatenate(out, axis=0)
