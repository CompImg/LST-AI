# How LST-AI installs and uses FastSurfer

Lesions are assigned to anatomical regions using a [FastSurfer](https://github.com/Deep-MI/FastSurfer)
segmentation of the T1. This is not an optional stage: every mode except `--segment_only`
runs it, the default one included, so FastSurfer is a hard requirement of a normal `lst`
run — and both the pip install and the Docker image bring it with them. There is nothing
to install by hand. This page documents how that works and the few knobs that exist.

## What the install does

FastSurfer is not published on PyPI, so `pip install` brings it in two parts, identically
on x86_64 and on arm64/aarch64:

- the Python dependencies of the segmentation path LST-AI drives are ordinary
  requirements in `setup.py`, resolved by pip like `greedy` and HD-BET;
- the FastSurfer source tree itself (~17 MB) is unpacked from the pinned release tarball
  into `<your environment>/share/lst-ai/FastSurfer-v2.5.4` — or into
  `~/.local/share/lst-ai/` when that prefix is not writable.

LST-AI finds it there on its own: no `FASTSURFER_HOME`, no `PATH` entry, nothing to add to
your `~/.bashrc`. The three VINN checkpoints (~65 MB) are fetched by FastSurfer on the
first run that annotates, the same way LST-AI fetches its own model bundle.

**This copy is the only one LST-AI uses.** A FastSurfer already on the machine — one
`FASTSURFER_HOME` points at, or a `run_fastsurfer.sh` on `PATH` — is ignored, and there is
no option to prefer it, so an annotation never depends on what a given machine happens to
have lying around. The version is pinned in
[../lst_ai/fastsurfer.py](../lst_ai/fastsurfer.py).

## Manual controls

Two things you can still do by hand:

```bash
# fetch the checkpoints ahead of time, for a machine that will later be offline
python -m lst_ai.fastsurfer --checkpoints

# re-download the tree, if it was interrupted or something under it was edited
python -m lst_ai.fastsurfer --force
```

Setting `LST_AI_SKIP_FASTSURFER=1` before `pip install` skips the download at install
time — it defers it to the first run that annotates, it does not substitute another
FastSurfer. A failed download never fails the install, for the same reason:
`--segment_only` still works, and the download is retried on first use.

## How it is invoked

LST-AI shells out to `run_fastsurfer.sh` with
`--seg_only --no_cereb --no_hypothal --no_cc --no_biasfield`: only the FastSurferVINN
whole-brain segmentation runs, not the surface pipeline, and the modules whose extra
dependencies and checkpoints LST-AI does not need stay off. When LST-AI itself runs as
root (a plain `docker run`), it passes FastSurfer's `--allow_root`, and drops it
otherwise — see the Docker notes in the [README](../README.md) about running as yourself.
