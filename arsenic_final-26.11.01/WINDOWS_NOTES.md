# Running on Windows

Two Windows-specific problems are already fixed in the shipped scripts. This
note explains them in case you hit a variant.

## 1. joblib / loky crash in `_count_physical_cores`

```
File "...joblib\externals\loky\backend\context.py", line 257, in _count_physical_cores
    cpu_info = subprocess.run(...)
File "...subprocess.py", line 1538, in _execute_child
    hp, ht, pid, tid = _winapi.CreateProcess(...)
```

**Cause.** joblib counts physical cores by shelling out to `wmic`, which
Microsoft removed in recent Windows 11 builds. The subprocess call then fails.

**Fix (already applied).** Both `arsenic_pipeline.py` and `final_system.py`
set these before scikit-learn is imported, which skips the probe:

```python
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
```

All random forests also run with `n_jobs=1`, so no worker pool is created.

**If you still see it** (for example in your own notebook), set the variable
in the shell before launching Python:

```bat
set LOKY_MAX_CPU_COUNT=4
python arsenic_pipeline.py
```

The order matters: the variable must be set before `sklearn` is imported, so
putting it halfway down a notebook will not help. Restart the kernel.

**Results are unaffected.** Verified after the change: DCRP As(III) Day-20
ROC-AUC 0.954, identical to the Linux run.

## 2. `Axes.boxplot() got an unexpected keyword argument 'tick_labels'`

**Cause.** `tick_labels` was added in matplotlib 3.9. Older versions call the
argument `labels`, and 3.11 removed `labels` again, so neither name is safe
across versions.

**Fix (already applied).** `fig8_mixup` now sets the labels after drawing:

```python
bp = ax.boxplot(data, widths=.55, patch_artist=True)
ax.set_xticks(range(1, len(short) + 1))
ax.set_xticklabels(short)
```

This works on every matplotlib version. The rest of the plotting code was
audited at the same time and uses no API newer than matplotlib 3.1.

## 3. Hardcoded data path

`--uploads` no longer defaults to a Linux path. The script searches, in order:

1. the folder given to `--uploads`
2. `data\` beside the script
3. the script's own folder
4. `.\data\` and the current directory

Filenames are matched by glob, so `as3___raw_data.xlsx` and
`as3_raw_data.xlsx` both work. The folder actually used is printed at startup.

## Quick check

```bat
cd C:\Users\romeo\dcrp_toxicity_model
python arsenic_pipeline.py
```

Expect `data: C:\Users\romeo\dcrp_toxicity_model\data` as the first line.

## If you want zero dependencies

`dcrp_system.py` embeds the dataset and the trained model and uses only the
standard library for prediction. It cannot hit either problem above:

```bat
python dcrp_system.py info
python dcrp_system.py demo
```
