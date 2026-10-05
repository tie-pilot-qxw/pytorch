# Workload shape survey

These are CPU-only scripts from 2026-09-17. For each real dataset, they ask how input shapes vary
from batch to batch. `bucket_cost.py` is the important one: it computes the cost curve of padding
into K buckets, which is the statistic that matters. Counting distinct shapes is not enough, because
padding to the maximum (K=1) needs no graph tricks at all. The write-up is in
`docs/notes/FINDINGS.md`, section "Workload survey: shape count is the **wrong** statistic".

Everything reads from `$DG_DATA` (default `/workspace/_deps/data`). The scripts also put
`$DG_DATA/pylibs` on `sys.path`, for packages installed with `pip install --target`. The directory
can be absent if numpy, pyarrow and tokenizers are importable otherwise.

| script | dataset | expected file(s) under `$DG_DATA` | how to get it |
|---|---|---|---|
| `bucket_cost.py` | proteome, KITTI | `human_proteome.fasta.gz`, `kitti/**/*.bin` | as below; a missing dataset is skipped |
| `prot.py` | UniProt human proteome | `human_proteome.fasta.gz` | `dl.sh` |
| `qm9.py` | QM9 | `qm9.parquet` | `dl.sh` |
| `gnn.py` | ogbn-arxiv | `arxiv_d/arxiv/raw/edge.csv.gz`, `arxiv_d/arxiv/split/time/train.csv.gz` | `dl.sh`, then `unzip arxiv.zip -d arxiv_d` |
| `coco.py`, `vlm.py` | COCO 2017 annotations | `coco_ann/annotations/instances_train2017.json` | `dl.sh`, then `unzip coco_ann.zip -d coco_ann` |
| `libri_dur.py` | LibriSpeech dev-clean | `libri/LibriSpeech/dev-clean/**/*.flac` | `dl.sh`, then `mkdir libri && tar xzf libri_devclean.tar.gz -C libri` |
| `md22.py` | MD22 (+ MD17 aspirin) | `md22_*.npz` or `MD22/md22_*.npz`; `md17_aspirin.npz` or `MD17/aspirin/raw/md17_aspirin.npz` | `dl2.sh` for MD22; MD17 aspirin as in the top-level README |
| `md17.py` | MD17 aspirin | `md17_aspirin.npz` | GDML repository (not in the download scripts) |
| `kitti.py` | KITTI raw drive 0093 | `kitti/**/velodyne_points/data/*.bin` | `dl2.sh`, then `unzip kitti_drive.zip -d kitti` |
| `ml.py` | MovieLens 25M | `ml/ml-25m/ratings.csv` | grouplens.org (not in the download scripts) |
| `tulu.py` | Tulu SFT mix, Qwen tokenizer | `tulu0.parquet`, `qwen_tok.json` | one parquet shard of the Tulu 3 SFT mixture and a Qwen `tokenizer.json` from Hugging Face (not in the download scripts) |
| `moe.py` | none (analytic simulation) | - | - |

The `e2e/` workloads use a different layout under the same `$DG_DATA` (see the top-level README).
The KITTI frames, the proteome and the MD22 files are shared between the two.
