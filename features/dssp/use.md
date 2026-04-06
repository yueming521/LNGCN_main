# DSSP Batch Processor (Linux) - Usage Guide

This guide explains how to run `robust_batch_process_linux.py` on Linux.

## 1) Environment Setup

Use conda (recommended), because `mkdssp` is **not** a pip package.

```yaml
name: dssp-batch
channels:
  - conda-forge
  - bioconda
  - defaults
dependencies:
  - python=3.10
  - dssp
  - pip
```

Create and activate:

```bash
conda env create -f environment.yml
conda activate dssp-batch
```

If you already have an environment (for example `myconda`):

```bash
conda install -n myconda -c conda-forge -c bioconda dssp -y
```

Verify command availability:

```bash
which mkdssp
mkdssp --version
```

---

## 2) Input/Output Path Configuration

Edit these variables in `robust_batch_process_linux.py` inside `main()`:

- `protein_ids_file`: plain text file, one protein ID per line
- `pdb_dir`: directory containing `{protein_id}.pdb`
- `output_base_dir`: output root directory
- `max_workers`: thread count
- `force_reprocess`: whether to re-check and regenerate missing outputs

Current defaults:

```python
protein_ids_file = "/teams/YingChiLab_1702378116/YuemingXiao/1upload_new/features/pdbtest/test.txt"
pdb_dir = "/teams/YingChiLab_1702378116/YuemingXiao/1upload_new/features/pdbtest/pdb"
output_base_dir = "/teams/YingChiLab_1702378116/YuemingXiao/1upload_new/features/pdbtest/dssp"
max_workers = 4
force_reprocess = True
```

Expected structure:

```text
pdbtest/
├── test.txt
├── pdb/
│   ├── A0A1B0GTZ2.pdb
│   ├── A0AVF1.pdb
│   └── ...
└── dssp/
```

---

## 3) Run the Script

From any directory:

```bash
python /teams/YingChiLab_1702378116/YuemingXiao/1upload_new/features/dssp/robust_batch_process_linux.py
```

Or from project root:

```bash
python 1upload_new/features/dssp/robust_batch_process_linux.py
```

---

## 4) Outputs

The script writes three output groups under `output_base_dir`:

- `DSSP/` → `{protein_id}.dssp`
- `standard_mmCIF/` → `{protein_id}_standard.cif`
- `experimental_mmCIF/` → `{protein_id}_experimental.cif`

State and logs:

- `processing_state.json`
- `processing.log`

Failure reports:

- `183failed_dssp.txt`
- `183failed_standard_mmcif.txt`
- `183failed_experimental_mmcif.txt`

---

## 5) Notes

- `mkdssp` must be in `PATH`.
- If interrupted, the script saves progress state and can continue.
- `force_reprocess=True` will process proteins that are missing **any** required output file.
