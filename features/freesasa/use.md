# FreeSASA Batch Processing Usage Guide

## Required Environment

To run the FreeSASA batch processing scripts, you need to set up a conda environment with the following dependencies:

```yaml
name: freesasa-batch
channels:
  - conda-forge
  - bioconda
  - defaults
dependencies:
  - python=3.10
  - freesasa>=2.2
  - dssp
  - pip
```

### Installation Steps

1. Create the conda environment:
   ```bash
   conda env create -f environment.yml
   ```

2. Activate the environment:
   ```bash
   conda activate freesasa-batch
   ```

## Usage Instructions

### Linux Version (run_rsa_linux.py)

This script is designed for Linux systems and provides a simplified interface for batch processing.

#### Basic Usage

```bash
python run_rsa_linux.py start
```

This command will:
- Read protein IDs from `test.txt` in the `pdbtest` directory
- Process corresponding PDB files from the `pdb` subdirectory
- Output results to the `freesasa_output` directory
- Use 8 threads for parallel processing
- Display real-time progress monitoring

#### Directory Structure

Ensure your directory structure looks like this:

```
/teams/YingChiLab_1702378116/YuemingXiao/1upload_new/features/pdbtest/
├── test.txt          # List of protein IDs (one per line)
├── pdb/              # Directory containing PDB files
│   ├── protein1.pdb
│   ├── protein2.pdb
│   └── ...
└── freesasa_output/  # Output directory (created automatically)
```

#### Output Files

For each processed protein, the following files are generated:
- `{protein_id}_complete.rsa` - Complete SASA output for all residues
- `{protein_id}_atoms_complete.txt` - Atom-by-atom SASA details
- `{protein_id}_residues_complete.txt` - Residue-by-residue SASA summary
- `{protein_id}_statistics.txt` - Detailed processing statistics

### Batch Processor (batch_complete_processor.py)

This is the core processing script that can be customized for different input/output configurations.

#### Command Line Options

```bash
python batch_complete_processor.py [options]
```

#### Options

- `--id-file FILE`: ID list file (default: yourid.txt)
- `--pdb-dir DIR`: PDB files directory (default: pdb)
- `--output-dir DIR`: Output directory (default: youroutput)
- `--threads N`: Number of threads (default: 6)

#### Example Usage

```bash
# Basic usage with defaults
python batch_complete_processor.py

# Custom configuration
python batch_complete_processor.py \
  --id-file my_proteins.txt \
  --pdb-dir /path/to/pdb/files \
  --output-dir /path/to/output \
  --threads 12
```

## Processing Details

### Algorithm Parameters

- **Algorithm**: Lee-Richards
- **Probe Radius**: 1.4 Å
- **Atomic Radii**: NACCESS compatible
- **Output Format**: Complete (includes all atoms and residues, even those with SASA = 0)

### Features

- **Multi-threading**: Parallel processing for improved performance
- **Error Recovery**: Continues processing even if individual files fail
- **Progress Monitoring**: Real-time progress display
- **Resume Capability**: Can resume interrupted processing
- **Comprehensive Output**: Generates multiple output formats for each protein

### Error Handling

- Failed protein IDs are logged to `fail.txt` in the output directory
- Processing continues for remaining proteins even if some fail
- Detailed error messages are logged for troubleshooting

## Troubleshooting

### Common Issues

1. **"ID file not found"**: Ensure the ID file exists and path is correct
2. **"PDB directory not found"**: Verify the PDB directory exists and contains .pdb files
3. **ImportError: No module named 'freesasa'**: Activate the correct conda environment
4. **Processing fails for specific proteins**: Check PDB file format and content

### Log Files

- `batch_complete.log`: Main processing log
- `processing.log`: Additional processing details
- `fail.txt`: List of failed protein IDs

## Performance Notes

- Processing time varies based on protein size and complexity
- Typical processing speed: ~0.5-2 seconds per protein
- Memory usage scales with protein size
- Multi-threading provides significant speedup for large batches