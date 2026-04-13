# Reproduction Workflow

## Real-CDS Benchmark

The main real-CDS benchmark is built from public NCBI CDS FASTA resources for three species:

- `ecoli` (`Escherichia coli`)
- `human` (`Homo sapiens`)
- `scerevisiae` (`Saccharomyces cerevisiae`)

The balanced benchmark membership used in the paper is listed in `manifests/real_cds_balanced_manifest.csv`.

## Reconstruction Steps

1. Obtain the corresponding NCBI CDS FASTA resources for the three species listed above.
2. Place the downloaded NCBI datasets under `gene_dataset/` using the same per-species directory structure expected by `benchmarks\prepare_exp3_real_codon_dataset.py`.
3. Run:

```powershell
uv run python benchmarks\prepare_exp3_real_codon_dataset.py --data-root .\gene_dataset
```

This preparation script:

- stages raw CDS FASTA files under `exp3_real_codon/raw/`
- filters valid CDS sequences
- removes terminal stop codons for the default main benchmark
- converts CDS sequences into codon-token sequences
- assigns each sequence to the paper's length strata
- constructs the balanced benchmark TSVs under `exp3_real_codon/sampled_balanced/`

## Main Benchmark File

The main paper benchmark is the balanced codon-token TSV:

`exp3_real_codon/sampled_balanced/overall.tsv`

Its rows correspond to the accessions listed in `manifests/real_cds_balanced_manifest.csv`.

## Running The Benchmark

After reconstruction, run:

```powershell
uv run python benchmarks\run_exp3_codon_experiments.py --input-tsv exp3_real_codon\sampled_balanced\overall.tsv
```

This produces the main Exp3 result tables under `exp3_real_codon/outputs/main/`.
