# Morse Structured Encoding Experiments

This repository contains the code, benchmark scripts, and lightweight sample datasets for the paper's experiments on reversible structured encoding of Morse sequences and transfer tasks on digit and codon token data.

It implements:

- the paper-aligned base structured encoder
- the paper-aligned dynamic-programming variant
- a second transfer experiment on fixed-length BCD digit sequences
- a third transfer experiment on real CDS-derived codon token sequences
- raw Morse / raw binary baselines
- RLE, gzip, and zstd generic compression baselines
- two grammar-based baselines: a Sequitur-style digram-substitution encoder and a Re-Pair encoder
- ablations for grouping, alternating-pattern handling, and fallback
- failure-case extraction
- a "when structured encoding is worthwhile" analysis

## Layout

```text
repo/
├─ sequence_utils.py
├─ morse_utils.py
├─ binary_digit_utils.py
├─ grammar_codecs.py
├─ codon_utils.py
├─ structured_codon_codecs.py
├─ baseline_codecs.py
├─ benchmarks/
│  ├─ run_paper_experiments.py
│  ├─ run_exp2_binary_digit_experiments.py
│  ├─ prepare_exp3_real_codon_dataset.py
│  └─ run_exp3_codon_experiments.py
├─ manifests/
│  └─ real_cds_balanced_manifest.csv
├─ docs/
│  └─ reproduction.md
├─ datasets/
└─ output/
```

## Quick Start

```powershell
cd D:\Paper_owner\morsecode_simplify\repo
uv sync
uv run python benchmarks\run_paper_experiments.py
uv run python benchmarks\run_exp2_binary_digit_experiments.py
uv run python benchmarks\prepare_exp3_real_codon_dataset.py --data-root .\gene_dataset
uv run python benchmarks\run_exp3_codon_experiments.py
```

Useful options:

```powershell
uv run python benchmarks\run_paper_experiments.py --time-repeat 3
uv run python benchmarks\run_paper_experiments.py --group-limit single_word=200 --group-limit multi_word_phrase=200
uv run python benchmarks\run_exp2_binary_digit_experiments.py --time-repeat 3
uv run python benchmarks\prepare_exp3_real_codon_dataset.py --data-root .\gene_dataset --sample-size-per-species-length-bin 150 --sample-size-per-species-natural 1000 --sample-size-per-species-length-bin-short-mid 100
uv run python benchmarks\run_exp3_codon_experiments.py --input-tsv exp3_real_codon\sampled_balanced\overall.tsv --time-repeat 3
```

## Outputs

The main script writes CSV and Markdown files under `output/`:

- `overall_method_summary.csv`
- `group_method_summary.csv`
- `sample_method_details.csv`
- `base_vs_dp_summary.csv`
- `ablation_summary.csv`
- `failure_cases.csv`
- `worthwhile_by_run_coverage.csv`
- `worthwhile_by_alternating_share.csv`
- `paper_results_summary.md`

The Exp2 script writes its outputs under `output/exp2_binary_digits/`:

- `sample_method_details.csv`
- `group_method_summary.csv`
- `overall_method_summary.csv`
- `base_vs_dp_summary.csv`
- `exp2_results_summary.md`

The Exp3 dataset-preparation script writes intermediate TSVs under `exp3_real_codon/`:

- `raw/<species>_cds.fna`
- `cleaned/<species>_cleaned.tsv`
- `cleaned/<species>_cleaned_short_mid.tsv`
- `tokenized/<species>_tokens.tsv`
- `tokenized/<species>_tokens_short_mid.tsv`
- `tokenized/<species>_tokens_with_stop.tsv`
- `tokenized/length_group_counts_main_drop_stop.tsv`
- `tokenized/length_histogram_main_drop_stop.tsv`
- `tokenized/length_group_counts_short_mid_drop_stop.tsv`
- `tokenized/length_histogram_short_mid_drop_stop.tsv`
- `sampled_balanced/overall.tsv`
- `sampled_balanced/L1_short.tsv`
- `sampled_balanced/L2_medium.tsv`
- `sampled_balanced/L3_long.tsv`
- `sampled_balanced/L4_very_long.tsv`
- `sampled_balanced/high_regularity.tsv`
- `sampled_balanced/low_regularity.tsv`
- `sampled_natural/overall.tsv`
- `sampled_balanced_with_stop/overall.tsv`
- `sampled_short_mid_balanced/overall.tsv`
- `sampled_short_mid_balanced/L1_very_short.tsv`
- `sampled_short_mid_balanced/L2_short.tsv`
- `sampled_short_mid_balanced/L3_mid_short.tsv`
- `sampled_short_mid_balanced/L4_longer.tsv`

The Exp3 benchmark script writes its outputs under `exp3_real_codon/outputs/`:

- `main/sample_method_details.csv`
- `main/results_overall.csv`
- `main/results_groups.csv`
- `main/results_base_vs_dp.csv`
- `main/worthwhile_analysis.csv`
- `main/exp3_results_summary.md`
- `appendix_natural/natural_distribution_appendix.csv`
- `appendix_natural/natural_sample_method_details.csv`
- `appendix_short_mid/sample_method_details.csv`
- `appendix_short_mid/results_overall.csv`
- `appendix_short_mid/results_groups.csv`
- `appendix_short_mid/results_base_vs_dp.csv`
- `appendix_short_mid/worthwhile_analysis.csv`
- `appendix_short_mid/exp3_results_summary.md`

## Data Availability

This public repository includes the source code, experiment drivers, and small paper-aligned sample inputs needed to inspect and rerun the core pipeline. Large raw genome files, tokenized intermediates, and bulk generated outputs are not versioned in GitHub because they exceed practical repository size limits. Those artifacts can be regenerated with `benchmarks\prepare_exp3_real_codon_dataset.py` from the original downloaded sequence data placed under `gene_dataset/`.

## Paper Submission Snapshot

The paper-submission snapshot of this repository is fixed at release/tag `v1.0.1-paper-submission` (DOI: `10.5281/zenodo.19548067`) so that the code and documentation referenced in the manuscript point to a stable version.

## Primary Morse Benchmark Inputs

The four primary Morse benchmark files referenced in the paper correspond to the following repository paths:

- `standard_samples.txt` -> `datasets/base/standard_samples.txt`
- `long_sentence_samples.txt` -> `datasets/long/long_sentence_samples.txt`
- `paragraph_samples.txt` -> `datasets/paragraph/paragraph_samples.txt`
- `long_text_samples.txt` -> `datasets/long_text/long_text_samples.txt`

## Reproducing The Real-CDS Benchmark

The real-CDS benchmark used in the main paper is derived from public NCBI CDS FASTA resources for three species: `ecoli` (*E. coli*), `human` (*H. sapiens*), and `scerevisiae` (*S. cerevisiae*).

Because the full raw `gene_dataset` files are too large to distribute directly in this repository, we provide:

- accession manifests for the balanced benchmark
- preprocessing scripts
- benchmark-construction scripts
- a small example subset for workflow validation

The file `manifests/real_cds_balanced_manifest.csv` records the per-CDS membership of the main paper's balanced benchmark.

The main benchmark can be reconstructed by:

1. obtaining the corresponding CDS FASTA records from NCBI using the accession list in `manifests/real_cds_balanced_manifest.csv`
2. removing terminal stop codons
3. converting the sequences into codon-token sequences
4. applying the length-stratified balanced benchmark construction implemented in `benchmarks\prepare_exp3_real_codon_dataset.py`

The exact reconstruction workflow used in this study is documented in `docs/reproduction.md`.

## Notes

- This repo now uses `uv` as the dependency and execution workflow.
- `zstandard` is a required dependency, so the `zstd` baseline is part of the default experiment setup.
- The grammar-based baselines are implemented in-repo and require no extra packages.
- All structured encoders in this repo use the paper protocol `ID\ID\...%RS` without the later compact-token and repeated-word-reference optimizations from `MorseFold-AI`.
- The real-codon Exp3 uses public CDS FASTA files placed under `gene_dataset/` and evaluates the printable codon-token string representation after terminal-stop removal by default.
