# Development notes

Records of investigations and measurements that stay in the repository but are not pages on the documentation site: what was tried, what it bought, and the scripts that reproduce the numbers. A note that should be on the site belongs in `docs/` instead, listed in the `NOTES` of `docs/make.jl`.

- `performance-findings.md`: single-core optimisation of the cell kernels (vectorisation, subnormals, `StaticRows`, the ARM exception), and the comparison with Athena++. Its scripts are in `performance-experiments/`.
