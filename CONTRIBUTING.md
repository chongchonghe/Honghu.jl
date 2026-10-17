# Contributing to Honghu.jl

Thank you for your interest in Honghu. Bug reports, questions, suggestions and code are all welcome. Everyone taking part is expected to follow the [code of conduct](CODE_OF_CONDUCT.md).

## Asking questions and reporting problems

Use the [issue tracker](https://github.com/chongchonghe/Honghu.jl/issues). It has templates for bug reports, feature requests and questions.

A good bug report has:

- the Julia version (`versioninfo()`) and the Honghu version or commit;
- the smallest script that shows the problem, ideally built from one of the problems in `Honghu.Problems`;
- what you expected, and what happened instead, with the full error message if there is one.

For a numerical problem, such as a wrong answer, a crash deep in a run or a loss of conservation, say which problem, resolution and time step it appears at, and whether it changes with resolution or with the reconstruction and Riemann solver.

## Proposing a change

1. For anything beyond a small fix, open an issue first to discuss the idea.
2. Fork the repository and make your change on a branch.
3. Run the test suite and, if you changed docstrings or documentation, the documentation build (see below).
4. Open a pull request that refers to the issue. CI runs the test suite on Julia 1.10 and 1.12 and builds the documentation site; both must pass.

What a change is expected to include:

- **A test that would fail without it.** For physics, prefer invariants that must hold exactly (conservation on periodic boundaries, direction isotropy, positivity without floor repairs) over thresholds tuned to the current scheme. The section "Testing changes" in `CLAUDE.md` describes the patterns the suite uses.
- **No allocation in hot loops.** `test/test_performance.jl` checks that every hot kernel allocates almost nothing per cell. Extend it if you add a kernel.
- **A docstring for every exported name**, saying what it is for. Cross-references are written as ``[`name`](@ref)``, and the documentation build fails if `name` has no docstring.
- **Matching style.** Follow the code around the change, and keep comments to the reasons that are not obvious from the code. `CLAUDE.md` describes the layout of the source, the invariants the code relies on and the performance rules.

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `ci:`, `chore:`).

## Running the tests

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # once
julia --project=. -t 4 test/runtests.jl               # the full suite, about 80 s
```

The suite is run directly rather than through `Pkg.test`, because `Pkg.test` turns bounds checking on, and that changes what the allocation tests measure.

## Building the documentation

The site is built from the docstrings, the README and the notes in `docs/` with [Quarto](https://quarto.org/) 1.9 or newer:

```bash
julia --project=. docs/make.jl                  # full build into docs/_site
julia --project=. docs/make.jl --generate-only  # write the pages only, then `quarto preview docs`
```

The build is strict: an unresolved ``[`name`](@ref)``, a source file with docstrings but no reference page, or a broken link fails it. [How this documentation is built](https://chongchonghe.github.io/Honghu.jl/documentation.html) explains the system and how to update each kind of page.

## License

Honghu is released under the [MIT License](LICENSE). By contributing, you agree that your contributions are released under the same license.
