---
name: Bug report
about: Something fails, crashes or gives a wrong answer
labels: bug
---

**What happened**

A clear description of the problem, and the full error message if there is one.

**How to reproduce**

The smallest script that shows it, ideally built from one of the problems in `Honghu.Problems`:

```julia
using Honghu

```

**What you expected**

For a numerical problem, say which problem, resolution and time step it appears at, and whether it changes with resolution or with the reconstruction and Riemann solver.

**Versions**

- Julia: (output of `versioninfo()`)
- Honghu: (version or commit)
