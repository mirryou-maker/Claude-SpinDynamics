# Contributors

Claude-SpinDynamics (Claude-SD) is an open-source GPU micromagnetic simulator
developed through human–AI collaboration.

## Authors

| Contributor | Role |
|---|---|
| **Chun-Yeol You** (DGIST) | Author, physics direction, validation, benchmarking, project lead |
| **Claude Code** — Anthropic (Claude Opus) | AI pair-programmer: kernel/class implementation, tests, GPU/CPU parity, benchmarks, documentation |

## How Claude contributed

The simulator was built end-to-end with [Claude Code](https://claude.com/claude-code) under a
specification-and-test workflow described in the project's `CLAUDE.md`:

- implemented C++/CUDA effective fields, spin torques, and integrators behind shared interfaces,
  each with a CPU reference and a Catch2 unit test;
- maintained agreement across four CUDA build variants (f32/f64 × cuFFT/VkFFT);
- ran and analyzed the µMAG standard problems and the cross-solver benchmark campaign;
- diagnosed and fixed a GPU stream-synchronization race exposed by cross-validation.

Commits made with Claude carry a trailer:

```
Co-Authored-By: Claude <noreply@anthropic.com>
```

> Note on the GitHub "Contributors" graph: that graph is generated automatically from commit
> author/co-author emails that map to GitHub user accounts. The `noreply@anthropic.com` co-author
> email is not a GitHub account, so Claude's contribution is credited here and in the commit
> trailers rather than in the auto-generated graph.

## Contributing

Issues and pull requests are welcome. New code should follow the conventions in `CLAUDE.md`
(SI units, the `IEffectiveField`/`ISpinTorque` interfaces) and ship with a test that passes on all
four CUDA builds.
