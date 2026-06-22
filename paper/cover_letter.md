# Cover letter

To the Editors, *npj Computational Materials*

Dear Editors,

Please consider our manuscript, **"Building and validating a GPU micromagnetic simulator with an AI coding
agent: cross-implementation testing exposes a concurrency defect,"** for publication as an Article in
*npj Computational Materials*.

AI coding agents are beginning to write substantial scientific software, yet the field lacks concrete,
quantified evidence on two questions of direct relevance to computational materials science: can such an agent
build a *validated, performant* simulator rather than a toy, and what verification methodology makes the result
trustworthy? Our work answers both with a complete, reproducible case study.

We report Claude-SpinDynamics, a C++20/CUDA micromagnetic simulator with a Python interface, developed
end-to-end with an AI coding agent under a specification-and-test discipline. The paper makes three
contributions that we believe fit the journal's scope and standards:

1. **A reproducible, quantified agent-development workflow** for research-grade GPU simulation software —
   specification-driven, test-driven, and verified across four build variants — with metrics mined from the
   project history.

2. **Rigorous validation and competitive performance.** Claude-SD reproduces the µMAG standard problems and
   agrees with mumax3, mumax⁺, MuMax-CO and an OOMMF double-precision anchor to within their mutual spread; its
   float32 build is up to 5× faster than mumax3 per field-evaluation on small/2-D problems, and it uniquely
   among GPU micromagnetic codes offers double precision and a choice of two FFT backends.

3. **A methodological result we believe is the paper's most important.** The simulator's deliberately
   redundant, multi-precision/multi-backend/multi-solver design enabled a cross-implementation comparison that
   *exposed a real GPU concurrency defect* — a stream-synchronization race that randomized topological charge
   near the skyrmion metastability boundary — invisible to any single build and to the standard problems. We
   diagnosed it by bisection against an independent solver's deterministic behaviour and fixed it. The lesson
   generalizes: because the characteristic failure of agent-written code is plausible-looking code that is
   subtly wrong, cross-implementation validation should be a first-class design goal, not an afterthought.

The work is timely for the *npj Computational Materials* readership: it sits at the intersection of
AI-for-science methodology and GPU materials simulation, advances both an open tool and a verification
practice, and is fully reproducible — the source, benchmark suite, and all data regenerate every table and
figure from a single command, and will be archived with a citable DOI.

This manuscript is original, has not been published previously, and is not under consideration elsewhere. The
author declares no competing interests. We have no objection to, and can suggest if helpful, reviewers with
expertise in micromagnetic simulation (mumax3/mumax⁺/OOMMF developers), GPU numerical methods, and AI-assisted
software engineering.

Thank you for your consideration.

Sincerely,
Chun-Yeol You
Department of Physics and Chemistry, DGIST, Daegu, Republic of Korea
cyyou@dgist.ac.kr
