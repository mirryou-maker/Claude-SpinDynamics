"""Build a talk deck (16:9) from the Claude-SD paper + figures."""
import pathlib
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

HERE = pathlib.Path(__file__).parent
FIG = HERE / "figures"

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
BLUE = RGBColor(0x1F, 0x77, 0xB4)
GREY = RGBColor(0x44, 0x44, 0x44)
RED = RGBColor(0xC0, 0x2A, 0x2A)
LIGHT = RGBColor(0xF2, 0xF5, 0xF8)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = prs.slide_width, prs.slide_height


def _tb(slide, l, t, w, h):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tb.text_frame.word_wrap = True
    return tb.text_frame


def bar(slide, color=NAVY, h=Inches(0.12)):
    s = slide.shapes.add_shape(1, 0, 0, SW, h)
    s.fill.solid(); s.fill.fore_color.rgb = color; s.line.fill.background()


def title_slide(title, subtitle, author):
    s = prs.slides.add_slide(BLANK)
    box = s.shapes.add_shape(1, 0, Inches(2.3), SW, Inches(2.0))
    box.fill.solid(); box.fill.fore_color.rgb = NAVY; box.line.fill.background()
    tf = _tb(s, Inches(0.8), Inches(2.45), Inches(11.7), Inches(1.7))
    p = tf.paragraphs[0]; p.text = title
    p.font.size = Pt(30); p.font.bold = True; p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    tf2 = _tb(s, Inches(0.8), Inches(4.55), Inches(11.7), Inches(0.8))
    p = tf2.paragraphs[0]; p.text = subtitle
    p.font.size = Pt(18); p.font.italic = True; p.font.color.rgb = BLUE
    tf3 = _tb(s, Inches(0.8), Inches(5.5), Inches(11.7), Inches(0.8))
    p = tf3.paragraphs[0]; p.text = author
    p.font.size = Pt(15); p.font.color.rgb = GREY
    return s


def header(slide, title):
    bar(slide)
    tf = _tb(slide, Inches(0.5), Inches(0.22), Inches(12.3), Inches(0.8))
    p = tf.paragraphs[0]; p.text = title
    p.font.size = Pt(26); p.font.bold = True; p.font.color.rgb = NAVY


def bullets_slide(title, bullets, foot=None):
    s = prs.slides.add_slide(BLANK); header(s, title)
    tf = _tb(s, Inches(0.7), Inches(1.25), Inches(12.0), Inches(5.3))
    for i, (txt, lvl, *col) in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("• " if lvl == 0 else "   – ") + txt
        p.font.size = Pt(22 - lvl*3); p.level = lvl
        p.font.color.rgb = col[0] if col else (NAVY if lvl == 0 else GREY)
        p.font.bold = bool(lvl == 0 and col and col[0] == RED)
        p.space_after = Pt(7)
    if foot:
        ftf = _tb(s, Inches(0.7), Inches(6.85), Inches(12.0), Inches(0.5))
        fp = ftf.paragraphs[0]; fp.text = foot
        fp.font.size = Pt(12); fp.font.italic = True; fp.font.color.rgb = GREY
    return s


def two_image_slide(title, img_left, img_right, caption=None):
    s = prs.slides.add_slide(BLANK); header(s, title)
    half = SW / 2
    top = Inches(1.4); maxh = Inches(5.0)
    for k, img in enumerate((img_left, img_right)):
        pic = s.shapes.add_picture(str(img), int(half*k) + Inches(0.3), top,
                                   width=int(half - Inches(0.6)))
        if pic.height > maxh:
            pic.height = maxh; pic.width = int(pic.height * (pic.width/pic.height))
        # center within its half
        pic.left = int(half*k + (half - pic.width)/2)
    if caption:
        tf = _tb(s, Inches(0.7), Inches(6.7), Inches(12.0), Inches(0.6))
        p = tf.paragraphs[0]; p.text = caption
        p.font.size = Pt(15); p.font.color.rgb = GREY; p.alignment = PP_ALIGN.CENTER
    return s


def image_slide(title, img, bullets=None, img_frac=0.62):
    s = prs.slides.add_slide(BLANK); header(s, title)
    iw = SW * img_frac
    pic = s.shapes.add_picture(str(img), Inches(0.35), Inches(1.25), width=int(iw))
    # clamp height
    if pic.height > Inches(5.7):
        pic.height = Inches(5.7); pic.width = int(pic.height * (pic.width/pic.height))
        pic.left = Inches(0.35)
    if bullets:
        tf = _tb(s, iw + Inches(0.6), Inches(1.4), SW - iw - Inches(0.9), Inches(5.4))
        for i, b in enumerate(bullets):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = "• " + b; p.font.size = Pt(16); p.font.color.rgb = GREY
            p.space_after = Pt(8)
    return s


# ---- slides ---------------------------------------------------------------
title_slide(
    "Building & validating a GPU micromagnetic simulator with an AI coding agent",
    "cross-implementation testing exposes a concurrency defect",
    "Chun-Yeol You  ·  DGIST   |   target: npj Computational Materials")

bullets_slide("Two questions", [
    ("AI coding agents now write substantial scientific software", 0),
    ("Q1  Can an agent build a *validated, performant* simulator — not a toy?", 0, BLUE),
    ("Q2  What verification makes the result trustworthy?", 0, BLUE),
    ("Micromagnetics: mature solvers (OOMMF, mumax3, mumax+), anchored to µMAG standard problems", 1),
    ("Prior AI work *optimized* existing codes; building+validating from scratch is the harder question", 1),
])

bullets_slide("Claude-SpinDynamics (Claude-SD)", [
    ("C++20 / CUDA core + Python API, built end-to-end with an AI agent (Claude Code)", 0),
    ("Solves LLG / stochastic-LLG on structured grids", 1),
    ("Distinctive capabilities", 0, BLUE),
    ("float32 AND float64  — unique among GPU micromagnetic codes (mumax family is f32-only)", 1),
    ("two demag FFT backends: cuFFT and VkFFT", 1),
    ("native SOT / STT / Zhang-Li, DMI, RKKY, magnetoelastic; per-cell materials", 1),
    ("RK4 / RK45-DP / Heun + Relax/Minimize; auto-integrator selection", 1),
])

image_slide("Agent-driven development & architecture", FIG / "Fig1_pipeline.png", [
    "Spec (CLAUDE.md): layers, SI conventions, test map",
    "Loop: kernel + CPU reference + Catch2 test",
    "Accept only if all 4 CUDA builds pass",
    "Fail -> diagnose -> fix (e.g. DMI race)",
], img_frac=0.6)

image_slide("Development is quantified", FIG / "Fig6_dev_metrics.png", [
    "155 commits over 31 days",
    "+136k lines of code churn",
    "tests 7 -> 345 (test-driven)",
    "test : source ratio = 0.55",
    "25 effective-field implementations",
], img_frac=0.62)

image_slide("Validation: µMAG standard problems SP#1–5", FIG / "Fig2_umag_validation.png", [
    "Agrees with mumax3 / mumax+ / OOMMF within their spread",
    "SP#4 <mx>(1ns): CS −0.979, OOMMF −0.984 (ref −0.986)",
    "SP#1 L_c = 99.7 nm (CS = mumax+)",
    "SP#2 (new): CS = mumax3 to ≤0.006",
], img_frac=0.6)

two_image_slide("SP#2 remanence — cross-solver agreement",
                FIG / "FigS3_sp2.png", FIG / "FigS3b_sp2_crosssolver.png",
                caption="Remanent ⟨m⟩/M_s and coercivity vs d/ℓ_ex; Claude-SD vs mumax3 (≤0.006).")

image_slide("Performance: a precision/size trade-off", FIG / "Fig5_landscape.png", [
    "Crossover ~0.1–0.5 M cells",
    "Small / 2-D: CS f32 fastest (5.3× vs mumax3) — CUDA-Graph",
    "Large 3-D: mumax3 / MuMax-CO lead (cuFFT-bound)",
    "f32 = 4–6× f64 (Blackwell Tensor-Core)",
    "No single code wins everywhere",
], img_frac=0.6)

two_image_slide("Performance: cross-solver throughput",
                FIG / "Fig3a_throughput.png", FIG / "Fig3b_scenario_bars.png",
                caption="Throughput vs cell count (crossover ~0.1–0.5 M) and per-scenario ms/eval bars.")

bullets_slide("The key result: a hidden defect", [
    ("Relax a seeded skyrmion across the DMI boundary; record topological charge Q", 0),
    ("Same Claude-SD build, repeated identical runs  →  DIFFERENT Q", 0, RED),
    ("run-to-run std of Q up to 0.48  — even in DOUBLE precision", 1, RED),
    ("Q scattered across topological sectors [−1.5, +1.5]", 1),
    ("mumax3 (relax & minimize): deterministic, robust  →  the diagnostic signal", 0, BLUE),
    ("A correct relaxation must not depend on thread scheduling  →  a defect", 1),
], foot="Fig. 4a")

image_slide("Diagnosis → fix → determinism restored", FIG / "Fig4_race_fix.png", [
    "Bisection: only damped-LLG + DMI scattered",
    "Cause: DMI GPU field missing set_stream override",
    "-> ran on its own stream, racing on d_H_out",
    "(FieldSumGPU single-stream mode skips sync)",
    "Fix: add override -> std(Q) = 0, all builds",
    "Invisible to single build & standard problems",
], img_frac=0.6)

bullets_slide("Why it matters", [
    ("The agent did NOT write defect-free code — it wrote a subtle GPU race", 0),
    ("…that passed every unit test and every standard problem", 1),
    ("What caught it: deliberately redundant design", 0, BLUE),
    ("multiple precisions × FFT backends × independent solvers", 1),
    ("Discrepancy (CS scatter vs mumax3 determinism) was the diagnostic", 1),
    ("Takeaway", 0, RED),
    ("for AI-built scientific software, cross-implementation validation must be a", 1),
    ("first-class design goal — the failure mode is plausible code that is subtly wrong", 1),
])

bullets_slide("Summary", [
    ("An AI agent built a µMAG-validated, dual-precision GPU micromagnetic simulator", 0),
    ("Competitive performance; uniquely offers double precision + two FFT backends", 0),
    ("Its multi-build / multi-solver design exposed AND fixed a real GPU concurrency bug", 0, BLUE),
    ("Open & reproducible", 0),
    ("GPLv3 · github.com/mirryou-maker/Claude-SpinDynamics · Zenodo DOI (on release)", 1),
    ("make_report.py regenerates every table & figure from one results file", 1),
])

out = HERE / "Claude-SD_talk.pptx"
prs.save(str(out))
print("wrote", out, "(", len(prs.slides._sldIdLst), "slides )")
