# SOT-driven skyrmion dynamics — Claude-SD vs mumax3

Same parameters, same-condition snapshots. Reproduces `examples/skyrmion_dynamics.py`
in mumax3 and compares. **Both codes use the same skyrmion convention** (see note).

## Parameters (identical in both codes)

| | value |
|---|---|
| grid / cell | 120 × 84 × 1, 3.5 nm |
| Ms / Aex / Ku1 | 6.0×10⁵ A/m / 1.5×10⁻¹¹ J/m / 6.0×10⁵ J/m³ (z easy axis) |
| α | 0.20 |
| interfacial DMI | D = 3.0×10⁻³ J/m² |
| initial state | Néel skyrmion (charge = 1, pol = −1) → **core down**, relaxed |
| drive | SOT, σ̂ = ŷ, J = 3.5×10¹¹ A/m² (magnitude), θ_SH / Pol = 0.25, damping-like |
| capture | every 0.1 ns for 0.9 ns |

## Run

```powershell
D:/Mumax3/mumax3.exe -f sk_dyn.mx3
D:/Mumax3/mumax3-convert.exe -numpy sk_dyn.out/*.ovf
python compare.py        # -> compare_montage.png, compare_curves.png
```

## Result — the two codes agree

![montage](compare_montage.png)
![curves](compare_curves.png)

With identical parameters, Claude-SD (top) and mumax3 (bottom) produce the **same
skyrmion**: same core polarity (core-down, Q ≈ −0.9 in both), the same up-right
trajectory (skyrmion-Hall deflection), and the **same deformation** — round →
teardrop → crescent — compressing against the boundary and annihilating on the
same ≈0.8–0.9 ns timescale. The core-area (deformation) curves overlap closely;
Claude-SD's computed |Q| decays marginally earlier as the skyrmion elongates.

## Convention notes

- **Skyrmion polarity (fixed in this repo).** Claude-SD's `neel_skyrmion` /
  `bloch_skyrmion` now follow the **mumax3 convention**: `pol = +1 → core up`,
  `pol = −1 → core down` (`m_z = −pol·cos θ`, since θ = π at the core). Previously
  `pol` was inverted, which flipped the topological charge sign and mirrored the
  skyrmion-Hall direction. The DMI sign convention follows Rohart–Thiaville /
  Sampaio, as in mumax3.
- **SOT current sign** is a drive-direction choice, not a skyrmion property: mumax3's
  `J = (0,0,−3.5×10¹¹)` here reproduces the same longitudinal push as Claude-SD's
  `J_c = +3.5×10¹¹` with `σ̂ = ŷ`. (Flipping the sign flips the direction of motion in
  either code.)
