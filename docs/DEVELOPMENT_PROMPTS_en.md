# Claude-SD Development Prompts — English translation

Full English translation of the 281 user prompts in [DEVELOPMENT_PROMPTS.md](DEVELOPMENT_PROMPTS.md).
Numbering and order are preserved. Project: NanoSpinDynamics → Claude-SpinDynamics, 2026-05-25 to 06-22.

---

## 2026-05-25
1. Commit the work so far, and organize the work items into documentation.
2. Start implementing Phase 1b.
3. Start implementing the Phase 1c LLG integrator.
4. Start implementing Phase 1d STT. Implement SOT (spin-orbit torque) at the same time.

## 2026-05-26
5. Save and commit the results so far, and recommend the next step.
6. Proceed in the recommended order.
7. Keep going.
8. Summarize the progress so far and recommend what to do next.
9. First implement the #1-priority Demag Field, continue through priority #2, then review the result.
10. Continue. If possible, proceed without asking me in the middle.

## 2026-05-27
11. Summarize the progress so far and recommend the next task.
12. Tell me how to change CLAUDE_CODE_MAX_OUTPUT_TOKENS.
13. I'll restart after changing the token count. Remember the work so far; when I restart I'll fix the Phase 2 bug — prepare for that.
14. Continue what you were doing.
15. How do I increase CLAUDE_CODE_MAX_OUTPUT_TOKENS?
16. Set it to the maximum 32000 and continue fixing the demag bug.
17. What's the currently configured TOKEN value?
18. Standardize `CLAUDE_CODE_MAX_OUTPUT_TOKENS` to 32000.
19. Fix the demag bug you were working on.
20. The error still occurs even with TOKEN set to 32000 — what's the solution?
21. Fix the demag bug you were working on.
22. Continue from where you stopped.
23. Proceed again, but break the work into smaller chunks.
24. Is the demag bug actually fixed?
25. Proceed again.

## 2026-05-28
26. Report the current problems.
27. Summarize the progress so far.
28. Add the following content to CLAUDE.md.
29. Summarize the progress so far.
30. Fix the Phase 2 demag bug, breaking the work into steps and proceeding step by step.
31. Keep going.
32. Organize the problems.
33. First run the #1 FFTW diagnosis.
34. Fix Bug #2 first.
35. Fix Bug #3 first.

## 2026-05-31
36. The demag bug keeps occurring and isn't getting fixed. Identify the root cause in detail and propose a step-by-step response.
37. Subdivide the bug-verification steps further and verify step by step; report after each step and ask whether to continue.
38. The demag bug still isn't fixed. To solve it, completely remove the existing demag code and rewrite it considering all the errors seen so far. But since writing it all at once keeps hitting the token max, first plan it as multiple steps; design the plan so it never exceeds the max token usage.
39. clear
40. The demag problem keeps failing due to token overflow. First analyze the root cause and propose a strategy, one that minimizes token consumption.
41. Keep going.

## 2026-06-01
42. The demag bug still isn't fixed. Propose a step-by-step strategy to solve it.

## 2026-06-02
43. The demag bug still isn't fixed. Propose a step-by-step strategy — don't fix the bugs yet, just give the detailed strategy.
44. Run only Step 1 and tell me the result.
45. Apply only the step 1-b change (change the term-3 coefficient to 1/3) and report the result.
46. Save the current state and proceed to the next step.
47. Try again.
48. Compare against the newell_g reference implementation — don't write code, just compare and report.
49. What's the strategy to fix the problems found?
50. First run step 1 and check the result.
51. Save the results so far and document the progress in a summary.
52. Break the recommended fix strategy into finer pieces and present it again.
53. Apply from A1 one step at a time; after each step, analyze the result and re-plan the fix strategy, then report.
54. Run the next step A2 test too and analyze the result. Then recommend the step after that.
55. Update diag_step1.cpp, analyze, then recommend the next step again.
56. Do option A, commit, summarize the results so far into a document, then analyze and recommend the next step.
57. Implement the top-recommended RK45 step by step, review it, then recommend what to do next.
58. For finite-temperature calculations you can't use the adaptive-timestep RK45 you just updated — you must use a fixed timestep. Add finite-temperature micromagnetics simulation to the overall plan.
59. (same as 58) ... add it to the plan — for now just add it to the plan; I'll code it later.
60. Run SP#4 Field B (190°) and summarize the result.
61. Save and summarize the results so far into documentation, and recommend the next step.
62. Start implementing the finite-temperature Phase T — but before starting, present a concrete step-by-step plan first.
63. Start from T1.
64. Recommend the next step.
65. Save the results so far to memory and commit.
66. Realize the T2 step.
67. Implement the T2 step.
68. Proceed with T3.
69. Proceed with T4.
70. Proceed with T5, summarize and report the result, then commit.
71. Save and document the results so far.
72. Recommend the next step.
73. Implement Phase 3 (CUDA); break it step by step and present a concrete plan first.
74. Start from Step 1 and report the result.

## 2026-06-03
75. Re-check Step 2 and Step 3; if there's no problem, skip them and proceed to Step 4.
76. Proceed with the Step 5 optimization.
77. First explain the step-by-step strategy for implementing Step 6.
78. Implement from 6a and review the performance improvement.
79. Proceed with 6b.

## 2026-06-04
80. Check and save the progress so far. Then recommend the next task to do.
81. First run the A GPU unit tests.
82. Proceed with the next step, 6c CUDA streams.
83. Run the large-grid benchmark 500×500×10.
84. Recommend the next step.
85. Implement the GPU kernel precompute.
86. Recommend the next step.
87. Complete the Python bindings.
88. Run µMAG SP#1 and save the result.
89. Recommend the next step.
90. Run the SP#1 phase diagram, commit when done, then recommend the next step.
91. Run the thickness dependence.
92. Make a Jupyter notebook.
93. Make a plan to implement the GPU full LLG step by step.
94. I'll remember this plan and develop it sequentially. In 40 min when usage resets, implement just G1 first.
95. Implement G1.
96. Implement G2.
97. Remember the current state and implement G3.
98. Build the G4 GPU LLG torque kernel and the RK4 stage kernel.
99. Implement G6, commit, and save the result.
100. Run the G7 benchmark.
101. Recommend the next task.
102. To proceed in the recommended order, first run SP#4 GPU 1 ns.
103. Proceed to the next step.
104. Implement HeunIntegratorGPU.
105. Save and commit the results so far.
106. Remember the next task and recommend it when I restart.

## 2026-06-06
107. Implement the next step, large-grid full LLG.
108. Add the HeunIntegratorGPU Python binding.
109. Run #3.
110. Check whether retrying this problem on a 2 nm grid is feasible given the current hardware, then give your opinion.
111. Skip this problem and proceed to the next step.
112. Implement the next item, periodic BC for demag.
113. Update CLAUDE.md.
114. What items remain from the original goal?
115. Run the examples at https://mumax.github.io/examples.html, make notebooks for them, and save the results.
116. Make an update plan to add the unimplemented features and tell me the implementation difficulty.
117. Add the above to the plan for future implementation.
118. Which examples still won't be implementable even after Phase C is complete?
119. Add "Rotating Cheese" and "Spinning hard disk" to the roadmap as the lowest priority.
120. Commit up to here and save to memory.
121. Document this part in paper form.
122. Is there a user manual written right now?
123. Save up to here so we can continue later.
124. When I ask for the next task, read next_steps.md and recommend it.
125. Implement Phase A from the to-do list step by step. Review each step's result and report to me.
126. Implement Phase B step by step.
127. I want to develop Phase B step by step — organize the list of items to develop step by step.
128. Remember this list and develop from B1-1.
129. Implement B1-2.
130. Implement B1-3.

## 2026-06-07
131. Implement B1-4.
132. Implement B1-5 too.
133. Implement B1-7.
134. Show the detailed plan for the B2 (MFM Imaging) phase.
135. Implement from B2-1.
136. Compare the current development state against the roadmap.
137. I'm going to rename the project to Claude-SpinDynamics and move the working folder to "d:\Claude-Code-R\Claude-SpinDynamics". Request the needed permissions.
138. Don't delete the old folder when done — keep it; start the work.
139. (Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& d:\Claude-Code-R\Claude-SpinDynamics\.venv\Scripts\Activate.ps1)
140. Figure out the development state of Claude-SpinDynamics being developed in this folder and recommend the next task.
141. Start Phase C1.
142. Implement Phase C1 step by step, one step at a time.
143. Ignoring Phase D, what remains?
144. Update README_mumax_examples.md and proceed with the project rename.
145. Rename the project and update README_mumax_examples.md.
146. Rename the project to Claude-SpinDynamics and update README_mumax_examples.md.

## 2026-06-19
147. Check the state of this folder and recommend follow-up tasks.
148. Defer Phase D1/D2 to the very lowest priority; first organize which mumax3 API functions (https://mumax.github.io/api.html) are implemented and which aren't.
149. Implement 1–5 in the recommended order, then recommend the next task.
150. Work through #1 to #5 sequentially and recommend follow-up tasks.
151. Do priorities 1, 2, 3 sequentially, then recommend follow-up (implementing the unimplemented MUMAX functions).
152. Implement from priority 1 sequentially, then recommend follow-up.
153. Do the next recommended task and, when done, recommend the follow-up.
154. Write priorities 1–5, and prepare to implement the unimplemented mumax API functions.
155. Implement #1 through #5 sequentially, then recommend remaining work.
156. Implement priorities 1–4 sequentially, then recommend follow-up.
157. Do them in priority order, then recommend the next task.
158. Do them in priority order, then recommend the next task.
159. Do them in priority order, then recommend the next task.
160. Do them in priority order, then investigate which mumax API functions are unimplemented, and recommend the next task.
161. Do priority tasks 1–4 sequentially and tell me the next recommended task.
162. Implement recommendations 1, 2, 3, 4 sequentially and report the mumax3 API development state.
163. Implement the low- and medium-difficulty unimplemented features.
164. Present a concrete plan to implement the unimplemented API.
165. Develop phases O and P in the recommended order, then recommend the next task.
166. Develop phases Q and R sequentially, then recommend the next task.
167. Proceed in the recommended order and recommend follow-up.
168. Implement phases W, X, Y, Z sequentially and review the unimplemented API.
169. Implement the unimplemented API one by one.
170. Recommend the next task.
171. Is multi-GPU support feasible to implement?
172. Implement the previously recommended priorities 1–4, and implement multi-GPU via approach A.
173. Review the whole implementation and recommend additional work.
174. Do priorities 1–5, and also implement Phase D sequentially. Then recommend follow-up.
175. Implement priorities 1–5.
176. Run priorities 1, 2, 3, 4, 5 sequentially.
177. Stop for now.
178. Run Priority 1 — API quality improvement, Priority 2 — µMAG validation extension, Priority 3 — GPU integrated-pipeline notebook, then recommend the next task.
179. (IDE file opened…) Run from recommendation #2, and review whether any part of the developed code's speed can be optimized, then report.
180. Fix priorities 1–5 sequentially, then review again for any further improvements. Consider using 3rd-party libraries if needed.
181. Implement the immediately-doable items, then the mid-term items first; give a more detailed explanation of the long-term float32 item. Also plan and recommend the 3rd-party-lib implementation.
182. Recommend the next task.
183. First I want to wrap up the optimization work. Present a strategy for the remaining optimizations and for using (free) 3rd-party libraries.
184. Download the needed packages and implement P9, P12, P11, P10, P13, P14 in that order.

## 2026-06-20
185. Find why it stopped and continue.
186. The work stopped — continue from where it left off.
187. Commit and recommend the next task.
188. Run #1 first.
189. First check the implementation state of the 3rd-party-lib optimizations and present a plan to implement the missing parts.
190. Proceed in the recommended order.
191. Are there any remaining optimization items?
192. Do #1 and #2 and recommend the remaining optimizations.
193. Implement the recommended mumax3 script runner.
194. Proceed with the recommendation.
195. Carry out the remaining mx3 extensions.
196. I want to run a benchmark comparing four simulations — mumax3, OOMMF, our program (float32), and our program (double). Make a plan I can review; I'll run it after checking the plan.
197. Include the mumax3-optimized program at "d:\Claude-Code-R\MuMax-CO" as a benchmark target and re-plan: (1) problem set = full set (c), (2) SP#4 grid = high resolution (250×64×1), (3) performance comparison with fixed step.
198. Proceed with fixed step, and do SP#3/#5 with (b).
199. First investigate the solver discrepancy to find and fix the cause, then prepare the benchmark again. Call our app "Claude-SD".
200. Proceed. And add the adaptive-step RK method to the benchmark if possible.
201. Start with the performance sweep and do the rest sequentially. Check carefully for bugs.
202. Why is Claude-SD slower than mumax, and what's the improvement potential?
203. Implement 1, 2, 3 first.
204. Proceed with #3.
205. Investigate #3 first and present a strategy.
206. Develop from #3, then investigate and report the result.
207. Investigate ways to improve performance, including 3rd-party libraries.
208. Do #1 then review; present a concrete strategy for doing #2.
209. Proceed with #2, run the performance test, then report.
210. Proceed with #1.
211. **Goal:** Provide a thorough comparison of VkFFT and cuFFT performance for non-power-of-two cell sizes, highlighting differences in throughput, efficiency, and limitations. **Context:** Analyze the data passed into cuFFT from mumax and from Claude-SD to see whether the inputs differ; if so, document how mumax preprocesses/transforms data before calling cuFFT (layout, normalization, padding, etc.). **Constraints:** distinguish cases with identical cell sizes; state hardware/software assumptions; keep comparisons on the same scale. **Format:** structured sections — VkFFT-vs-cuFFT results on non-power-of-two sizes, mumax→cuFFT data-flow analysis, and a summary of insights.
212. Implement priorities 1, 2, 3, but make VkFFT usage an option. After implementing, proceed with the next optimization follow-up.
213. Recommend the next task.
214. Implement the #1-priority P4. Then do the #3 VkFFT validation and prepare the next benchmark task (comparison with mumax3 etc.).

## 2026-06-21
215. Make a benchmarking plan.
216. Execute the plan.
217. The current benchmark conclusion still seems to be that Claude-SD performs markedly worse than mumax — how can it be improved?
218. Fix the f32 bug, and implement the VkFFT-32 and VkFFT null-stream fixes. Also implement: (5) the structural gap — (A) Exchange/Anisotropy/Zeeman fused kernel and (B) MAC Y/Z symmetry (Hermitian symmetry).
219. Does the above benchmark hold for the non-2^N case too?
220. In which cases can the current Claude-SD (VkFFT-32, cuFFT-32, etc.) show an advantage over mumax?
221. Using the above results, write new benchmark scenarios — both CS-favorable and mumax-favorable — but evaluate fairly under identical conditions (same 32/64, same step size, adaptive step, RK45, etc.).
222. The most-used solver in practice is the adaptive RK45, I believe — wouldn't it be better to add a comparison for that solver?
223. Run the full benchmark and organize the results in detail.
224. Using the latest CS build, rewrite the examples in the notebooks folder, run all options (32, 64, cuFFT, VkFFT) and mumax, and document the comparison.
225. Explain the STT/SOT caveats in more detail.
226. Since non-zero-temperature simulation is also essential for STT/SOT, plan for it. Also, can't adaptive RK45 be used when using STT/SOT?
227. Include mumax+ in the benchmark done so far and re-plan.
228. First run phases 0, 1, 2. If the results are fine, prepare the next phase 3.
229. Is there any problem including nb1 through nb40 in the benchmark too?
230. Proceed with #1.
231. Recommend the remaining tasks.
232. Do #1, #2, #3.
233. (Your previous response had no visible output — please continue and produce a visible response.)
234. Save the benchmark results so far.
235. Recommend follow-up tasks.
236. Proceed through p1 to p4 sequentially.
237. What's the remaining optimization strategy?
238. Run priorities 1 through 5 sequentially and report the results.
239. What do we lose when swapping RK4IntegratorGPU for HeunIntegratorGPU, in exchange for the speed gain?
240. Make the above analysis available so the user can be recommended an integrator when choosing one in the notebook.
241. Record everything so far and put it in the report.
242. (IDE file opened…) Check whether the current benchmark results are from the best version.
243. Re-run the benchmark with the latest version and update the report.
244. Explain in detail what the "BUILD 2 f32 anomaly reproduction" phenomenon is.
245. Fix the f32 timing bug.
246. In the current latest version, considering all the development so far, organize and report any further bugs to fix or performance to improve.
247. Run #1 through #6 sequentially, and when done, review again for further bugs or performance improvements.
248. I'm about to wrap up Claude-SD development. One last time, check and report any improvements, fixes, or things to supplement.
249. Sequentially fix the 4 Critical items + the Should-fix README/docstring/junk cleanup, and run the 4-build test too.
250. Stop the current background jobs and handle all the remaining recommended items.
251. Plan a benchmark comparing Claude-SD vs mumax, mumax+, MuMax-CO (see d:\Claude-Code-R\MuMax-CO), as follows — and include the tables/figures needed so the results can later be written up as a paper/report, plotting directly in Python if needed: (1) use the NBs under notebooks (modify *.py for benchmarking if needed); (2) use the Standard Problems (modify *.py if needed); (3) build 2D and 3D scenarios for cases where each Claude-SD version (cuFFT-32/64, VkFFT-32/64) is favorable and where mumax/mumax+ is favorable; (4) compare fairly with the same or a similar integrator; (5) focus on T=0 but also compare T>0; (6) include auto-integrator-selection per problem in the NB *.py; (7) after the benchmark, compare the strengths/weaknesses of Claude-SD vs mumax/mumax+ and explain which simulations each is favorable for; (8) since this is for a paper, add the needed references. Review, revise, and improve this strategy.
252. Are we still waiting for the SP#2 validation result?
253. SP#2 is taking really too long. Run the follow-up tasks first; afterward, per your advice, first run SP#2 as a lightweight diagnosis and analyze why it's slow.
254. Show the benchmark summary as a table, and: is the SP#2 problem a Claude-SD problem, or does the same phenomenon appear in mumax/mumax+/MuMax-CO too?
255. Keep going.
256. Add mumax+ to the benchmark targets. And keep going with the remaining work.
257. Check whether additional benchmarking using the notebooks under the notebooks folder is meaningful.
258. The results in the notebooks folder are a mix that aren't all from the latest Claude-SD build. So first re-run all NBs on the latest build, then recommend the scenario for a full re-run and targeted reinforcement.
259. Run the three targeted reinforcements: NB45 hardening/reporting + NB43 reconfiguration + integration.

## 2026-06-22
260. Try again.
261. Retry, and if it fails, repeat retrying every 5 minutes.
262. Is the classifier problem still unresolved?
263. What's the estimated completion time?
264. Do the mx3 modification in advance.
265. What are the remaining recommended tasks?
266. Run 1+2.
267. Register the needed files at https://github.com/mirryou-maker/ so users can download and use it.
268. Write a user manual — include guidance on Claude-SD's advantages, installation, beginner and advanced usage, and how to extend it using Claude Code.
269. I'd like to submit a paper about the Claude-SD development — recommend an appropriate journal.
270. Stop any unnecessary running jobs.
271. I'll write a paper targeting npj Computational Materials following the strategy above. Include the figures and tables needed for it, write the Supplementary content too, and plan any additional simulations/benchmarks needed. First, before writing, present the overall flow, table of contents, and strategy.
272. First run the needed additional work P1, P2, P3, P4.
273. Proceed with option 1 and first verify whether it resolves the issue.
274. Do A and B first, then through P2/P4, then present the refined overall strategy once more.
275. Sequentially execute: F1 schematic, F2 assembly, F5, F6 figure generation, mumax3 SP#4/SP#1 accuracy parsing (complete T3), public repo + Zenodo DOI, large-scaling midpoints, MuMax-CO dynamics timing, and OOMMF table plumbing.
276. Write the paper (abstract, body, references) and the needed Supplementary and cover letter.
277. Remember up to here. I'll finalize after deciding on submission.
278. I'd like to make a PPT from the paper written so far.
279. Insert the figures going into the paper (d:\Claude-Code-R\Claude-SpinDynamics\paper\figures\) into the related pages of the PPT.
280. Add per-figure captions and speaker notes too.
281. Organize the prompts I entered to develop Claude-SD and save them to a *.md file.
