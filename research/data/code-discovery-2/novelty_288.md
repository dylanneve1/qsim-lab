# Novelty check: weight-6 [[288,16,16]] (literature as of 2026-10-05)

## Verdict

- **I found no weight-6 [[288,16,16]] code anywhere I looked.** I also found no weight-6 qubit CSS code with n <= 300 and k*d^2/n > 14.11, no weight-6 code with n <= 288, k >= 16 and d >= 16, and no weight-6 [[288,k,d]] with k >= 16 and d >= 14. Among the papers checked, Liang et al.'s [[254,14,16]] (14.11) is still the weight-6 maximum for n <= 300. For comparison, [[288,16,16]] gives 16*256/288 = **14.22**, which is +0.8% over 14.11.
- **The parameters [[288,16,16]] themselves are already published, but at weight 9.** The code is a quantum Tanner code over Q8 with [6,3,3] local codes.
  - Source: Leverrier, Rozendaal & Zemor, arXiv:2512.20532, Table 1(b), row "288 16 16 Q 8 50k".
  - Its distance was estimated with QDistRnd using 50k trials, so it is an upper-bound estimate.
  - Wang, Liu, Li, Kubica & Gu reproduce it in arXiv:2601.15446, Table 1 ("[[288,16,16]] 9 9", where w = 9) and Table 4.
  - **Any claim should therefore say "weight-6", not just "[[288,16,16]]".**
- **A better code exists at weight 7.** Qian & Li, arXiv:2608.08996, list [[288,16,18]] at w = 7 with an exact MILP distance and Q = 18.00. Their w is max(check weight, total X+Z qubit degree).

## Papers checked (one line each; rows are copied from the arXiv HTML text, with only whitespace normalised)

1. **Okada & Kasai, 2607.14091 (pair-partition CPM): no.** Table I has no L=6 (row-weight-6) entries; the smallest L is 8. Rows: "[[232,62,12]] 6 (3,8) 29 0.267", "[[248,66,12]] 6 (3,8) 31 0.266", "[[296,78,12]] 6 (3,8) 37 0.264". The second column is girth, so these codes have weight 8. Weight-10 rows: "[[170,8,20]]† 6 (5,10) 17 0.047" and "[[190,8,18]]† 6 (5,10) 19 0.042".
2. **Okada & Kasai, 2605.23894 (two-branch finite-field bases): no.** All bases have row weight 10 or more. Examples: "(3,10) F11 5 [[110,48,6]]", "(4,10) F11 5 [[110,28,10]]", "(3,16) F17 8 [[272,174,6]]", "(4,16) F17 8 [[272,142,8]]". The headline code is [[10240,4108,18<=d<=32]].
3. **Okada & Kasai, 2606.27130: no.** It has a single (3,18)-regular code, [[34542,23032,18]], at weight 18. There are no codes with n <= 300.
4. **Baldelli et al., 2609.24201 (quasi-dyadic QLDPC): no.** The new codes have weight 8 or more, for example "C_QD3 ⟦256,96,16⟧ 16 ..." and "C_QD4 ⟦256,130,8⟧ 16 ...". The only weight-6 rows are baselines: "C_BB2 ⟦288,12,18⟧ 6 3 ...".
5. **Leverrier, Rozendaal & Zemor, 2512.20532 (small quantum Tanner codes): yes, [[288,16,16]] appears, but at weight 9, not 6.** Table 1(b) heading is "Parameters of quantum Tanner codes for [6,3,3] local codes and generator weight 9". Rows: "288 16 16 Q 8 50k" and "288 8 19 C 8 50k". The weight-6 instances in Table 1(a) all have k = 2, up to "252 2 20".
6. **Wang, Liu, Li, Kubica & Gu, 2601.15446 (check-weight-constrained codes): the [[288,16,16]] code here is weight 9.**
   - Table 1: "Quantum Tanner code ... [[288,16,16]] 9 9".
   - Table 4: "Leverrier et al. [47] ... Q 8 [[288,16,(16,16)]] [6,3,3] [6,3,3] 9 [9,9] [9,9] 1.58".
   - Table 5, weight-6 twisted-torus row: "[[248,10,18]], [[254,14,16]], [[294,10,20]], [[310,10,22]], [[340,16,18]], [[360,8,24]] 6".
   - No weight-6 code above 14.11 at n <= 300.
7. **Mian et al., 2608.12509 (quantum Tanner codes at moderate blocklength): no.** "S3 [[288,8,(≤15,≤15)]] 12 12 0.028 6.25 50K 50M" (check weight 12).
8. **Qian & Li, 2608.08996 (multi-agent discovery): no weight-6 code with n <= 300.**
   - Weight-6 codes: "[[336,12,20]], w = 6" and "[[400,16,≤22]], w = 6".
   - At n = 288: "7 [[288,16,18]] 18.00 Z12 × Z48 normal |K| = 4" (A = y^2+y^7+x, B = y^3+x+x^2+x^5 y^9). Also "[[288,24,18]], w = 8" and "[[288,18,18]], w = 9".
9. **Liang, Liu, Song & Chen, 2503.03827 (twisted tori, weight 6): maximum for n <= 300 is 14.11.**
   - "[[254,14,16]]: x^-1 y^-3, y^-6, (0,127), (1,25), 14.11"
   - "[[288,16,12]]: x^-1 y^3, x^3 y^-1, (0,12), (12,0), 8"
   - "[[288,12,18]]: ... 13.5"
   - "[[294,10,20]]: ... 13.61"
   - Above n = 300: "[[310,10,≤22]] ... 15.61" (the text also writes "[[310,10,22]]: kd^2/n = 15.61"), "[[336,10,≤22]] ... 14.40", "[[340,16,18]] ... 15.25".
10. **Aydin, Tamo & Barg, 2606.17268 (coset codes): no weight-6 code above 14.11.**
    - Weight-6 rows: "[[224,12,16]] 6 1/38 0.54% ... 13.71" and, from Table 6, "[[280,12,16]] C14×D10 C5 [1,4,38] [1,7,52]" (10.97) and "[[168,16,10]] C14×S3 C3 [1,12,48] [1,7,44]" (9.52).
    - Weight-7 rows (3+4 supports) are better: "[[248,10,≤21]] C62×C2 C2×C2 [1,6,51] [1,9,10,113]" (≤17.78) and "[[210,10,≤19]] C105 C5 [1,14,76] [1,41,34,102]" (≤17.19).
11. **Lu, Guo, Liu & Yang, 2608.09115 (divisor-driven cyclic bicycle): no.** All codes have n <= 170 and weight 8 or more, for example "[[66,20,7]]2 ... w=19 ... 14.85".
12. **Lu, Yang & Guo, 2609.06572 (weight-8 BB codes): no.** The codes are weight 8 only, at n = 72 and 144.
13. **C. Liu, 2609.36213 (BB codes over group algebras): no.** Only n = 144: "D3 ... 12 12 (6,6)" and "A4×C6 ... 16 12 (8,8)".
14. **Cruz-Benito et al. (IBM), 2606.02418 (LLM-evolved BB codes): no.**
    - Weight-6 CSS codes at n = 288: "[[288,16,12]] | (12,12) | x³+y+y² | y³+x+x² | 12 | 8.0", "[[288,24,12]]§ ... 12.0" (this is a direct sum of two gross codes), and "[[288,32,6]] ... 4.0".
    - "[[180,6,≤21]] (FOM ≤ 14.7)" is a non-CSS code at weight 8 with only a partial upper bound.
15. **Symons, Rajput & Browne, 2511.13560 (covering-graph BB codes): no.**
    - Weight-6 codes at n = 288: "[[288,20,6]] 12 12 8 x⁹y⁹+x⁶y⁷+y⁸ 1+x⁷y⁶+x²y⁹ 2.5" and "[[288,16,12]] 12 12 8 x³y³+x⁶y⁷+y⁸ x⁶+xy⁹+x²y⁹ 8".
    - Their strong n = 288 codes are all weight 8 (four-term A and B): "[[288,14,≤24]] ... ≤28", "[[288,20,≤22]] ... ≤33.6", "[[288,24,≤14]] ... ≤16.3".
16. **Hirasaki & Lee, 2607.28621 (lifting lifted-product codes): no.** The weight-6 lift of the gross-family base (a = x³+y+y², b = y³+x+x²) is "⟦216,12,14⟧" (10.89). "⟦128,16,12⟧" has weight 8.
17. **Lin, Liu, Lim & Pryadko, 2502.19406 (GB/2BGA codes, single-shot decoding): no.** Table 4 covers weight 6 (w_a = w_b = 3) restricted to d_S = 3. Rows: "288 16 12 3 12 12 1+x+x²y³ 1+y+x³y⁸", "288 20 6 3 12 12 1+x+x⁵y³ 1+y+x³y⁵", and above 300, "450 16 16 3 ...".
18. **Wang & Pryadko, 2606.08771 (BB surface codes with boundaries): no.** Weight-6 planar codes only, for example "12 10 288 1+y+x³y⁻¹ 1+xy+x³y² No 4.167", which is [[288,12,10]].
19. **Postema & Kokkelmans, 2502.17052: no.** "5 27 270 1+z+z² 1+z²+z²⁵ 4 16", which is [[270,4,16]].
20. **Galimova, 2603.17703 (trivariate bicycle): no.** The best weight-6 code is "[[140,6,14]] 2×5×7 6 14 6 8.40".
21. **Liang & Chen, 2510.05211 (self-dual BB): no.** Codes have n <= 200 and weight 8.
22. **Hong, 2607.27644 (ZSZ-LP codes, rate 1/5): no.** Check weight is 9, for example "⟦150,30,10⟧" and "⟦210,42,12⟧".
23. **Davenport, Blue & Chuang, 2606.05044 (GB codes as cyclic submodules): no.** Checked from the abstract only, because the HTML returned 404. The codes are k = 2 MCR codes with stabiliser weight 8 to 16.
24. **Tile codes 2504.09171 and planar open-boundary codes 2504.08887: no.**
    - Tile codes: "3 6 [[288,8,12]] 4" and "4 8 [[288,18,13]] 10.6" (weight 8).
    - Open-boundary codes: [[288,8,12]] and [[268,8,12]].
25. **Other papers with no hit (nothing with n <= 300, weight 6 and k*d^2/n > 14):**
    - Two-block and bicycle constructions: Lin & Pryadko 2306.16400 (their "[[64,18,8]]" is W_a = W_b = 4, i.e. weight 8); 2305.06890; Coprime BB 2408.10001; Multivariate bicycle 2406.19151.
    - Other: Copy-cup 2602.23307; Bayesian-optimisation search 2601.18562; RL weight reduction 2502.14372; certificates 2610.03214; Handbook/EC Zoo 2606.11484 (its only n = 288 code is [[288,12,18]]).
26. **Qudit twisted-torus papers, 2602.20158 and 2602.04443: out of scope.** They contain weight-6 codes over Z_p (p >= 3) with k*d^2/n up to about 20 or more at n <= 300, for example "[[242,10,22]]_3". These are not qubit codes. The q = 2 row in 2602.20158 is still "[[254,14,16]] ... 14.11".
27. **GitHub unitaryfoundation/qldpc-challenge: not fetched (the brief was arXiv only); search snippets only, distances unverified.**
    - No [[288,16,16]] submission found. There is a weight-6 BB "[[288,12,16]]" (PR #2256, 10.67).
    - Weight-6 cyclic GB submissions all have n > 300: [[310,10,22]] (#2051), [[434,16,22]] (#2049, 17.84), [[558,14,28]] (#2045), [[558,10,32]] (#2046), [[630,20,24]] (#2047).

## Ambiguities and caveats

- **Search terms.** I searched exactly "[[288,16,16]]", "288,16,16" and "[[288, 16, 16]]" (with quantum code / qLDPC / bicycle), plus "n=288 k=16 d=16". The only real hit was the weight-9 quantum Tanner code.
- **Coverage.** This was a ~25-minute check, not an exhaustive survey. Unindexed preprints, and parameter sets that appear only in figures or supplementary/ancillary files, would be missed.
- **How I read the papers.** I used WebFetch on arXiv abs and HTML pages. Where WebFetch truncated a long page, I re-read the same arXiv HTML read-only with curl plus text extraction. Two cases went further:
  - The 2601.15446 hit was found that way.
  - The Handbook was read as the arXiv PDF in memory.
  - No files other than this report were written.
- **Weight conventions differ between papers.** Qian & Li's w = max(check weight, qubit X+Z degree), so for BB-type codes w = 6 is ordinary weight 6. Aydin, Tamo & Barg use "weight" to mean maximum stabiliser weight.
