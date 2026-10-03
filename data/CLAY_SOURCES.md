# Clay source audit

The descriptions in `millennium_problems.json` are concise paraphrases of the original mathematical formulations linked by the Clay Mathematics Institute. All six official description links were opened and checked on 2026-09-29. The dated upload-directory components in PDF URLs are hosting paths, not asserted dates of formulation.

Only the mathematical targets are included in the JSON descriptions. Current webpage status labels, announcements, reported breakthroughs, numerical verification updates, recent partial results, prize decisions, and other outcome clues are excluded. Source URLs are provenance metadata, not instructions for models to browse. The forecasting runner should supply the descriptions as fixed text and keep retrieval disabled. This audit file is documentation and should not be included in forecasting prompts.

| JSON ID | Original official description | Location checked | Scope preserved |
| --- | --- | --- | --- |
| `bsd` | Andrew Wiles, [The Birch and Swinnerton-Dyer Conjecture](https://www.claymath.org/wp-content/uploads/2022/05/birchswin.pdf) | PDF pp. 1-2; displayed conjecture on p. 2 | Rank equals order of vanishing for elliptic curves over Q. The refined leading-coefficient formula is identified as a stronger related conjecture. |
| `hodge` | Pierre Deligne, [The Hodge Conjecture](https://www.claymath.org/wp-content/uploads/2022/06/hodge.pdf) | PDF pp. 1-2; displayed conjecture on p. 2 | Smooth projective varieties over C and rational Hodge classes; rational combinations of algebraic cycle classes. |
| `navier_stokes` | Charles L. Fefferman, [Existence and Smoothness of the Navier-Stokes Equation](https://www.claymath.org/wp-content/uploads/2022/06/navierstokes.pdf) | PDF pp. 1-2; conditions (4)-(11) and alternatives (A)-(D) | All four alternatives; zero force in A/B, admissible smooth force in C/D; Euclidean and periodic domains. |
| `p_vs_np` | Stephen Cook, [The P Versus NP Problem](https://www.claymath.org/wp-content/uploads/2022/06/pvsnp.pdf) | PDF pp. 1-2, section 1 | Standard Turing-machine definitions of P and NP and the equality question. |
| `riemann` | Enrico Bombieri, [Problems of the Millennium: The Riemann Hypothesis](https://www.claymath.org/wp-content/uploads/2022/05/riemann.pdf) | PDF p. 1, section I | All nontrivial zeros of the Riemann zeta function; no generalization to other L-functions is required. |
| `yang_mills` | Arthur Jaffe and Edward Witten, [Quantum Yang-Mills Theory](https://www.claymath.org/wp-content/uploads/2022/06/yangmills.pdf) | PDF pp. 5-6, especially section 4 | Every compact simple gauge group, nontrivial theory on R^4, axiomatic existence, and positive finite mass gap. |

The JSON is a six-entry array with fields `id`, `title`, `description`, `source_url`, and `original_statement_url`. Descriptions define mathematical scope only; the forecasting protocol separately defines AI contribution, deadlines, and evidentiary thresholds. These paraphrases aid consistent elicitation and do not replace the full mathematical statements when adjudicating a claimed solution.
