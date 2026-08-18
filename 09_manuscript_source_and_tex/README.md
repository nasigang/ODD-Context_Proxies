# 09. Manuscript Source and LaTeX

Full compile-ready LaTeX sources for the main paper and supplementary material.

## Subdirectories
- `main_paper/`:
  - `main.tex`: Document entrypoint (WACV review format, datasets track).
  - `preamble.tex`: Required packages (tabularx, booktabs, microtype, etc.).
  - `references.bib`: Complete, primary-source verified BibTeX with 16 cited entries.
  - `sec/`: Modular section files (`0_abstract.tex` through `8_conclusion.tex`).
  - `wacv.sty` & `ieeenat_fullname.bst`: Official WACV 2027 style and bibliography templates.
- `supplement/`:
  - `supplement.tex`: Single-column 2-page supplementary material entrypoint.
  - `sec_supp/`: Modular supplement sections (`s1_feature_confirmation.tex` through `s4_provenance.tex`).
