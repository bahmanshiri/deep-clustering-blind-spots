# Neurocomputing submission — formatting changes and open items

## What was changed to match the Guide for Authors

1. **Abstract shortened to 250 words** (was 350). The guide caps abstracts at 250
   words; the two-paragraph abstract was trimmed while preserving every
   numerical result and both theorem references.
2. **Added "CRediT Authorship Contribution Statement"** section (required by
   the guide) — inserted before "Declaration of Competing Interest," as a
   placeholder for you to fill in with real author initials and roles once
   the manuscript is de-anonymized.
3. **Added "Declaration of Generative AI and AI-Assisted Technologies"**
   section (required if any AI tool was used in manuscript preparation) —
   inserted immediately before "Declaration of Competing Interest," as a
   placeholder. Delete it if no disclosable AI tools were used, or fill in
   the tool name(s) and purpose if they were.
4. **Figures extracted as separate submission files.** The guide requires
   each figure as its own file named by a logical convention
   (Figure_1, Figure_2, …). All 27 figures from `figures/` were copied and
   renamed into `paper/figures_for_submission/Figure_1.png` … `Figure_27.png`,
   matching the figure numbers used in-text and in the captions.
5. **Verified already-compliant items** (no change needed):
   - File format is single-column Word (double-column is LaTeX-only per the
     guide) — confirmed via the document's section properties.
   - US Letter page size.
   - Equations are native, editable Word equation objects, not images.
   - Tables are native Word tables, not images.
   - Reference list is in a single consistent author–year style throughout —
     the guide explicitly allows any consistent style at initial submission
     ("Our journal reference style will be applied to your article after
     acceptance, at proof stage"), so no reformatting to numbered [1],[2]…
     citations was needed at this stage.
   - Appendices are already lettered (A, B, C) with separately numbered
     equations/tables, as required.
   - "Declaration of Competing Interest" section is already filled in.
   - Keywords: 6 provided, within the allowed 1–7 range.

## Items that need YOUR input before this is submission-ready

1. **Title page identity — DONE.** Updated to:
   - Bahman Shiri Lonbar¹* — Department of Computer Engineering, East
     Azerbaijan Science and Research Branch, Islamic Azad University,
     Tabriz, Iran. (corresponding author)
   - Zeinab Hassanifar² — Department of Accounting, Faculty of Management,
     University of Tehran, Tehran, Iran.
   - Corresponding e-mail: Lonbarllm@gmail.com
   Please double-check the superscript/affiliation formatting once you open
   the file in Word.
2. **CRediT statement — DONE.** Filled in as:
   "B.S.L. (Bahman Shiri Lonbar): Conceptualization, Methodology, Software,
   Formal analysis, Investigation, Visualization, Supervision, Project
   administration, Writing – original draft. Z.H. (Zeinab Hassanifar): Data
   curation, Writing – review & editing." Adjust if any role doesn't match
   what was actually done.
3. **Declaration of generative AI use** — state which tools (if any) were
   used and for what purpose, or delete the section if not applicable.
4. **Data Availability statement** — replace the bracketed placeholder with
   the real repository URL/DOI (the guide requires Option C: deposit +
   cite/link, or an explanation of why data cannot be shared).
5. **Acknowledgements / funding sources** — replace the bracketed
   placeholder, or delete the section if there is nothing to disclose.
6. **Competing interests declarations tool** — the guide also requires
   completing Elsevier's online "declarations tool" at submission and
   uploading the resulting Word document separately (this is a portal step,
   not something in the manuscript file itself).
7. **Vitae / photos** — only required if this article type calls for author
   biographies; not included here since it depends on the corresponding
   author's identity being finalized.

## Files in this package

- `BlindSpots_Neurocomputing_v24_Submission.docx` — the updated manuscript
  (editable Word source, single-column, US Letter).
- `Highlights.docx` — separate highlights file (3–5 bullets, each ≤85
  characters, per the guide's requirement that highlights be submitted as a
  separate file with "highlights" in the filename).
- `figures_for_submission/Figure_1.png` … `Figure_27.png` — each figure as
  its own file, matching in-text figure numbers, ready for individual
  upload during submission.
- Everything else (`code/`, `data/`, `figures/`, `Dockerfile`,
  `requirements.txt`, `README.md`) is unchanged from your original
  reproducibility package.

## Update — Acknowledgements removed, Data Availability pending, code audit

4. **Acknowledgements — REMOVED.** Since no funding body supported this work,
   the entire Acknowledgements section (heading + placeholder text) was
   deleted from the manuscript, per the guide's own instruction ("...or
   delete this section if none apply").
5. **Data Availability — awaiting your repository link.** You mentioned
   you'd provide it, but it wasn't included in your last message yet.
   Please send the actual GitHub/Zenodo URL (and, ideally, a DOI if you
   archive it on Zenodo) so it can be inserted in place of the current
   placeholder text in the manuscript.

## Code/data audit performed on your request

Every file mentioned anywhere in the manuscript text (not just Table 13)
was checked against every file actually present in `code/` and `data/`.
Findings:

- **Fixed:** Table 13 (Reproducibility Manifest) was missing three files
  that the manuscript body *does* cite by name: `dataset_lib.py` (Section
  3, Proposition 1), `generate_hard_dataset.py` (Section 4), and
  `ICS_Eigengap_HDBSCAN.py` (Section 6.5). All three are now added as new
  rows in Table 13 with the correct section references, so the manifest
  now matches every in-text code citation.
- **For your decision — files present in `code/` but not cited anywhere in
  the manuscript, and not listed in Table 13:**
  - `make_figures.py` and `make_figures_v2.py` — earlier iterations
    superseded by `make_figures_v3.py` (and `make_figures_v4_radial_update.py`
    for the two figures it updates). These look like development history
    rather than the scripts that produced the published figures. Recommend
    removing them from the public repository to avoid a reviewer running
    the wrong version and getting different-looking figures.
  - `generate_paper.js` — this is the build script that assembles the
    manuscript Word document itself (a paper-generation/authoring tool),
    not code that reproduces any reported experimental result. It's
    unusual to include a document-generation script in a reproducibility
    package; recommend removing it unless you specifically want to
    disclose your authoring tooling.
  - `analyze_pipelines.py` and `dec_idec_figure.py` — these ARE legitimate
    (an early naive-vs-informed pipeline comparison, and the script that
    regenerates Figure 13, respectively). They're fine to keep even though
    Table 13 doesn't name them individually; left as-is.
  Nothing was deleted without confirmation — say which of the two flagged
  items (old `make_figures` versions, `generate_paper.js`) you want removed
  from the zip, and it will be rebuilt.
- **Data files:** every `data/*.csv` / `data/*.json` file corresponds to an
  output of one of the scripts above; nothing appeared orphaned.
