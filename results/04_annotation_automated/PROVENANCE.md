# Provenance of the returned annotation workbooks

Checked on 15 October 2026 by inspecting the files in `annotation/returned/`.

| Check | Annotator A | Annotator B |
|---|---|---|
| Rows coded (study / practice) | 285 / 10 | 285 / 10 |
| Review IDs and texts unchanged | yes | yes |
| Application that wrote the file | openpyxl 3.1.5 (Python), not Excel | openpyxl 3.1.5 (Python), not Excel |
| File written (UTC) | 2026-10-15 05:06:09.04 | 2026-10-15 05:06:09.19 |
| Rows with a free-text note | 285 of 285 (notes were optional) | 285 of 285 |
| Notes starting "The review…" / "The reviewer…" | 217 | 257 |
| Mean responsibility rating (1 = external, 7 = reviewer) on rows the annotator labelled DISAPPOINTMENT | 6.94 | 2.42 |

The two files were written by a script within the same second, every row carries a
model-style rationale, and annotator A's responsibility scale is reversed relative to A's
own labels on 225 of 228 rows. The annotators could not be contacted to confirm how the
labels were produced.

**Decision.** The labels are treated as coming from two unverified, probably LLM-based,
annotators. They are reported as a secondary check on the GPT-4o systems, never as human
ground truth. The four 1–7 dimension ratings are not used, because annotator A's
responsibility scale is reversed and the two annotators' controllability ratings do not
correlate (Spearman −0.04).

Analysis: `code/automated_annotation.py` → this folder.
