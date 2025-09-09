Third‑Party Licenses Overview
=============================

This project is a convenience layer (wrappers, glue code and utilities) around a collection of independent third‑party local feature extractors and matchers. Each wrapped baseline keeps its own original license. Your rights and obligations for any given model / baseline are governed solely by that model's upstream license, not by this repository.

You (the user) are fully responsible for:

- Verifying that every baseline you enable is compatible with your intended use (research, internal evaluation, commercial product, etc.).
- Complying with attribution, notice reproduction, redistribution, share‑alike, copyleft, patent, non‑commercial, and any other conditions imposed by the upstream licenses.
- Tracking additional transitive licenses for weights, datasets, and dependencies pulled in by those upstream projects.

Nothing in this repository or its distribution should be construed as granting permission to use any third‑party component beyond what its own license allows. This document is informational only and not legal advice.

If any summary here conflicts with the actual license text, the original license text prevails.

---------------------------------------------------------------------
Quick Summary Table
---------------------------------------------------------------------

| Baseline / Component | License File              | License (short)        | Commercial Use | Notes |
|----------------------|---------------------------|------------------------|----------------|-------|
| ALike                | licenses/LICENSE_ALIKE.txt         | BSD 3‑Clause           | Yes (Permissive) | Keep attribution & disclaimer. |
| ALIKED               | licenses/LICENSE_ALIKED.txt        | BSD 3‑Clause           | Yes (Permissive) | Same as above. |
| DISK                 | licenses/LICENSE_DISK.txt          | GPLv3                  | Yes (Copyleft) | Derivatives & combined distribution must satisfy GPLv3. |
| LightGlue            | licenses/LICENSE_LIGHTGLUE.txt     | Apache 2.0             | Yes (Permissive) | NOTICE retention; patent grant. |
| LoFTR                | licenses/LICENSE_LOFTR.txt         | Apache 2.0             | Yes (Permissive) | Same as above. |
| R2D2                 | licenses/LICENSE_R2D2.txt          | CC BY‑NC‑SA 3.0        | No (Non‑Commercial) | Share‑Alike + Non‑Commercial. |
| RoMa                 | licenses/LICENSE_ROMA.txt          | MIT                    | Yes (Permissive) | Include license. |
| SuperGlue            | licenses/LICENSE_SUPERGLUE.txt     | Non‑Commercial Research | No (Non‑Commercial) | Magic Leap research license; distribution restrictions. |
| SuperPoint (original)| (Upstream Magic Leap)     | Non‑Commercial Research | No (Non‑Commercial) | Check original repo & license. |
| SuperPoint (open impl)| baseline_superpoint_open | Likely Apache/MIT (verify) | Usually Yes | Inspect upstream project actually used. |
| Others (D2Net, DeDoDe, DALF, DEAL, DELF, SOSNet, TFeat, ORB, XFeat, LogPolar) | See feature/ + submodules | Mixed | Varies | Consult each upstream repository/license. |

Legend: High‑level shorthand only. Always read the full license text.

---------------------------------------------------------------------
Included Full Text License Files
---------------------------------------------------------------------

The verbatim license texts for the baselines explicitly vendored here are kept alongside the source:

- licenses/LICENSE_ALIKE.txt
- licenses/LICENSE_ALIKED.txt
- licenses/LICENSE_DISK.txt
- licenses/LICENSE_LIGHTGLUE.txt
- licenses/LICENSE_LOFTR.txt
- licenses/LICENSE_R2D2.txt
- licenses/LICENSE_ROMA.txt
- licenses/LICENSE_SUPERGLUE.txt
- licenses/LICENSE_DINOv3.txt

If you add or update a baseline, also add (or refresh) its license file and extend the summary table. Avoid renaming the original license file if upstream references a specific filename.

---------------------------------------------------------------------
Practical Compliance Tips (Non‑Exhaustive)
---------------------------------------------------------------------

- Maintain an internal manifest mapping baseline name -> version/commit -> license identifier.
- Separate non‑commercial components (e.g., SuperGlue, original SuperPoint, R2D2) from any production build pipeline.
- For GPLv3 components (e.g., DISK) ensure that any distribution of binaries including modified GPL code makes the complete corresponding source (with your modifications) available under GPLv3.
- Preserve LICENSE / NOTICE files for permissive licenses (BSD, MIT, Apache 2.0) and add attribution in documentation or about pages where appropriate.
- For Share‑Alike (CC BY‑NC‑SA) adaptations, release them under the same license if you distribute them (still non‑commercial).
- When uncertainty exists (e.g., planned commercial integration, SaaS, redistribution at scale), consult qualified legal counsel.

---------------------------------------------------------------------
Disclaimer
---------------------------------------------------------------------

This overview is provided in good faith but WITHOUT ANY WARRANTY (see each license). It is not legal advice. You are solely responsible for performing your own independent license review and ensuring full compliance.
