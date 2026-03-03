Notes — Docs & GUI Sync (2026-03-03)
=====================================

Documentation
-------------

- **API Reference**: Added autodoc pages for 9 previously undocumented modules:
  ``cli``, ``pgs_features``, ``mature_ranker``, ``mature_model``,
  ``geom_hairpin_finder``, ``geom_stem_features``, ``geom_bulges``,
  ``geom_energy``, ``parallel``.

- **Pipeline Reference**: Rewrote ``pipeline.rst`` to cover all 13 CLI
  subcommands with concrete ``mirpv-ng`` examples. Added missing stages:
  ``scored-to-peaks`` (9.5), ``peaks-to-known-early`` (10e),
  ``candidates-filter``, and ``validate-seqonly``.

- **Quick Start**: Added ``validate-seqonly`` usage example.

- **Build Config**: Added ``myst_parser`` extension for ``.md`` files,
  ``autodoc_mock_imports`` for PySide6/pysam/xgboost to prevent import errors.

GUI
---

- **Sequence-only advanced parameters**: Added collapsible section with
  ``--max-hairpin-len``, ``--max-seq-only-len``, ``--window-len``, ``--step``,
  ``--tier1-min-pairs``, ``--tier1-min-mfe`` (via ``STAGE_PARAMS["score-fasta"]``).

- **Parallelism controls** for sequence-only mode: ``--threads``, ``--backend``.

- **Command Preview**: New "Preview" button shows the exact CLI command
  that will be run, in a read-only text area.

- **Config Export**: New "Export Config" button saves all current GUI settings
  (inputs, parameters, stage selections) to a JSON file for reproducible runs.

- Wired advanced ``score-fasta`` and ``predict-mature`` stage parameters into
  ``_build_commands()`` for sequence-only mode.
