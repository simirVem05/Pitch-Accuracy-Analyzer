# Axis 2 v2 migration report

Migration of `data/main_data/` from v1 note segmentation (schema 1) to the canonical v2 segmentation (schema 2), and
build of `data/ml_dataset/axis2_v2/`. Everything was regenerated from the saved `harmony.npz` vocal tracks. No
source audio was used and none was required: no Demucs, no CREPE, no audio preprocessing.

- **Migration timestamp:** `2026-10-14T18:42:42+00:00`
- **Provenance commit:** `2364668c` on branch `axis2-segmentation-v2`
- **Canonical config:** `note_segmentation_v2.CANONICAL_CONFIG`:
  `mode="gap_v1", hard_gap_s=0.5, separation_cents=100, gap_before="local", gap_context_frames=10,
  target_estimator="core_median"`, with all other fields at their defaults. This is exactly `gapv1local_g0.5_s100`
  from `ml/audits/axis2_segmentation_v2/segmentation_v2_summary.md`.

**Outcome: completed. Every check that was run passed.** Nothing was deleted. The 63 original `notes.npz` and
`metadata.json` files are preserved in place as `notes_v1.npz` and `metadata_v1.json`, and also in a verified
read-only backup. `harmony.npz`, `dataset_selections.json` and `axis2_v1/` are byte-identical to before.

---

## 1. Phase 1: baseline

**Repository.** At the start, `main` had three untracked paths (`backend/note_segmentation_v2.py` and the two audit
directories) and no modified tracked files, so no unrelated changes needed a decision. All work is on the new
branch `axis2-segmentation-v2`; it is not merged and not pushed.

| commit | contents |
|---|---|
| `293b1e89` | segmentation v2 module, tail audit, v2 shadow evaluation (as completed in the previous session) |
| `d5abd1a8` | pre-registered 13-song shadow script, committed **before** its first run |
| `c786d7cf` | 13-song shadow results |
| `80eb5df7` | regeneration tooling, `CANONICAL_CONFIG`, builder refactor, v1 audits repointed to `notes_v1.npz` |
| `2364668c` | `--main-dir` for the ML builder; out-of-repo path handling (the provenance commit) |

**Audit conclusions re-verified, not assumed.** The full 100-config sweep and the 22-config follow-up were re-run
from scratch into `/tmp`. Both reproduce the committed `parameter_comparison.csv`,
`followup/parameter_comparison_gap_reference.csv` and `_verification.json` **byte for byte**.

**Which code produces which notes**

| | code | used by |
|---|---|---|
| v1 | `backend/note_segmentation.py:segment_notes_with_frame_indices` | production backend (`main.py`, `api.py`, `evaluate.py`); `build_main_dataset.py` (audio builder); `notes_v1.npz` |
| v2 | `backend/note_segmentation_v2.py:segment_notes_v2_with_frame_indices` + `CANONICAL_CONFIG` | `scripts/resegment_main_dataset.py`; canonical `notes.npz` (schema 2); `axis2_v2` |

**Downstream consumers found**
- **Of `notes.npz` / `metadata.json`:**
  - the tail audit;
  - stable-center `stable_center_audit.py`, `control_subwindow.py` and `examples_and_plots.py`;
  - `evaluate_segmentation_v2.verify_v1`.

  All five are v1 analyses that cross-check against `axis2_v1`, so they now read `notes_v1.npz` when it exists
  (§5, check 13).
- **Of the Axis 2 ML dataset:** `ml/baselines/*` (`axis2_v1` only; unaffected), plus split membership read by the
  audits (`axis2_v1`).
- **Of `manifest.json`:** `build_main_dataset.py` only.
- **Not consumers:** the production backend and frontend read nothing under `data/`.

## 2. Phase 2: the 13 songs outside the 50-song audit

`ml/audits/axis2_segmentation_v2/shadow_excluded_songs.py` uses the same harness, classifier, metrics and v1 mapping
as the 50-song audit. Its 11 pass criteria were fixed in code and committed (`d5abd1a8`) before the first run, so they
could not be tuned to the result. Each criterion is derived from the 50-song value with a stated tolerance.

| criterion | rule | 13-song result | |
|---|---|---|---|
| C1 | v1 reproduces canonical notes exactly | 13/13 songs, 5,313 notes | PASS |
| C2 | no v2 note bridges a gap > 0.5 s | 0 (v1: 249) | PASS |
| C3 | clear-invalid falls ≥ 75% relative | 12.70% → 0.51% (−96.0%) | PASS |
| C4 | no song gains clear-invalid notes | none | PASS |
| C5 | likely-invalid decreases | 20.63% → 7.51% | PASS |
| C6 | ≥ 90% of v1-valid notes unchanged | 92.70% | PASS |
| C7 | adjacent-pitch merge rate ≤ v1 | 0.53% → 0.22% | PASS |
| C8 | ≤ 2% of vibrato-like notes split | 0 of 42 | PASS |
| C9 | same-pitch tiny notes ≤ 3% of notes | 1.82% | PASS |
| C10 | clear-invalid built only from valid v1 ≤ 0.1% | 0 | PASS |
| C11 | every song yields notes | 13/13 | PASS |

Other measures on the 13 songs:
- IQR > 400 ¢: 3.41% → 1.04%.
- Maximum internal gap: 22.82 s → 0.50 s.
- Notes < 0.10 s: 11.5% → 19.8%.
- Alternating-semitone chains (vibrato over-splitting, inherited from v1): 4.12% → 4.16% of notes.

**Note count rises more here than on the 50 songs: +17.7% vs +14.9%.** The two largest increases,
`billiejean_mj_full_song` (+40%) and `mark_ronson_uptown_funk` (+43%), are the weakest extractions in the set. v1 had
100 and 104 clear-invalid notes on them; v2 leaves 5 and 3. **The 13 songs support extending the configuration to all
63.**

## 3. Phase 3: tooling and its tests

| file | role |
|---|---|
| `scripts/main_data_io.py` | audio-free note-array construction, atomic writers, schema-aware validation, `npz_content_sha256`. Moved **verbatim** out of `build_main_dataset.py`: an AST comparison shows 8 functions byte-identical; `_note_arrays` differs only in a type hint and docstring. |
| `scripts/resegment_main_dataset.py` | `stage` / `compare-v1` / `backup` / `promote` / `validate`, from `harmony.npz` only |
| `scripts/build_axis2_ml_dataset.py` | re-implementation of the uncommitted `axis2_v1` builder |
| `scripts/build_main_dataset.py` | imports the shared helpers; now refuses to write into a store at a different schema, so an audio rebuild cannot silently downgrade the v2 store |

**The recovered `axis2_v1` corruption procedure:**
- `rng = default_rng(sha256(f"20260824:{song}:{idx}")[:8])`;
- `magnitudes = rng.uniform(lows, highs)` (one draw per band);
- `positives = rng.choice(4, 2, replace=False)`.

Tests, all of which were run:

| test | result |
|---|---|
| v1 re-segmented from `harmony.npz` through the new code vs canonical `notes.npz` | **63/63 byte-identical**, 27,554 notes |
| `axis2_v1` rebuilt from v1 notes vs the published dataset | `examples.npz`, `manifest.csv`, `splits.json`, `summary.json` **all byte-identical**; 88,964 shifts re-derived from the seed |
| v2 staging run twice with the same timestamp | byte-identical (`diff -r`) |
| staged v2 vs the audit's per-note output (`v2_notes.csv`, recommended config) | 25,547 / 25,547 notes match (start, end, frames, target, core median); 13-song counts match `per_song.csv` |
| staged `axis2_v2` | 10/10 independent checks: lengths, offsets, 5 variants, clean = 0, 2+/2−, band limits, seed re-derivation, contours = canonical + shift, song-level disjoint splits, manifest rows |
| rehearsal on a scratch copy (`rehearse_migration.sh`) | see below |

**Rehearsal fault injection.** Run on a full scratch copy of `data/main_data`:

| injected fault | expected behaviour | observed |
|---|---|---|
| a song's `notes.npz` modified after the backup | promote stops at that song | stopped at `at_last` after 3 songs |
| a stale `notes_v1.npz` that does not match the backup | promote stops; song untouched | stopped; song left at schema 1 |
| a song half-written (sidecars and new notes, old metadata) | validation rejects it | rejected ("note count differs") |
| resume after the faults | completes exactly the remainder | completed the remaining 3 |
| second promote | no-op | 63 already promoted |
| final validation | all pass | all pass |

## 4. Phase 4: migration

1. **Staged** all 63 songs at `2364668c` into `data/migration_v2/staging_v2_final/` and validated each against its
   canonical `harmony.npz`. Result: 31,803 notes.
2. **Backed up** `data/main_data/` to `data/backups/main_data_v1_20261014/`.
   - 191 files: 63 × (notes, harmony, metadata), plus the manifest and `.DS_Store`.
   - Verified three independent ways: the script re-hashed the copy and re-opened every `.npz` and `.json`;
     `diff -r` found the trees identical; `shasum -a 256 -c` reported 0 failures of 191.
   - Made read-only.
   - Hash records: `data/backups/main_data_v1_20261014.{backup_manifest.json,SHA256SUMS}`, copied here as
     `backup_manifest.json` and `backup_SHA256SUMS`.
3. **Recorded protected hashes** before writing anything: 69 files (63 `harmony.npz`, `dataset_selections.json`, the
   5 `axis2_v1` files).
4. **Promoted** song by song, each step atomic via `os.replace`:
   1. copy `notes.npz` → `notes_v1.npz` and `metadata.json` → `metadata_v1.json`, each verified against the backup
      hash;
   2. replace `notes.npz`, then `metadata.json`;
   3. validate the song;
   4. after all 63 songs, write the manifest.

   All 63 were promoted. The built-in validation passed (`promote_validation.json`).
5. **Built** `data/ml_dataset/axis2_v2/` from the canonical store, and separately from the staging directory with the
   same timestamp. The two builds are byte-identical.

## 5. Phase 5: post-migration validation

| # | check | result |
|---|---|---|
| 1 | all 63 songs accounted for: manifest = song directories = `dataset_selections.json` | PASS (63/63/63) |
| 2 | every file loads | 191 `.npz` with `allow_pickle=False`, 132 `.json`, 0 failures |
| 3 | manifest records, metadata and `notes.npz` counts agree; every manifest path exists | PASS; manifest total 31,803 |
| 4 | every contour frame equals its source frame in `harmony.npz` (time, f0, confidence) | PASS (schema-2 validation, 63/63) |
| 5 | provenance accurate: re-segmenting `harmony.npz` reproduces each stored `notes_npz_content_sha256`; config = canonical; provenance commit exists in git | PASS 63/63 |
| 6 | schema consistent: manifest 2, all `metadata.json` 2, all `metadata_v1.json` 1 | PASS |
| 7 | `notes_v1.npz` and `metadata_v1.json` byte-identical to the backup's originals | PASS 63/63 |
| 8 | v1 still regenerates exactly from `harmony.npz` (now compared against `notes_v1.npz`) | PASS, 63/63 byte-identical |
| 9 | `axis2_v1` unchanged; rebuilding it from the preserved sidecars reproduces it byte for byte | PASS |
| 10 | all 63 `harmony.npz` and `dataset_selections.json` byte-identical | PASS (69/69 protected hashes; `shasum -c` against the backup: 0 failures) |
| 11 | regeneration twice: re-staging from the migrated store with the same timestamp | byte-identical to the promoted staging |
| 12 | v1 → v2 changes match the audit | canonical = staging = audit, note for note; 50 songs 22,241 → 25,547, 13 songs 5,313 → 6,256 |
| 13 | audit tooling still describes v1 after migration | tail audit, stable-center audit, control-subwindow, the 13-song shadow evaluation and the 50-song follow-up all regenerate their committed outputs **byte for byte** |
| 14 | no change to Axis 1 or the production backend | `git diff 18e39c4f HEAD` on `backend/{main,api,harmony,scoring,preprocess,separation,note_segmentation,evaluate}.py` and `frontend/`: empty. The only backend change is the new `note_segmentation_v2.py`. |
| 15 | backup still intact after migration | PASS |
| 16 | file permissions unchanged | PASS (0600 before and after; the atomic writers create 0600) |

## 6. Before / after

**Canonical store, all 63 songs:**

| | v1 (schema 1) | v2 (schema 2) |
|---|---|---|
| notes | 27,554 | 31,803 (+15.4%) |
| clear-invalid | 3,358 (12.19%) | 267 (0.84%) |
| likely-invalid | 5,637 (20.46%) | 2,564 (8.06%) |
| IQR > 400 ¢ | 898 (3.26%) | 385 (1.21%) |
| internal gap > 0.5 s | 1,273 (4.62%) | 0 |
| internal gap > 1.0 s | 479 | 0 |
| maximum internal gap | 22.82 s | 0.50 s |
| notes < 0.10 s | 3,652 (13.25%) | 6,399 (20.12%) |
| notes < 0.06 s | 1 | 672 (2.11%) |

**`axis2_v1` → `axis2_v2`:**

| | `axis2_v1` | `axis2_v2` |
|---|---|---|
| songs | 50 | 50 (same membership, same song-level splits) |
| clean notes | 22,241 | 25,547 |
| examples | 111,205 | 127,735 |
| train / validation / test notes | 15,416 / 3,242 / 3,583 | 17,737 / 3,740 / 4,070 |
| frames | 2,011,895 | 1,998,885 (−13,010 = 5 × 2,602 frames v2 drops in < 3-frame pieces) |
| notes filtered | 0 | 0 |

## 7. Hashes (SHA-256)

| item | hash |
|---|---|
| `data/main_data` + `data/ml_dataset` + `dataset_selections.json`, before (197 files, aggregate) | `ac5a1882e4d0f41be20b587861bded8b3e9f2307dbdd91dd7c15ce0d61a5df19` (also matched just before migration) |
| `data/main_data`, after (aggregate over 317 files: 63 × 5 + manifest + `.DS_Store`) | `0470518cc4bf85612bb4346fe01b3b4a251b04697e41323f3b7679afb7d0c836` |
| `data/main_data/manifest.json` (after) | `4f5fa5da0fb3051bb412bf5c2399cb1d85013b285a8149a188e7381b0576b999` |
| `data/dataset_selections.json` (unchanged) | `93048d6ed30f8cbe1052893c680da7a7bc0a09146df6b3a29f4a627f37beea10` |
| `axis2_v1/examples.npz` (unchanged) | `3957a9c854dd9bd31630f197802ea1da6758cf5da4f0ef34506b366e1a59efb6` |
| `axis2_v2/examples.npz` | `d55b829c1a86211c43fc1e17f46b65d64f7abcb17233304fe3f582850168d702` |
| `axis2_v2/manifest.csv` | `497e3ea46e885d01a8d824a03d6216ec3b38c33e266c198da8f2052e7e1990d8` |
| `axis2_v2/summary.json` | `d331f156d081a5f94a5f5f1fe112ab7b3aa3e8d4d2ed98b2c21b429dc9d30340` |
| `axis2_v2/splits.json` | `681f7232ec82ad6a30418ba07ebf30b5f4fa10e6989651cc2ec6595341fd44b7` |
| backup record `main_data_v1_20261014.backup_manifest.json` | `a8e9b7f68d8ba293c3e255f5308a326f0cbb176df6f3fd8f88be068fef16d404` |
| backup `SHA256SUMS` | `f36dab7f731dfd7a33c630597b2611c75ef0da16089eb32fff33f9cb9c0b6cb8` |

Per-file hashes:
- every original file: `backup_SHA256SUMS`;
- every protected file and every backed-up file: `backup_manifest.json`;
- every staged song: `stage_report.json`.

## 8. Failures encountered during the work

None affected the data.

- **First sweep re-verification crashed.** `multiprocessing` on macOS cannot spawn workers from a stdin script. It
  was re-run from a script file, which passed.
- **First ML-builder validation was very slow.** Indexing `NpzFile` inside a loop decompresses on every access. It
  was fixed by loading the arrays once, before any output was moved into place.
- **First rehearsal attempt didn't run.** zsh does not word-split a variable holding a command, so nothing executed
  (exit 127); only the scratch copy existed. It was re-run as a bash script.
- **First version of the "interrupted song" fault was invalid.** It targeted a song that had already been promoted.
  It was corrected to an unpromoted song and re-run.

## 9. Unresolved issues and follow-ups

1. **No off-machine backup.** `data/` is gitignored, so the backup on this disk and the in-place `*_v1` sidecars are
   the only copies. Copy `data/backups/main_data_v1_20261014/` and its two record files somewhere else.
2. **Production and the dataset now segment differently, by design.** The production backend still uses v1, as
   required. CLAUDE.md's invariant that the dataset "cannot drift from production behavior" no longer holds for
   notes. Swapping production to v2 changes Axis 1 inputs and needs Axis 1 re-validation, so it remains a separate
   decision.
3. **`build_main_dataset.py` (the audio path) still writes v1 / schema 1.** It now refuses to touch the migrated
   store. If audio ever returns, it would need to adopt v2, or a schema-1 rebuild would have to go to a new directory.
4. **The ML baselines were not re-run on `axis2_v2`.** `ml/results/axis2_v1/` still describes `axis2_v1` only.
5. **CLAUDE.md is out of date.** It doesn't cover schema 2, `notes_v1.npz`, `axis2_v2`, the new scripts, the
   committed `docs/RESEARCH.md`, or the missing `docs/NEW_VERSION.md`. It was deliberately not edited in this task.
6. **Inherited segmentation limitations remain, as documented in the audit.**
   - v1's vibrato over-splitting: ~6% of notes sit in alternating-semitone chains, an upper bound.
   - Same-pitch tiny tails after short gaps: ~1.6–1.8% of notes.
7. **Staging and rehearsal directories remain** in `data/migration_v2/` (390 MB). They are safe to delete once you
   are satisfied; this task deliberately deleted nothing.
8. **Additions beyond the brief.** Besides `notes_v1.npz`, each song also keeps `metadata_v1.json`, which makes
   per-song rollback a pair of renames. `axis2_v2/` also contains `_build_report.json` and a generated `README.md`.

**Rollback.** For each song: `notes_v1.npz` → `notes.npz` and `metadata_v1.json` → `metadata.json`; then restore
`manifest.json` from the backup. Alternatively, restore the whole of `data/main_data/` from the read-only backup.
Either way, delete `data/ml_dataset/axis2_v2/`. Nothing else was written.
