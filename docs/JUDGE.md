Graded mesh verdicts (−5..+5) with good/bad tags, and grade-aware Review
  
 Context                                                                                                       

 The mesh-verdict corpus is binary accept/reject and it has proven unusable for its own purpose: the 2026-08-07 review produced 3 accepts
 against 81 rejects, which LEFTOVERS §8 itself calls "not a thin corpus, an unusable one" — a probe fitted to it learns "reject" and scores
 96% doing so, and a binary row can say a mesh failed but never how close it came. The user wants richer quality data: an integer grade
 −5..+5 (+5 = game-useable as-is, −5 = completely unusable), an optional tag vocabulary (good and bad) attached to any grade, and Review
 surfaces that actually use the accumulated data (grade averages, tag tallies, grade deltas in matched pairs). Image-stage labels
 (reference/blank) stay binary — they feed binary logistic probes and the fast two-key labelling loop is what makes a 100-image pass
 viable.

 Decisions made by the user:
 - Integer grades −5..+5, model (mesh) stage only.
 - Backfill existing corpus: accept → +3, reject → −3.
 - Binary "usable" cut wherever one is needed: grade ≥ +3.
 - Tags optional at every grade; vocabularies proposed by me (bad set keeps the five stored spellings).
 - Keyboard: sign-then-digit — 1–5 = +1..+5, R arms negative then 1–5 = −1..−5, 0 = 0, Shift+1-5 toggles bad tags, Ctrl+1-5 good tags, S
 skip, Esc cancels/clears, A removed from the mesh loop (label pass keeps its A/R).

 Verified structural facts (confirmed by direct reads):
 - The next migration is 10 — migration 9 is the trash (jobs.deleted_at, db.py:276-298).
 - service/verdicts.py imports service/findings.py at module top (verdicts.py:47), so grade/tag constants cannot live in verdicts.py if
 findings.py needs them — they go in src/warlock/vectors.py, the exact precedent CLAUDE.md states for VECTOR_PARAMS, re-exported by
 service/verdicts.py.
 - service/judge.py's trainable stages are IMAGE_STAGES only (reference/blank), which stay binary — the judge service and judge.py need
 zero behavioural change. The mesh probe is unbuilt (LEFTOVERS §8); a graded corpus is its future regression target, noted only.

 Design

 1. Vocabulary constants — src/warlock/vectors.py

 GRADE_MIN, GRADE_MAX = -5, 5
 USABLE_GRADE = 3                              # grade >= USABLE_GRADE is "usable"
 BINARY_GRADES = {"accept": 3, "reject": -3}   # migration 10's backfill + missing-grade fallback

 BAD_TAGS = ("holes", "bad-shape", "bad-texture", "wrong-style", "broken")   # spellings frozen: stored corpus
 GOOD_TAGS = ("clean-shape", "good-texture", "on-style", "sharp-detail", "good-topology")
 TAGS = GOOD_TAGS + BAD_TAGS

 Good tags mirror bad where a mirror exists (clean-shape↔bad-shape, good-texture↔bad-texture, on-style↔wrong-style); sharp-detail = detail
 survived reconstruction (what trellis most often loses), good-topology = imports/derives cleanly (constructive counterpart of broken).
 Deliberately no game-ready tag — that is what grade +5 already asserts. Five per polarity so Ctrl/Shift+1-5 map positionally. Tags stored
 in the existing reasons JSON column, one namespace: the vocabularies are disjoint strings, so polarity is recoverable from the tag itself;
 the findings writer splits by membership so bench/findings.py stays pure-stdlib with no warlock imports.

 service/verdicts.py: re-export all of the above; delete REASONS (only src refs are verdicts.py, review_mode.REASON_KEYS, inspector.py:1018
 — all rewritten here).

 2. Schema — migration 10 (src/warlock/db.py)

 - _SCHEMA verdicts table (lines 50-64) gains grade INTEGER (nullable; comment: -5..+5, model stage only; NULL on image labels).
 - Append MIGRATIONS entry 10 (multi-statement, precedent: migration 8):

 [
     "ALTER TABLE verdicts ADD COLUMN grade INTEGER",
     "UPDATE verdicts SET grade = 3 WHERE grade IS NULL AND stage = 'model' AND verdict = 'accept'",
     "UPDATE verdicts SET grade = -3 WHERE grade IS NULL AND stage = 'model' AND verdict = 'reject'",
 ]

 The grade IS NULL guards make a partial replay idempotent; on a fresh DB the ADD COLUMN guard (_ADD_COLUMN_RE, ~db.py:314) skips the ALTER
 and the UPDATEs touch zero rows. Image-stage rows keep grade NULL permanently. Backfill at ±3 (not ±5): accept asserted "usable" and
 nothing stronger — the mildest grade satisfying the cut, leaving ±4/±5 for judgements a binary reviewer never made.

 - verdict TEXT survives as a derived field written at record time (grade >= 3 → "accept"). This keeps four readers unchanged: prune
 retention (jobs.py:680), service/judge.py label reads, latest_verdicts/unverdicted_models SQL, and every findings-v3 reader. One writer
 (record_verdict) owns the derivation, so it is not the two-spellings hazard.
 - add_verdict (885-931): keyword grade: int | None = None, one more INSERT column. _verdict_to_dict needs no change (dict(row)).

 3. Service API — src/warlock/service/verdicts.py

 def verdict_for_grade(grade: int) -> str:   # "accept" if grade >= USABLE_GRADE else "reject"

 def record_verdict(svc, job_id, *, verdict=None, grade=None, reasons=(), source=SOURCE_HUMAN, stage="model")

 Validation: stage checked first. Model stage: grade required, non-bool int in −5..+5 (Invalid(field="grade")); a passed verdict refused
 (Invalid(field="verdict")) — one door in, the derived column can never disagree. Image stages: verdict in VERDICTS as today; a passed
 grade refused. Tags validated against TAGS, legal at any grade. Existing gates (image file exists / mesh status == "done") unchanged.
 Return dict gains "grade".

 4. Review mode — src/warlock/studio/review_mode.py

 - ReviewState: pending_reject → pending_negative: bool; new pending_tags: list[str]. Both cleared by recording, Esc, step, advance (armed
 state belongs to the unit on screen).
 - New maps/helpers: GRADE_KEYS = {str(i): i for i in range(1, 6)}, BAD_TAG_KEYS/GOOD_TAG_KEYS (digit → tag, positional), toggle_tag(state,
 tag), grade_text(grade) -> str ("+4"/"-2"/"0"/""). Delete REASON_KEYS.
 - record(ctx, grade: int, tags=()): calls record_verdict(..., grade=grade, reasons=tags, source=SOURCE); sets unit["grade"],
 unit["verdict"] = verdict_for_grade(grade) (advance/_recount/open_sweep keep keying on verdict is None), unit["tags"].
 - _unit (213-224): add "grade": seen.get("grade"); "reasons" becomes "tags".
 - handle_key (1076-1122) order: labels pass → Esc (disarm + clear tags) → Left/Right → current-None guard → Ctrl+digit (good tag) →
 Shift+digit (bad tag) → "0" → "1"-"5" (sign from pending_negative) → "r" arms → "s". Modifiers read from event.mod (pygame.key.name
 returns "1" regardless of mods; no collision with global Alt+digit, answered above the modes). "a" falls through unconsumed. _label_key
 untouched.
 - Blind: no new logic — recorded grades are the user's own data and keep showing (existing pinned rule); score_line suppression and
 blind_order unchanged.
 - Module docstring: replace "Accept has no such second step, because there is only one way for a mesh to be right" with the graded
 rationale (eleven ways to be almost right; the grade itself now carries what a bare reject lacked; tags optional at every grade).

 5. UI — widgets, main.py, inspector

 - src/warlock/studio/widgets.py: grade_buttons(id_prefix, enabled) -> int | None (row of 11 buttons -5..+5, wrapping via width check) and
 tag_toggles(id_prefix, pending, enabled) ("Good:" row + "Bad:" row of selectable toggles).
 - main.py _review_verdict (2564-2617): grade buttons + tag toggles + muted caption "+5 ships as-is, +3 usable, -5 unusable. Tags are
 optional."; Skip (S) stays; delete the armed "Why?" block; Recorded line = grade_text + tags.
 - main.py unit-list marks (2479, and label grid 2258): grade_text(unit.get("grade")), falling back to CHECK/X icons when verdict present
 but grade absent.
 - main.py _review_findings (2619-2684): render tag_line (see §6) muted under each preset's metrics_line.
 - Shortcut help table (~2940-2949): mirror the new key map.
 - panes/inspector.py _verdict (1002-1022): same two widgets; state.py:741 inspector_reject_armed replaced by inspector_tags_job: str |
 None + inspector_tags: list[str] (different job id clears both; recording clears both); verdict_armed/arm_verdict deleted; toast
 f"Recorded: {grade_text(grade)}.".

 6. Findings v4 — src/warlock/service/findings.py + src/warlock/bench/findings.py

 FINDINGS_VERSION = 4, strictly additive (every v3 key keeps its value and meaning — "accept" is now the derived usable cut, computed
 identically).

 - _summarise (265-290) additions: graded_n, mean_grade (round 2, None if no grades), grades (sparse histogram, string keys), tags:
 {"good": [[tag, n], ...], "bad": [[tag, n], ...]} split by membership in GOOD_TAGS/BAD_TAGS imported from ..vectors (no cycle).
 accepts/accept_rate/wilson_low/sources/top_reasons computed exactly as today.
 - _marginals trim tuple gains graded_n, mean_grade (hint needs the mean per scoped bucket).
 - Vector ranking: primary sort stays -wilson_low of the usable rate; mean_grade becomes the first tie-breaker (then -n, key). A mean−SE
 bound degenerates at n=1 (sd=0 → bound=mean, re-creating the lucky-5/5 pathology Wilson was adopted to kill), and over the all-±3
 backfilled corpus the mean is an affine function of the usable rate anyway. Displayed always; revisit with a measurement doc once real
 spread exists.
 - _comparisons (433-440): win = higher grade. _grade_of(row) reads grade falling back to BINARY_GRADES[verdict] (0 if neither); delta = a
 − b; tie is delta == 0 (reproduces today's semantics exactly over the backfill). Entry gains "grade_delta": {"mean", "pairs"}.
 - bench/findings.py renderings (ASCII/Latin-1 only; · U+00B7 safe):
   - hint: with mean_grade → "usable 6/8 (41%+) · avg +2.6"; entry without it (v3 file) → today's exact "accept 6/8 (41%+)". Subject
 suffixes unchanged.
   - vector_line: "usable 80% of 20 (61%+) · avg +2.6"; legacy → today's string.
   - comparison_lines: winner header gains ", avg +1.4 grade" (mean re-oriented winner-minus-loser via existing flip machinery); legacy
 docs → today's strings.
   - New tag_line(entry) -> str | None: "good: good-texture x4, on-style x2 · bad: holes x3" (top 3 per polarity, None when empty).

 7. Untouched but verified/annotated

 - service/judge.py (120, 226), judge.py, record_label, labelling pass, by_score, blind_order, pump_*: unchanged; judge tests must pass
 unmodified.
 - service/jobs.py prune retention (~680): code unchanged; add one docstring sentence ("accept" is the derived usable cut now).
 - judge.py docstring + LEFTOVERS §8: note the future mesh probe's target is grade regression.

 8. Documents

 - docs/measurements/2026-08-09-grade-scale.md — written FIRST (repo rule: a constant the corpus is keyed on gets its document before it
 changes). Pre-registration style of 2026-08-09-judge-threshold.md: the scale semantics, the ±3 backfill argument, the ≥+3 cut with the
 round-trip consistency proof (backfilled accepts must remain usable; +3 ≥ +3 makes the pair one decision), frozen BAD_TAGS spellings,
 revisit conditions.
 - docs/manual/13-review.md: Judging section rewritten to the new key map; "What works" quoted strings updated. docs/manual/14-shortcuts.md
 (58-79): Review table replaced; labelling table untouched. tests/manual/ must stay green.
 - CLAUDE.md: rewrite the verdict paragraph (grades, derived verdict with one writer, ±3 backfill per measurement doc, tags one namespace
 split by the writer, vocabulary in warlock/vectors.py, Wilson-on-usable + mean tie-break); stage paragraph gains "image labels stay
 binary, grade stays NULL"; "reasons are a mesh-stage concept" → "grades and tags are mesh-stage concepts". Fix stale
 verdicts.append_verdict mention if touched.
 - LEFTOVERS.md §10: answer the "binary first or five-class?" open question in place — never renumber §s. Update §7/§8 phrasing describing
 the mesh corpus as accept/reject.

 Implementation order (TDD; uv run pytest green at each step; never edit src/ while the suite runs)

 1. Measurement doc docs/measurements/2026-08-09-grade-scale.md (no code).
 2. Vocabulary in src/warlock/vectors.py + re-exports. Tests: vocabularies disjoint, BAD_TAGS spellings frozen, BINARY_GRADES consistent
 with USABLE_GRADE.
 3. Schema in src/warlock/db.py (_SCHEMA, migration 10, add_verdict(grade=)). Tests in tests/test_verdicts_db.py: fresh-vs-migrated DB
 converge, backfill values, image rows stay NULL, idempotent replay, migration literals equal vectors.BINARY_GRADES.
 4. Service service/verdicts.py (verdict_for_grade, new record_verdict). Rewrite validation pins (tests/test_verdicts_db.py:130 area):
 grade required/range/bool-refused on model, verdict refused on model, grade refused on image stages, unknown tag refused, good tag at
 negative grade OK, +3 boundary, return carries grade.
 5. Findings v4 service/findings.py. Rewrite tests/test_findings_service.py _summarise exact-dict pin (:71) and add mean/histogram/tags
 cases; tests/test_findings_comparisons.py grade-delta logic + BINARY_GRADES fallback.
 6. Bench reader bench/findings.py (hint/vector_line/comparison_lines/tag_line). Tests: new strings, explicit v3-legacy string pins,
 ASCII-only assertion.
 7. Review mode review_mode.py. Rewrite tests/test_review_mode.py binary pins (:273-:412): digit grades, r+digit negatives, 0, tag toggles,
 Esc/step/advance clear pending, tags ride the recorded row, re-review supersedes, blind shows own grades, a records nothing.
 8. UI: widgets.py (grade_buttons, tag_toggles), main.py (_review_verdict, marks 2479/2258, _review_findings tag_line, help table),
 inspector.py + state.py. Rewrite tests/test_inspector_verdict.py.
 9. Verify untouched consumers: tests/test_judge*.py pass unmodified; prune docstring sentence in service/jobs.py.
 10. Docs: manual 13/14 + main.py help table together; uv run pytest tests/manual; CLAUDE.md + LEFTOVERS §10.
 11. Full uv run pytest (repo norm: also passes with WARLOCK_NATIVE=0; native paths untouched here).

 Verification

 - Full suite green (uv run pytest), including tests/manual/.
 - Manual DB check: open the live DB copy after migration 10 — model rows carry ±3, image rows NULL, verdict strings unchanged.
 - Launch the app (/run or uv run warlock studio), enter Review: grade a unit with 4, r+2, 0, toggle tags with Ctrl/Shift+digit, confirm
 the unit list shows +4/-2/0 marks, "What works" shows usable … · avg … lines and tag tallies, blind mode still hides judge scores while
 showing own grades, and the label pass (images) still answers to A/R only.
 - findings.json regenerates as version 4 and the generate panes' hints render the new strings.

 Out of scope (deliberate)

 - Mesh-probe grade regression (LEFTOVERS §8 — blocked on a corpus with spread; this change creates that corpus).
 - Grade histogram UI, AI-judge grading, per-grade retention tiers beyond the ≥+3 cut.