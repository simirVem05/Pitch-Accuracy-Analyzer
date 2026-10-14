#!/usr/bin/env bash
# Rehearsal of the migration on a scratch copy of data/main_data, with fault injection.
set -u
cd "$(dirname "$0")/../../.."
R=data/migration_v2/rehearsal
TS=2026-10-14T12:00:00+00:00
rs() { venv/bin/python scripts/resegment_main_dataset.py --main-dir "$R/main_data" "$@"; }
rm -rf "$R"; mkdir -p "$R" && cp -Rp data/main_data "$R/main_data"
rs stage --segmentation v2 --staging "$R/staging" --timestamp "$TS" > "$R/stage.log" 2>&1; echo "stage exit $?"
rs backup --backup-dir "$R/backup" > "$R/backup.log" 2>&1; echo "backup exit $?"; tail -2 "$R/backup.log"

echo "--- fault 1: a song's notes.npz changes after the backup"
cp "$R/main_data/at_last/notes.npz" "$R/at_last_notes.bak"; printf x >> "$R/main_data/at_last/notes.npz"
rs promote --staging "$R/staging" --backup-dir "$R/backup" > "$R/promote1.log" 2>&1; echo "promote exit $? (expect nonzero)"
tail -1 "$R/promote1.log"; echo "songs promoted before stop: $(grep -c 'promoted (' "$R/promote1.log")"
cp "$R/at_last_notes.bak" "$R/main_data/at_last/notes.npz"

echo "--- fault 3: a stale notes_v1.npz sidecar that does not match the backup"
W=work_rihanna_full_song
cp "$R/main_data/adele_hello/notes.npz" "$R/main_data/$W/notes_v1.npz"
rs promote --staging "$R/staging" --backup-dir "$R/backup" > "$R/promote_f3.log" 2>&1; echo "promote exit $? (expect nonzero)"
tail -1 "$R/promote_f3.log"; echo "songs promoted in this run before stop: $(grep -c 'promoted (' "$R/promote_f3.log")"
[ "$(venv/bin/python -c "import json;print(json.load(open('$R/main_data/$W/metadata.json'))['dataset_schema_version'])")" = 1 ] && echo "$W left untouched at schema 1"
rm "$R/main_data/$W/notes_v1.npz"

echo "--- fault 2: interrupted inside an unpromoted song (sidecars + new notes written, metadata still v1)"
S=you_raise_me_up
cp "$R/main_data/$S/notes.npz" "$R/main_data/$S/notes_v1.npz"; cp "$R/main_data/$S/metadata.json" "$R/main_data/$S/metadata_v1.json"
cp "$R/staging/$S/notes.npz" "$R/main_data/$S/notes.npz"
venv/bin/python - "$R/main_data/$S" <<'PY'
import sys; sys.path.insert(0, "scripts")
from pathlib import Path
from main_data_io import validate_song_output, ValidationError
try:
    validate_song_output(Path(sys.argv[1])); print("half-written song passed validation (BAD)")
except ValidationError as e:
    print("half-written song rejected by validation:", e)
PY
echo "--- resume"
rs promote --staging "$R/staging" --backup-dir "$R/backup" > "$R/promote2.log" 2>&1; echo "promote exit $?"
echo "already promoted: $(grep -c 'already promoted' "$R/promote2.log"), promoted now: $(grep -c 'promoted (' "$R/promote2.log")"
echo "--- rerun (idempotency)"
rs promote --staging "$R/staging" --backup-dir "$R/backup" > "$R/promote3.log" 2>&1; echo "promote exit $?"
echo "already promoted: $(grep -c 'already promoted' "$R/promote3.log"), promoted now: $(grep -c 'promoted (' "$R/promote3.log")"
echo "--- validate"
rs validate --backup-dir "$R/backup" > "$R/validate.log" 2>&1; echo "validate exit $?"; grep -E '"passed": false|"all_passed"' "$R/validate.log"
