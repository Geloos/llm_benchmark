#!/usr/bin/env python3
"""
injection_set.py

What it does:
  Loads jailbreaks/injections.jsonl and normalises it into the one shape the rest of the
  pipeline uses, folding the two-stage rows (a primer + its activation) into a single
  injection. Imported by both inject_prompts.py and summarize_results.py so the two agree
  on what an injection is.

  A row either carries one `payload` + `tag`, or -- the VG_ vocabulary-guessing rows -- a
  `variants` list of {guess, payload, tag} plus `"rotate": "clockwise"`: the attacker
  does not know the label word, so the guesses are sprayed across the log, marker k
  getting variant k mod len(variants). An optional `template` holding `{guess}` is checked
  against every variant's payload, so a hand-edited variant cannot drift from it.

How to run it:
  Not a CLI. Import it:  import injection_set;  injections = injection_set.load(path)

What it outputs:
  A list of dicts:
      {"id": "SPLIT_01_tagrule", "category": "payload_split",
       "source_ids": ["SPLIT_01a_primer_tagrule", "SPLIT_01b_activation_tagrule"],
       "stages": [{"payload": ..., "tag": ...}, {"payload": ..., "tag": ...}]}
  Single-stage entries get a one-element `stages` list and source_ids == [id].
  A rotating row's stage is {"rotate": "clockwise", "variants": [{"guess", "payload",
  "tag"}, ...]} instead; inject_prompts.pick_stage() resolves it per marker.
"""

import json
import re
import sys

STAGE_RE = re.compile(r'^(?P<base>.+_\d+)(?P<stage>[ab])(?P<rest>_.*)?$')

STAGE_WORD_RE = re.compile(r'_(?:primer|activation)(?=_|$)')

REQUIRED_KEYS = ('id', 'payload', 'tag')
VARIANT_KEYS = ('guess', 'payload', 'tag')
ROTATIONS = ('clockwise',)


def read_rows(path):
    rows = []
    with open(path, encoding='utf-8-sig') as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as e:
                sys.exit('ERROR: %s line %d is not valid JSON: %s' % (path, lineno, e))
            if 'variants' in rec:
                check_variants(path, lineno, rec)
            else:
                missing = [k for k in REQUIRED_KEYS if not rec.get(k)]
                if missing:
                    sys.exit('ERROR: %s line %d (%s) is missing: %s'
                             % (path, lineno, rec.get('id', '?'), ', '.join(missing)))
            rows.append((lineno, rec))
    return rows


def check_variants(path, lineno, rec):
    where = '%s line %d (%s)' % (path, lineno, rec.get('id', '?'))
    if not rec.get('id'):
        sys.exit('ERROR: %s is missing: id' % where)
    variants = rec['variants']
    if not isinstance(variants, list) or not variants:
        sys.exit('ERROR: %s: variants must be a non-empty list' % where)
    rotate = rec.get('rotate', 'clockwise')
    if rotate not in ROTATIONS:
        sys.exit('ERROR: %s: unknown rotate %r (expected one of: %s)'
                 % (where, rotate, ', '.join(ROTATIONS)))
    template = rec.get('template')
    if template is not None and '{guess}' not in template:
        sys.exit('ERROR: %s: template has no {guess} slot' % where)
    for i, v in enumerate(variants):
        missing = [k for k in VARIANT_KEYS if not v.get(k)]
        if missing:
            sys.exit('ERROR: %s variant %d is missing: %s' % (where, i, ', '.join(missing)))
        if template is not None and template.replace('{guess}', v['guess']) != v['payload']:
            sys.exit('ERROR: %s variant %d (%r): payload does not match the template'
                     % (where, i, v['guess']))


def to_stage(rec):
    if 'variants' in rec:
        return {'rotate': rec.get('rotate', 'clockwise'),
                'variants': [{k: v[k] for k in VARIANT_KEYS} for v in rec['variants']]}
    return {'payload': rec['payload'], 'tag': rec['tag']}


def stage_of(injection_id):
    m = STAGE_RE.match(injection_id)
    if not m:
        return None
    return m.group('base'), m.group('stage'), m.group('rest') or ''


def merged_id(base, rest):
    return base + STAGE_WORD_RE.sub('', rest)


def single(rec):
    return {
        'id': rec['id'],
        'category': rec.get('category'),
        'source_ids': [rec['id']],
        'stages': [to_stage(rec)],
    }


def load(path):
    rows = read_rows(path)

    halves = {}
    for _, rec in rows:
        parsed = stage_of(rec['id'])
        if parsed:
            halves.setdefault(parsed[0], {})[parsed[1]] = rec

    injections, emitted = [], set()
    for lineno, rec in rows:
        parsed = stage_of(rec['id'])
        if not parsed:
            injections.append(single(rec))
            continue

        base, stage = parsed[0], parsed[1]
        pair = halves[base]
        if 'a' not in pair or 'b' not in pair:
            print('WARNING: %s line %d: %s has no matching %s half; treating it as a '
                  'single-stage injection'
                  % (path, lineno, rec['id'], 'b' if stage == 'a' else 'a'),
                  file=sys.stderr)
            injections.append(single(rec))
            continue

        if base in emitted:
            continue
        emitted.add(base)
        primer, activation = pair['a'], pair['b']
        injections.append({
            'id': merged_id(base, stage_of(primer['id'])[2]),
            'category': primer.get('category'),
            'source_ids': [primer['id'], activation['id']],
            'stages': [to_stage(primer), to_stage(activation)],
        })

    return injections


def categories_by_id(injections):
    out = {}
    for inj in injections:
        category = inj.get('category') or 'unknown'
        for name in [inj['id']] + inj['source_ids']:
            out[name] = category
    return out
