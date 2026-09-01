#!/usr/bin/env python3
"""
injection_set.py

What it does:
  Loads jailbreaks/injections.jsonl and normalises it into the one shape the rest of the
  pipeline uses, folding the two-stage SPLIT_ rows (a primer + its activation) into a
  single injection. Imported by both inject_prompts.py and summarize_results.py so the
  two agree on what an injection is.

How to run it:
  Not a CLI. Import it:  import injection_set;  injections = injection_set.load(path)

What it outputs:
  A list of dicts:
      {"id": "SPLIT_01_tagrule", "category": "payload_split",
       "source_ids": ["SPLIT_01a_primer_tagrule", "SPLIT_01b_activation_tagrule"],
       "stages": [{"payload": ..., "tag": ...}, {"payload": ..., "tag": ...}]}
  Single-stage entries get a one-element `stages` list and source_ids == [id].
"""

import json
import re
import sys

STAGE_RE = re.compile(r'^(?P<base>.+_\d+)(?P<stage>[ab])(?P<rest>_.*)?$')

STAGE_WORD_RE = re.compile(r'_(?:primer|activation)(?=_|$)')

REQUIRED_KEYS = ('id', 'payload', 'tag')


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
            missing = [k for k in REQUIRED_KEYS if not rec.get(k)]
            if missing:
                sys.exit('ERROR: %s line %d (%s) is missing: %s'
                         % (path, lineno, rec.get('id', '?'), ', '.join(missing)))
            rows.append((lineno, rec))
    return rows


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
        'stages': [{'payload': rec['payload'], 'tag': rec['tag']}],
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
            'stages': [{'payload': primer['payload'], 'tag': primer['tag']},
                       {'payload': activation['payload'], 'tag': activation['tag']}],
        })

    return injections


def categories_by_id(injections):
    out = {}
    for inj in injections:
        category = inj.get('category') or 'unknown'
        for name in [inj['id']] + inj['source_ids']:
            out[name] = category
    return out
