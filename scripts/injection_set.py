#!/usr/bin/env python3
"""
injection_set.py

What it does:
  Loads jailbreaks/injections.jsonl and normalises it into the one shape the rest of the
  pipeline uses. Shared by inject_prompts.py (which needs the payloads) and
  summarize_results.py (which needs id -> category), so both agree on what an "injection"
  is -- in particular on how the two-stage payload_split pairs are folded together.

  Each jsonl row is {"id", "category", "payload", "tag"}: `payload` is the free-text
  variant (Apache user-agent, sshd username, ...), `tag` the underscored auditd-safe
  variant of the same idea.

  Split payloads are written as two rows -- SPLIT_01a_primer_tagrule teaches the rule,
  SPLIT_01b_activation_tagrule fires it -- and only work when both land in the same file.
  load() pairs them into a single injection whose `stages` list is consumed in marker
  order: stage 0 into the first marker, the last stage into every later marker.

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

# SPLIT_01a_primer_tagrule -> base SPLIT_01, stage a, rest _primer_tagrule. The trailing
# digits + single a/b letter are what mark a row as one half of a pair; ordinary ids
# (DO_01_canonical, OB64_01_hinted, AF_02_long_token) have a '_' there and do not match.
STAGE_RE = re.compile(r'^(?P<base>.+_\d+)(?P<stage>[ab])(?P<rest>_.*)?$')

# Dropped from the merged id: the pair is one injection, so "which half" no longer applies.
STAGE_WORD_RE = re.compile(r'_(?:primer|activation)(?=_|$)')

REQUIRED_KEYS = ('id', 'payload', 'tag')


def read_rows(path):
    """Parse the jsonl into (line_number, dict) pairs; exit on a malformed or incomplete
    row rather than failing later inside a regex callback."""
    rows = []
    # utf-8-sig, not utf-8: a Windows editor re-saving the jsonl can prepend a BOM, which
    # would otherwise blow up json.loads on line 1 only.
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
    """('SPLIT_01', 'a', '_primer_tagrule') for one half of a pair, else None."""
    m = STAGE_RE.match(injection_id)
    if not m:
        return None
    return m.group('base'), m.group('stage'), m.group('rest') or ''


def merged_id(base, rest):
    """SPLIT_01 + _primer_tagrule -> SPLIT_01_tagrule (the name the output file gets)."""
    return base + STAGE_WORD_RE.sub('', rest)


def single(rec):
    """One jsonl row -> a one-stage injection."""
    return {
        'id': rec['id'],
        'category': rec.get('category'),
        'source_ids': [rec['id']],
        'stages': [{'payload': rec['payload'], 'tag': rec['tag']}],
    }


def load(path):
    """Read `path` and return the normalised injection list, in file order. Rows whose id
    ends in a stage letter are merged with their partner; a half without its partner is
    kept as an ordinary single-stage injection."""
    rows = read_rows(path)

    # base -> {'a': rec, 'b': rec, ...} so a half can find its partner before we emit.
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

        if base in emitted:            # already emitted when we hit the other half
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
    """id -> category, keyed on both the merged id and every source id, so a result file
    named either way still resolves. Missing categories become 'unknown'."""
    out = {}
    for inj in injections:
        category = inj.get('category') or 'unknown'
        for name in [inj['id']] + inj['source_ids']:
            out[name] = category
    return out
