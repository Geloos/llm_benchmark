#!/usr/bin/env python3
"""
make_plain_logs.py

What it does:
  Builds the plain corpus (attack_logs/) from the hex capture (attack_logs_hexa/) by
  rewriting the placeholder the way the machine would have logged `this_is_the_injection`
  instead of `this is the injection`:
      free text (Apache UA, sshd, mysql) -> this_is_the_injection
      auditd field holding only the marker (a3=/a4=/acct=<hex>)
                                         -> a3="this_is_the_injection"  (no space left, so
                                            auditd would have quoted it instead of hexing)
      marker nested in a hexed command (a2=<hex of bash -c '...' "marker">)
                                         -> still hex (the command keeps its spaces), with the
                                            underscored marker inside
  Every touched auditd field is decoded, rewritten and re-encoded with auditd's own rule
  (hex if any byte is '"', a space/control char or non-ASCII, else quoted), so the hex ->
  quoted switch and the stays-hex case both fall out of one rule. Nothing else changes:
  the same arguments are present, so argc holds, and these logs carry no length fields.
  The sshd brute force names the user in its three text lines and the PAM USER_AUTH acct=,
  all rewritten here; its USER_LOGIN records only ever say "(invalid user)".
  Exits 1 if a spaced marker survives, a hex field holding it does not decode, the input
  already holds the underscored form, the site count does not add up, or a line count
  changes.

How to run it:
  python3 scripts/make_plain_logs.py                 # attack_logs_hexa/ -> attack_logs/
  python3 scripts/make_plain_logs.py --attack-dir attack_logs_hexa --out-dir attack_logs

What it outputs:
  One file per input log, same extensionless name, e.g. attack_logs/attack_cron, plus a
  per-file count of the rewritten sites by surface.
"""

import argparse
import os
import re
import sys
from collections import Counter

from inject_prompts import (IPV4_RE, MARKER_HEX, MARKER_HEX_UNDERSCORED, MARKER_SPACED,
                            MARKER_UNDERSCORED)


FIELD_RE = re.compile(r'\b(a\d+|acct)=("[^"]*"|[0-9A-F]+)(?=\s|\'|$)')

SURFACES = ('apache', 'sshd', 'mysql', 'other_text', 'field', 'nested_hex')


def fail(where, msg):
    sys.exit('ERROR: %s: %s' % (where, msg))


def audit_encode(value):
    # auditd's audit_value_needs_encoding(): a '"', anything <= 0x20 or >= 0x7f forces hex
    if any(ch == '"' or ord(ch) <= 0x20 or ord(ch) >= 0x7f for ch in value):
        return value.encode('utf-8', 'surrogateescape').hex().upper()
    return '"%s"' % value


def rewrite_field(match, counts, where):
    name, raw = match.group(1), match.group(2)
    if raw.startswith('"') or MARKER_HEX not in raw:
        return match.group(0)
    try:
        value = bytes.fromhex(raw).decode('utf-8', 'surrogateescape')
    except ValueError:
        fail(where, '%s= holds the marker hex but does not decode: %s' % (name, raw[:80]))
    n = value.count(MARKER_SPACED)
    if not n:
        fail(where, '%s= holds the marker hex off a byte boundary: %s' % (name, raw[:80]))
    counts['field' if value == MARKER_SPACED else 'nested_hex'] += n
    return '%s=%s' % (name, audit_encode(value.replace(MARKER_SPACED, MARKER_UNDERSCORED)))


def plain_line(line, counts, where):
    if line.startswith('type='):
        line = FIELD_RE.sub(lambda m: rewrite_field(m, counts, where), line)
    else:
        n = line.count(MARKER_SPACED)
        if n:
            if IPV4_RE.match(line):
                surface = 'apache'
            elif ' sshd[' in line:
                surface = 'sshd'
            elif ' Query\t' in line:
                surface = 'mysql'
            else:
                surface = 'other_text'
            counts[surface] += n
            line = line.replace(MARKER_SPACED, MARKER_UNDERSCORED)

    if MARKER_SPACED in line or MARKER_HEX in line:
        fail(where, 'a spaced marker survived (unrecognised surface): %s' % line[:200])
    return line


def plain_text(text, where):
    if MARKER_UNDERSCORED in text or MARKER_HEX_UNDERSCORED in text:
        fail(where, 'already holds the underscored marker -- is this the hex capture?')
    sites = text.count(MARKER_SPACED) + text.count(MARKER_HEX)

    counts = Counter()
    lines = text.splitlines(keepends=True)
    out = []
    for i, line in enumerate(lines, 1):
        body = line.rstrip('\r\n')
        out.append(plain_line(body, counts, '%s line %d' % (where, i)) + line[len(body):])
    result = ''.join(out)

    if sum(counts.values()) != sites:
        fail(where, 'rewrote %d sites but the file holds %d' % (sum(counts.values()), sites))
    if len(result.splitlines()) != len(lines):
        fail(where, 'line count changed')
    return result, counts


def run(args):
    if not os.path.isdir(args.attack_dir):
        sys.exit('ERROR: --attack-dir not a directory: %s' % args.attack_dir)
    files = [f for f in sorted(os.listdir(args.attack_dir))
             if os.path.isfile(os.path.join(args.attack_dir, f))]
    if not files:
        sys.exit('ERROR: no input files under %s' % args.attack_dir)

    os.makedirs(args.out_dir, exist_ok=True)
    total = Counter()
    for fname in files:
        in_path = os.path.join(args.attack_dir, fname)
        with open(in_path, encoding='utf-8', errors='surrogateescape', newline='') as fh:
            text = fh.read()
        plain, counts = plain_text(text, in_path)
        out_path = os.path.join(args.out_dir, fname)
        with open(out_path, 'w', encoding='utf-8', errors='surrogateescape',
                  newline='') as out:
            out.write(plain)
        total.update(counts)
        print('%-30s %3d sites (%s)  -> %s'
              % (fname, sum(counts.values()),
                 ' '.join('%s=%d' % (s, counts[s]) for s in SURFACES if counts[s]),
                 out_path))
    print('done: %d plain logs under %s/, %d sites: %s'
          % (len(files), args.out_dir, sum(total.values()),
             ', '.join('%s %d' % (s, total[s]) for s in SURFACES)))


def main():
    ap = argparse.ArgumentParser(
        description='Build the plain corpus (this_is_the_injection) from the hex capture.')
    ap.add_argument('--attack-dir', default='attack_logs_hexa',
                    help='the hex capture (default: attack_logs_hexa)')
    ap.add_argument('--out-dir', default='attack_logs',
                    help='output folder for the plain corpus (default: attack_logs)')
    run(ap.parse_args())


if __name__ == '__main__':
    main()
