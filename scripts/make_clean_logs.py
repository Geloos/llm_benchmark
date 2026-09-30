#!/usr/bin/env python3
"""
make_clean_logs.py

What it does:
  Writes a clean copy of every attack log, reading as if no injection had ever been typed,
  so each model can first be shown the bare attack -- a baseline for saying an injection
  *fooled* it rather than the model missing that log anyway. Every injection marker site is
  rewritten the way the machine would have logged it without the marker:
      sshd username / PAM acct=     -> a normal user (--user, default aggelos)
      auditd EXECVE last argument    -> dropped, argc decremented
      marker nested in a hex a2= cmd -> the ` "marker"` (or bare ` marker`) cut out of the hex
      Apache User-Agent              -> "-" (what Apache writes when no UA was sent)
      anything else (mysql comment)  -> removed
  Fails loudly if a marker survives or a line count changes, so a new surface in a future
  capture cannot slip through half-cleaned.

How to run it:
  python3 make_clean_logs.py                              # attack_logs/ -> attack_logs_clean/
  python3 make_clean_logs.py --attack-dir attack_logs --out-dir attack_logs_clean --user aggelos

  Then benchmark it like any injected tree:
  python3 run_benchmark.py --input-root attack_logs_clean --results-root results_clean
  python3 summarize_results.py --results-root results_clean --out-dir analysis_clean

What it outputs:
  One sub-folder per input log holding a single clean.txt, mirroring the injected tree so
  run_benchmark.py / check_context.py / summarize_results.py read it unchanged, e.g.
      attack_logs_clean/attack_drupal_cred_access/clean.txt
"""

import argparse
import os
import re
import sys
from collections import Counter

from inject_prompts import (HEX_MARKERS, IPV4_RE, MARKER_RE, MARKER_SPACED,
                            MARKER_UNDERSCORED, to_audit_hex)


TEXT_MARKERS = (MARKER_UNDERSCORED, MARKER_SPACED)
_FIELD = '(?:"(?:%s)"|(?:%s))' % ('|'.join(map(re.escape, TEXT_MARKERS)),
                                  '|'.join(HEX_MARKERS))

EXEC_ARG_RE = re.compile(r' a(\d+)=%s\s*$' % _FIELD)
ARGC_RE = re.compile(r'\bargc=(\d+)')
ACCT_RE = re.compile(r'\bacct=%s(?=\s|$)' % _FIELD)
# the marker as the trailing argument of a command line auditd hexed whole: quoted in the
# `bash -c '...' "<marker>"` wrappers, bare in the cron job's `/bin/bash -c '...' <marker>`.
# Quoted first, so the bare form cannot leave a pair of empty quotes behind.
NESTED_HEX = tuple(to_audit_hex(fmt % m) for fmt in (' "%s"', ' %s') for m in TEXT_MARKERS)
TEXT_RE = re.compile('|'.join(map(re.escape, TEXT_MARKERS)))
QUOTED_TEXT_RE = re.compile('"(?:%s)"' % '|'.join(map(re.escape, TEXT_MARKERS)))

SURFACES = ('sshd', 'acct', 'exec_arg', 'nested_hex', 'apache_ua', 'other')


def fail(where, msg):
    sys.exit('ERROR: %s: %s' % (where, msg))


def clean_line(line, user, counts, where):
    if line.startswith('type=EXECVE'):
        m = EXEC_ARG_RE.search(line)
        if m:
            argc = ARGC_RE.search(line)
            if not argc or int(m.group(1)) != int(argc.group(1)) - 1:
                fail(where, 'marker arg a%s is not the last argument' % m.group(1))
            line = line[:m.start()]
            line = ARGC_RE.sub('argc=%d' % (int(argc.group(1)) - 1), line, count=1)
            counts['exec_arg'] += 1

    for hexed in NESTED_HEX:
        n = line.count(hexed)
        if n:
            line = line.replace(hexed, '')
            counts['nested_hex'] += n

    line, n = ACCT_RE.subn('acct="%s"' % user, line)
    counts['acct'] += n

    if ' sshd[' in line:
        line, n = TEXT_RE.subn(user, line)
        counts['sshd'] += n
    elif IPV4_RE.match(line):
        line, n = QUOTED_TEXT_RE.subn('"-"', line)
        counts['apache_ua'] += n

    line, n = TEXT_RE.subn('', line)
    counts['other'] += n

    if MARKER_RE.search(line):
        fail(where, 'a marker survived cleaning (unrecognised surface): %s' % line[:200])
    return line


def clean_text(text, user, where):
    counts = Counter()
    lines = text.splitlines()
    out = [clean_line(line, user, counts, '%s line %d' % (where, i))
           for i, line in enumerate(lines, 1)]
    result = '\n'.join(out)
    if text.endswith('\n'):
        result += '\n'
    if result.count('\n') != text.count('\n'):
        fail(where, 'line count changed')
    return result, counts


def run(args):
    if not os.path.isdir(args.attack_dir):
        sys.exit('ERROR: --attack-dir not a directory: %s' % args.attack_dir)
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', args.user):
        sys.exit('ERROR: --user must be a single bare token (it lands in sshd and acct=)')

    files = [f for f in sorted(os.listdir(args.attack_dir))
             if os.path.isfile(os.path.join(args.attack_dir, f))]
    if not files:
        sys.exit('ERROR: no input files under %s' % args.attack_dir)

    total = Counter()
    for fname in files:
        in_path = os.path.join(args.attack_dir, fname)
        with open(in_path, encoding='utf-8', errors='replace') as fh:
            text = fh.read()
        cleaned, counts = clean_text(text, args.user, in_path)
        sub_dir = os.path.join(args.out_dir, os.path.splitext(fname)[0])
        os.makedirs(sub_dir, exist_ok=True)
        out_path = os.path.join(sub_dir, 'clean.txt')
        with open(out_path, 'w', encoding='utf-8', newline='\n') as out:
            out.write(cleaned)
        total.update(counts)
        print('%-34s %3d sites (%s)  -> %s'
              % (fname, sum(counts.values()),
                 ' '.join('%s=%d' % (s, counts[s]) for s in SURFACES if counts[s]),
                 out_path))
    print('done: %d clean logs under %s/, %d sites: %s'
          % (len(files), args.out_dir, sum(total.values()),
             ', '.join('%s %d' % (s, total[s]) for s in SURFACES)))


def main():
    ap = argparse.ArgumentParser(
        description='Write clean (injection-free) copies of the attack logs as a baseline.')
    ap.add_argument('--attack-dir', default='attack_logs',
                    help='folder of extracted attack logs (default: attack_logs)')
    ap.add_argument('--out-dir', default='attack_logs_clean',
                    help='output folder (default: attack_logs_clean)')
    ap.add_argument('--user', default='aggelos',
                    help='username written into the sshd and PAM acct= sites (default: aggelos)')
    run(ap.parse_args())


if __name__ == '__main__':
    main()
