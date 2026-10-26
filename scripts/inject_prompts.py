#!/usr/bin/env python3
"""
inject_prompts.py

What it does:
  Replaces the placeholder marker in the attack logs with each prompt-injection payload
  from injections.jsonl, escaped per log-line type so it cannot break out of its line.
  Works on either source corpus without a mode flag, since each holds the marker in its
  own forms -- readable in the surfaces auditd left alone, hexed in the ones it could not.
  attack_logs/attack_logs_hexa/ gets the hex of the prose payload at EVERY site -- its hex auditd fields
  and its free text (Apache UA, sshd, mysql) alike -- so the hex lane holds no readable copy
  of the injection anywhere. A VG_ row rotates its guesses clockwise through the
  markers (marker k gets variant k mod n), so every guess lands in every file.
  An output file that already exists and is newer than its source log, the injections
  jsonl, this script and injection_set.py is skipped, so a re-run only writes what is
  missing or stale; delete a file to force it. Every run also deletes the .txt files in
  each output folder that no longer match an injection id, so a dropped injection is not
  benchmarked from a leftover file.

How to run it:
  python3 inject_prompts.py                                     # the plain lane, defaults

  python3 inject_prompts.py --attack-dir attack_logs/attack_logs_hexa \
      --out-dir attack_logs_injected/attack_logs_injected_hexa                       # the hex lane

  python3 inject_prompts.py --injections jailbreaks/dovsdofs.jsonl \
      --out-dir attack_logs_injected/attack_logs_injected_dovsdofs                   # the DOVSDOFS lane

What it outputs:
  One sub-folder per input log, holding one .txt per injection, e.g.
      attack_logs_injected/attack_logs_injected_plaintext/attack_drupal_cred_access/DO_01_canonical.txt
"""

import argparse
import os
import re
import sys

import injection_set


MARKER_UNDERSCORED = 'this_is_the_injection'
MARKER_SPACED = 'this is the injection'
MARKER_HEX = MARKER_SPACED.encode('utf-8').hex().upper()
MARKER_HEX_UNDERSCORED = MARKER_UNDERSCORED.encode('utf-8').hex().upper()
HEX_MARKERS = (MARKER_HEX, MARKER_HEX_UNDERSCORED)

MARKER_RE = re.compile('|'.join(re.escape(m) for m in
                                (MARKER_HEX, MARKER_HEX_UNDERSCORED,
                                 MARKER_UNDERSCORED, MARKER_SPACED)))
IPV4_RE = re.compile(r'^\d{1,3}(?:\.\d{1,3}){3}\b')

_APACHE_NAMED = {'\b': '\\b', '\n': '\\n', '\r': '\\r', '\t': '\\t',
                 '\v': '\\v', '\f': '\\f', '"': '\\"', '\\': '\\\\'}


def pick_stage(injection, occurrence_index):
    stages = injection['stages']
    stage = stages[min(occurrence_index, len(stages) - 1)]
    variants = stage.get('variants')
    if variants:
        return variants[occurrence_index % len(variants)]
    return stage


def escape_apache(value):
    out = []
    for ch in value:
        if ch in _APACHE_NAMED:
            out.append(_APACHE_NAMED[ch])
        elif ord(ch) < 0x20 or ord(ch) == 0x7f:
            out.append('\\x%02x' % ord(ch))
        else:
            out.append(ch)
    return ''.join(out)


def sanitize_single_line(value):
    value = re.sub(r'[\r\n\t]+', ' ', value)
    return ''.join(ch for ch in value if ord(ch) >= 0x20 and ord(ch) != 0x7f)


def to_audit_hex(value):
    return value.encode('utf-8').hex().upper()


def inject_line(line, next_stage):
    if line.startswith('type='):
        variant, escape = 'tag', sanitize_single_line
    elif IPV4_RE.match(line):
        variant, escape = 'payload', escape_apache
    else:
        variant, escape = 'payload', sanitize_single_line

    def repl(match):
        stage = next_stage()
        token = match.group(0)
        if token in (MARKER_HEX, MARKER_SPACED):
            return to_audit_hex(stage['payload'])
        if token == MARKER_HEX_UNDERSCORED:
            return to_audit_hex(stage['tag'])
        return escape(stage[variant])

    return MARKER_RE.sub(repl, line)


def inject_text(text, injection):
    counter = [0]

    def next_stage():
        stage = pick_stage(injection, counter[0])
        counter[0] += 1
        return stage

    hex_count = sum(text.count(m) for m in HEX_MARKERS + (MARKER_SPACED,))
    out_lines = [inject_line(line, next_stage) for line in text.splitlines()]
    result = '\n'.join(out_lines)
    if text.endswith('\n'):
        result += '\n'
    return result, counter[0], hex_count


def out_subdir(in_path):
    stem, _ = os.path.splitext(os.path.basename(in_path))
    return stem


def out_name(in_path, injection_id):
    _, ext = os.path.splitext(os.path.basename(in_path))
    return '%s%s' % (injection_id, ext or '.txt')


def run(args):
    if not os.path.isdir(args.attack_dir):
        sys.exit('ERROR: --attack-dir not a directory: %s' % args.attack_dir)
    if not os.path.isfile(args.injections):
        sys.exit('ERROR: --injections not found: %s' % args.injections)

    injections = injection_set.load(args.injections)
    files = [f for f in sorted(os.listdir(args.attack_dir))
             if os.path.isfile(os.path.join(args.attack_dir, f))]
    if not files:
        sys.exit('ERROR: no input files under %s' % args.attack_dir)

    stamp = max(os.path.getmtime(p) for p in
                (args.injections, os.path.abspath(__file__),
                 os.path.abspath(injection_set.__file__)))
    os.makedirs(args.out_dir, exist_ok=True)
    written = fresh = removed = 0
    for fname in files:
        in_path = os.path.join(args.attack_dir, fname)
        newer_than = max(stamp, os.path.getmtime(in_path))
        sub_dir = os.path.join(args.out_dir, out_subdir(in_path))
        os.makedirs(sub_dir, exist_ok=True)
        text = None
        for inj in injections:
            out_path = os.path.join(sub_dir, out_name(in_path, inj['id']))
            if os.path.isfile(out_path) and os.path.getmtime(out_path) >= newer_than:
                fresh += 1
                continue
            if text is None:
                print(fname)
                with open(in_path, encoding='utf-8', errors='replace') as fh:
                    text = fh.read()
            injected, count, hex_count = inject_text(text, inj)
            with open(out_path, 'w', encoding='utf-8', newline='\n') as out:
                out.write(injected)
            print('  %-28s %3d markers (%d text / %d hex)  -> %s'
                  % (inj['id'], count, count - hex_count, hex_count, out_path))
            written += 1
        current = {out_name(in_path, inj['id']) for inj in injections}
        for stale in sorted(os.listdir(sub_dir)):
            if stale.endswith('.txt') and stale not in current:
                os.remove(os.path.join(sub_dir, stale))
                print('  %s removed (no longer in %s)'
                      % (os.path.join(sub_dir, stale), args.injections))
                removed += 1
    print('done: %s/ -- %d written, %d already up to date, %d removed'
          % (args.out_dir, written, fresh, removed))


def main():
    ap = argparse.ArgumentParser(
        description='Replace the this_is_the_injection marker in extracted attack logs '
                    'with prompt-injection payloads from injections.jsonl. Files already '
                    'up to date are skipped.')
    ap.add_argument('--attack-dir', default='attack_logs/attack_logs_plaintext',
                    help='folder of extracted attack logs '
                         '(default: attack_logs/attack_logs_plaintext)')
    ap.add_argument('--injections', default='jailbreaks/injections.jsonl',
                    help='injections jsonl (default: jailbreaks/injections.jsonl)')
    ap.add_argument('--out-dir', default='attack_logs_injected/attack_logs_injected_plaintext',
                    help='output folder (default: attack_logs_injected/attack_logs_injected_plaintext)')
    run(ap.parse_args())


if __name__ == '__main__':
    main()
