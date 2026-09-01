#!/usr/bin/env python3
"""
inject_prompts.py

What it does:
  Replaces the placeholder marker in the attack logs with each prompt-injection payload
  from injections.jsonl, escaped per log-line type so it cannot break out of its line.
  Works on either source corpus without a mode flag, since each holds the marker in its
  own form; --hex-variant picks whether the hex auditd fields carry the prose payload
  (default) or the underscored tag.

How to run it:
  python3 inject_prompts.py --attack-dir attack_logs \
      --injections jailbreaks/injections.jsonl --out-dir attack_logs_injected

  python3 inject_prompts.py --attack-dir attack_logs_hexa \
      --out-dir attack_logs_injected_hexa                       # the hex lane

  python3 inject_prompts.py --attack-dir attack_logs_hexa --hex-variant tag \
      --out-dir attack_logs_injected_control                    # the encoding control

  Optional: --only-id <injection id> and --only-file <basename> to narrow the run.

What it outputs:
  One sub-folder per input log, holding one .txt per injection, e.g.
      attack_logs_injected/attack_drupal_cred_access/DO_01_canonical.txt
"""

import argparse
import os
import re
import sys

import injection_set


MARKER_UNDERSCORED = 'this_is_the_injection'
MARKER_SPACED = 'this is the injection'
MARKER_HEX = MARKER_SPACED.encode('utf-8').hex().upper()

MARKER_RE = re.compile('|'.join(re.escape(m) for m in
                                (MARKER_HEX, MARKER_UNDERSCORED, MARKER_SPACED)))
IPV4_RE = re.compile(r'^\d{1,3}(?:\.\d{1,3}){3}\b')

_APACHE_NAMED = {'\b': '\\b', '\n': '\\n', '\r': '\\r', '\t': '\\t',
                 '\v': '\\v', '\f': '\\f', '"': '\\"', '\\': '\\\\'}


def pick_stage(injection, occurrence_index):
    stages = injection['stages']
    return stages[min(occurrence_index, len(stages) - 1)]


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


def inject_line(line, next_stage, hex_variant='payload'):
    if line.startswith('type='):
        variant, escape = 'tag', sanitize_single_line
    elif IPV4_RE.match(line):
        variant, escape = 'payload', escape_apache
    else:
        variant, escape = 'payload', sanitize_single_line

    def repl(match):
        stage = next_stage()
        if match.group(0) == MARKER_HEX:
            return to_audit_hex(stage[hex_variant])
        return escape(stage[variant])

    return MARKER_RE.sub(repl, line)


def inject_text(text, injection, hex_variant='payload'):
    counter = [0]

    def next_stage():
        stage = pick_stage(injection, counter[0])
        counter[0] += 1
        return stage

    hex_count = text.count(MARKER_HEX)
    out_lines = [inject_line(line, next_stage, hex_variant)
                 for line in text.splitlines()]
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
    if args.only_id:
        injections = [inj for inj in injections
                      if args.only_id in [inj['id']] + inj['source_ids']]
        if not injections:
            sys.exit('ERROR: no injection with id %s in %s' % (args.only_id, args.injections))

    files = [f for f in sorted(os.listdir(args.attack_dir))
             if os.path.isfile(os.path.join(args.attack_dir, f))]
    if args.only_file:
        files = [f for f in files if f == args.only_file]
        if not files:
            sys.exit('ERROR: %s not in %s' % (args.only_file, args.attack_dir))
    if not files:
        sys.exit('ERROR: no input files under %s' % args.attack_dir)

    os.makedirs(args.out_dir, exist_ok=True)
    grand = 0
    for fname in files:
        in_path = os.path.join(args.attack_dir, fname)
        with open(in_path, encoding='utf-8', errors='replace') as fh:
            text = fh.read()
        sub_dir = os.path.join(args.out_dir, out_subdir(in_path))
        os.makedirs(sub_dir, exist_ok=True)
        print(fname)
        for inj in injections:
            injected, count, hex_count = inject_text(text, inj, args.hex_variant)
            out_path = os.path.join(sub_dir, out_name(in_path, inj['id']))
            with open(out_path, 'w', encoding='utf-8', newline='\n') as out:
                out.write(injected)
            print('  %-28s %3d markers (%d text / %d hex)  -> %s'
                  % (inj['id'], count, count - hex_count, hex_count, out_path))
            grand += 1
    print('done: %d injected files under %s/' % (grand, args.out_dir))


def main():
    ap = argparse.ArgumentParser(
        description='Replace the this_is_the_injection marker in extracted attack logs '
                    'with prompt-injection payloads from injections.jsonl.')
    ap.add_argument('--attack-dir', default='attack_logs',
                    help='folder of extracted attack logs (default: attack_logs)')
    ap.add_argument('--injections', default='jailbreaks/injections.jsonl',
                    help='injections jsonl (default: jailbreaks/injections.jsonl)')
    ap.add_argument('--out-dir', default='attack_logs_injected',
                    help='output folder (default: attack_logs_injected)')
    ap.add_argument('--hex-variant', choices=('payload', 'tag'), default='payload',
                    help="which text to hex into auditd fields: 'payload' (default, the "
                         "real-attacker case) or 'tag' (the CONTROL lane -- same text as "
                         "attack_logs/ carries, so plain-vs-this isolates the encoding)")
    ap.add_argument('--only-id', help='only this injection id (default: all)')
    ap.add_argument('--only-file', help='only this input basename (default: all)')
    run(ap.parse_args())


if __name__ == '__main__':
    main()
