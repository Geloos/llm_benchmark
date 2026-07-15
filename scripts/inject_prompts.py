#!/usr/bin/env python3
"""
inject_prompts.py

What it does:
  Replaces the `this_is_the_injection` placeholder marker in extracted attack logs
  with real prompt-injection payloads from injections.jsonl.

How to run it:
  python3 inject_prompts.py --attack-dir attack_logs \
      --injections jailbreaks/injections.jsonl --out-dir attack_logs_injected

  Optional: --only-id <injection id> and --only-file <basename> to narrow the run.

What it outputs:
  One sub-folder per input log, holding one .txt per injection, e.g.
      attack_logs_injected/attack_drupal_cred_access/T0_ignore.txt
"""

import argparse
import json
import os
import re
import sys


MARKER = 'this_is_the_injection'
MARKER_HEX = MARKER.encode('utf-8').hex().upper()

MARKER_RE = re.compile(re.escape(MARKER))
MARKER_HEX_RE = re.compile(re.escape(MARKER_HEX))
AUDIT_FIELD_RE = re.compile(r'(\w+)="' + re.escape(MARKER) + r'"')
IPV4_RE = re.compile(r'^\d{1,3}(?:\.\d{1,3}){3}\b')

# Apache mod_log_config (ap_escape_logitem) named escapes; other control chars -> \xHH.
_APACHE_NAMED = {'\b': '\\b', '\n': '\\n', '\r': '\\r', '\t': '\\t',
                 '\v': '\\v', '\f': '\\f', '"': '\\"', '\\': '\\\\'}


def load_injections(path):
    """Read injections.jsonl -> list of dicts (one per non-blank line)."""
    injections = []
    with open(path, encoding='utf-8') as fh:
        for raw in fh:
            raw = raw.strip()
            if raw:
                injections.append(json.loads(raw))
    return injections


def pick_payload(injection, occurrence_index):
    """The payload sub-dict (has 'general'/'tag') to use for this marker occurrence.
    prep+activation entries: occurrence 0 -> prep, later -> activation. Single-stage
    entries: the top-level injection for every occurrence."""
    if 'prep' in injection and 'activation' in injection:
        return injection['prep'] if occurrence_index == 0 else injection['activation']
    return injection


def audit_needs_hex(value):
    """auditd hex-encodes a whole field value if it holds a space, a double-quote, or a
    non-printable char; otherwise it is emitted quoted."""
    return any(c == ' ' or c == '"' or ord(c) < 0x20 or ord(c) > 0x7e for c in value)


def audit_field(key, value):
    """Render `key` + `value` the way auditd would: quoted, or key=<UPPERHEX>."""
    if audit_needs_hex(value):
        return '%s=%s' % (key, value.encode('utf-8').hex().upper())
    return '%s="%s"' % (key, value)


def escape_apache(value):
    """Apache %{User-Agent}i escaping: named escapes for the usual control chars, \\xHH
    for the rest, \\" for quotes, \\\\ for backslash. Keeps the UA a single valid token."""
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
    """Collapse CR/LF/TAB runs to one space and drop other control chars, so an injected
    username (or any free-text field) never splits the log line. Mirrors how sshd cleans
    control chars out of a logged username."""
    value = re.sub(r'[\r\n\t]+', ' ', value)
    return ''.join(ch for ch in value if ord(ch) >= 0x20 and ord(ch) != 0x7f)


def inject_audit_line(line, next_payload):
    """Replace markers in an audit (type=...) line: key="MARKER" -> auditd-rendered tag,
    and the hex-embedded marker inside a2=<hex> -> hex of the tag."""
    def repl_kv(match):
        payload = next_payload()
        return audit_field(match.group(1), payload['tag'])

    def repl_hex(_):
        payload = next_payload()
        return payload['tag'].encode('utf-8').hex().upper()

    line = AUDIT_FIELD_RE.sub(repl_kv, line)
    line = MARKER_HEX_RE.sub(repl_hex, line)
    return line


def inject_freetext_line(line, next_payload):
    """Replace markers in a free-text log line with the escaped `general` payload."""
    if IPV4_RE.match(line):
        escaper = escape_apache          # Apache combined access line -> quoted UA field
    else:
        escaper = sanitize_single_line   # sshd username line (or any other free-text)

    def repl(_):
        payload = next_payload()
        return escaper(payload['general'])

    return MARKER_RE.sub(repl, line)


def inject_line(line, next_payload):
    if line.startswith('type='):
        return inject_audit_line(line, next_payload)
    return inject_freetext_line(line, next_payload)


def inject_text(text, injection):
    """Inject every marker in one file's text, occurrence-order-aware. Returns
    (new_text, marker_count)."""
    counter = [0]

    def next_payload():
        payload = pick_payload(injection, counter[0])
        counter[0] += 1
        return payload

    out_lines = [inject_line(line, next_payload) for line in text.splitlines()]
    result = '\n'.join(out_lines)
    if text.endswith('\n'):
        result += '\n'
    return result, counter[0]


def out_subdir(in_path):
    """attack_drupal_cred_access.txt -> attack_drupal_cred_access (per-log output folder)."""
    stem, _ = os.path.splitext(os.path.basename(in_path))
    return stem


def out_name(in_path, injection_id):
    """attack_drupal_cred_access.txt + T3_soc -> T3_soc.txt (goes under out_subdir())."""
    _, ext = os.path.splitext(os.path.basename(in_path))
    return '%s%s' % (injection_id, ext or '.txt')


def run(args):
    if not os.path.isdir(args.attack_dir):
        sys.exit('ERROR: --attack-dir not a directory: %s' % args.attack_dir)
    if not os.path.isfile(args.injections):
        sys.exit('ERROR: --injections not found: %s' % args.injections)

    injections = load_injections(args.injections)
    if args.only_id:
        injections = [inj for inj in injections if inj.get('id') == args.only_id]
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
            injected, count = inject_text(text, inj)
            out_path = os.path.join(sub_dir, out_name(in_path, inj['id']))
            with open(out_path, 'w', encoding='utf-8') as out:
                out.write(injected)
            print('  %-16s %3d markers  -> %s' % (inj['id'], count, out_path))
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
    ap.add_argument('--only-id', help='only this injection id (default: all)')
    ap.add_argument('--only-file', help='only this input basename (default: all)')
    run(ap.parse_args())


if __name__ == '__main__':
    main()
