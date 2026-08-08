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
  --audit-view raw keeps auditd lines verbatim (payload hex-encoded); the default
  `interpreted` decodes them the way ausearch -i / auditbeat / wazuh hand them to an
  analyst, so a hex-encoded payload is not accidentally an obfuscation test.

What it outputs:
  One sub-folder per input log, holding one .txt per injection, e.g.
      attack_logs_injected/attack_drupal_cred_access/DO_01_canonical.txt
"""

import argparse
import os
import re
import sys

import injection_set


MARKER = 'this_is_the_injection'
MARKER_HEX = MARKER.encode('utf-8').hex().upper()

MARKER_RE = re.compile(re.escape(MARKER))
MARKER_HEX_RE = re.compile(re.escape(MARKER_HEX))
AUDIT_FIELD_RE = re.compile(r'(\w+)="' + re.escape(MARKER) + r'"')
IPV4_RE = re.compile(r'^\d{1,3}(?:\.\d{1,3}){3}\b')

# --- ausearch -i view -------------------------------------------------------------
# auditd hex-encodes a field only when its value is "unsafe" (space, quote, control
# char); every analyst-facing tool -- ausearch -i, aureport, auditbeat, wazuh, the
# splunk auditd TA -- decodes it back before a human or a classifier ever sees it.
# Only these keys are ever hex-encoded, so only these are safe to decode: a bare
# `pid=2876` is decimal, not hex, and must be left alone.
AUDIT_HEX_KEYS = {'proctitle', 'cwd', 'name', 'exe', 'comm', 'key', 'acct', 'path',
                  'dir', 'cmd', 'data', 'file', 'ocomm', 'old', 'new'}
AUDIT_ARG_RE = re.compile(r'a\d+\Z')
AUDIT_HEXVAL_RE = re.compile(r'(?<![\w=])([a-z][\w-]*)=([0-9A-Fa-f]{2,})(?=\s|$)')
AUDIT_QUOTED_RE = re.compile(r'([a-z][\w-]*)="([^"\n]*)"')

# Apache mod_log_config (ap_escape_logitem) named escapes; other control chars -> \xHH.
_APACHE_NAMED = {'\b': '\\b', '\n': '\\n', '\r': '\\r', '\t': '\\t',
                 '\v': '\\v', '\f': '\\f', '"': '\\"', '\\': '\\\\'}


def pick_stage(injection, occurrence_index):
    """The stage dict (has 'payload'/'tag') to use for this marker occurrence. Two-stage
    (paired SPLIT_*) injections: occurrence 0 -> primer, later -> activation. Single-stage
    injections have one stage, so every occurrence gets it."""
    stages = injection['stages']
    return stages[min(occurrence_index, len(stages) - 1)]


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


def decode_hex_value(hexstr):
    """The text behind an auditd hex field, or None if it is not decodable text (then the
    field stays hex, exactly as a real tool would leave it). NUL is what auditd uses to
    separate proctitle argv entries; ausearch renders those as spaces."""
    if len(hexstr) % 2:
        return None
    try:
        text = bytes.fromhex(hexstr).decode('utf-8')
    except (ValueError, UnicodeDecodeError):
        return None
    text = text.replace('\x00', ' ')
    if any(ord(c) < 0x20 or ord(c) == 0x7f for c in text):
        return None
    return text


def interpret_audit_line(line):
    """Render one auditd line the way `ausearch -i` does: hex values decoded to text and
    quotes dropped. Applied to EVERY type= line, injected or not -- decoding just the
    injected field would leave the payload as the one readable string on a line of hex,
    which is an artifact the classifier could key on instead of the payload's content."""
    # a0..aN hold argv only on EXECVE; on SYSCALL they are pointer addresses, so decoding
    # them there would turn an address into gibberish text on the rare line whose bytes
    # happen to be printable. ausearch draws the same distinction from the record type.
    is_execve = line.startswith('type=EXECVE')

    def repl_hex(match):
        key, value = match.group(1), match.group(2)
        if AUDIT_ARG_RE.match(key):
            if not is_execve:
                return match.group(0)
        elif key not in AUDIT_HEX_KEYS:
            return match.group(0)
        text = decode_hex_value(value)
        return match.group(0) if text is None else '%s=%s' % (key, text)

    line = AUDIT_HEXVAL_RE.sub(repl_hex, line)
    return AUDIT_QUOTED_RE.sub(lambda m: '%s=%s' % (m.group(1), m.group(2)), line)


def inject_audit_line(line, next_stage):
    """Replace markers in an audit (type=...) line: key="MARKER" -> auditd-rendered tag,
    and the hex-embedded marker inside a2=<hex> -> hex of the tag."""
    def repl_kv(match):
        return audit_field(match.group(1), next_stage()['tag'])

    def repl_hex(_):
        return next_stage()['tag'].encode('utf-8').hex().upper()

    line = AUDIT_FIELD_RE.sub(repl_kv, line)
    line = MARKER_HEX_RE.sub(repl_hex, line)
    return line


def inject_freetext_line(line, next_stage):
    """Replace markers in a free-text log line with the escaped free-text payload."""
    if IPV4_RE.match(line):
        escaper = escape_apache          # Apache combined access line -> quoted UA field
    else:
        escaper = sanitize_single_line   # sshd username line (or any other free-text)

    def repl(_):
        return escaper(next_stage()['payload'])

    return MARKER_RE.sub(repl, line)


def inject_interpreted_audit_line(line, next_stage):
    """Audit line in the ausearch -i view: no hex, no quotes, so the marker is plain text
    and the tag goes in as-is (control chars stripped so the line cannot split)."""
    def repl(_):
        return sanitize_single_line(next_stage()['tag'])

    return MARKER_RE.sub(repl, line)


def inject_line(line, next_stage, audit_view):
    if line.startswith('type='):
        if audit_view == 'interpreted':
            # interpret FIRST, inject after: decoding a line that already holds the
            # payload would decode the payload too, which would quietly destroy the
            # OHEX_ category (its payloads are hex strings on purpose).
            return inject_interpreted_audit_line(interpret_audit_line(line), next_stage)
        return inject_audit_line(line, next_stage)
    return inject_freetext_line(line, next_stage)


def inject_text(text, injection, audit_view='interpreted'):
    """Inject every marker in one file's text, occurrence-order-aware. Returns
    (new_text, marker_count)."""
    counter = [0]

    def next_stage():
        stage = pick_stage(injection, counter[0])
        counter[0] += 1
        return stage

    out_lines = [inject_line(line, next_stage, audit_view) for line in text.splitlines()]
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

    injections = injection_set.load(args.injections)
    if args.only_id:
        # match the merged id (SPLIT_01_tagrule) or either half (SPLIT_01a_primer_tagrule)
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
            injected, count = inject_text(text, inj, args.audit_view)
            out_path = os.path.join(sub_dir, out_name(in_path, inj['id']))
            # newline='\n' so a run on Windows emits the same LF-terminated bytes a run on
            # the cluster does -- real auditd/syslog is LF, and CRLF would both inflate the
            # token count and hand the classifier an artifact no SOC pipeline produces.
            with open(out_path, 'w', encoding='utf-8', newline='\n') as out:
                out.write(injected)
            print('  %-28s %3d markers  -> %s' % (inj['id'], count, out_path))
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
    ap.add_argument('--audit-view', choices=('interpreted', 'raw'), default='interpreted',
                    help='how auditd lines are presented. interpreted (default) = the '
                         '`ausearch -i` view a SOC pipeline actually delivers: hex '
                         'decoded, quotes dropped. raw = verbatim /var/log/audit/audit.log '
                         'with the payload hex-encoded, which also makes every auditd '
                         'injection an obfuscation test')
    run(ap.parse_args())


if __name__ == '__main__':
    main()
