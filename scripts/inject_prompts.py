#!/usr/bin/env python3
"""
inject_prompts.py

What it does:
  Replaces the placeholder marker in extracted attack logs with real prompt-injection
  payloads from injections.jsonl.

  Works on either source corpus without a mode flag, because the two hold the marker in
  disjoint forms and the form itself records how the machine treated it:

    attack_logs/       captured with the marker typed as one underscored token, so auditd
                       left it quoted:  a3="this_is_the_injection"
    attack_logs_hexa/  captured with it typed WITH SPACES, so auditd hex-encoded it and
                       dropped the quotes:
                         acct=746869732069732074686520696E6A656374696F6E
                       -- sometimes nested inside a bigger blob, e.g. an a2= holding a
                       whole `bash -c 'id' "this is the injection"` command line.

  --hex-variant picks what goes INTO those hex fields, which is how the encoding effect is
  told apart from the wording effect:

    payload  (default)  the prose payload, spaces and all -- what a real attacker who typed
                        spaces actually produces. Differs from attack_logs/ in BOTH encoding
                        and wording, since a quoted field there can only hold the `tag`.
    tag                 the underscored tag: byte-identical text to what attack_logs/
                        carries, merely hex-encoded. A synthetic CONTROL -- auditd would
                        never hex a space-free token -- so plain-vs-this isolates the
                        encoding, and this-vs-payload isolates the wording.

How to run it:
  python3 inject_prompts.py --attack-dir attack_logs \
      --injections jailbreaks/injections.jsonl --out-dir attack_logs_injected

  python3 inject_prompts.py --attack-dir attack_logs_hexa \
      --out-dir attack_logs_injected_hexa            # the hex lane

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


# The placeholder, in the three shapes the two source corpora hold it in. attack_logs/
# was captured with the marker typed as one underscored token, so auditd left it a
# quoted value. attack_logs_hexa/ was captured with it typed WITH SPACES, so auditd did
# what it does to any string it cannot leave as a bare token: hex-encoded it and dropped
# the quotes. The free-text surfaces (Apache UA, sshd username, mysql) have no such rule
# and still hold it verbatim.
MARKER_UNDERSCORED = 'this_is_the_injection'         # attack_logs/, every surface
MARKER_SPACED = 'this is the injection'              # attack_logs_hexa/, free text
MARKER_HEX = MARKER_SPACED.encode('utf-8').hex().upper()  # attack_logs_hexa/, auditd

# One alternation over all three, so occurrence order stays file order even where the
# forms mix -- pick_stage() counts occurrences to decide which SPLIT_ stage fires.
# Longest first: the forms share no prefix today, but that ordering is what stops a
# future marker edit from quietly matching a shorter alternative first.
MARKER_RE = re.compile('|'.join(re.escape(m) for m in
                                (MARKER_HEX, MARKER_UNDERSCORED, MARKER_SPACED)))
IPV4_RE = re.compile(r'^\d{1,3}(?:\.\d{1,3}){3}\b')

# Apache mod_log_config (ap_escape_logitem) named escapes; other control chars -> \xHH.
_APACHE_NAMED = {'\b': '\\b', '\n': '\\n', '\r': '\\r', '\t': '\\t',
                 '\v': '\\v', '\f': '\\f', '"': '\\"', '\\': '\\\\'}


def pick_stage(injection, occurrence_index):
    """The stage dict (has 'payload'/'tag') to use for this marker occurrence. Two-stage
    (paired SPLIT_*) injections: occurrence 0 -> primer, later -> activation. Single-stage
    injections have one stage, so every occurrence gets it."""
    stages = injection['stages']
    return stages[min(occurrence_index, len(stages) - 1)]


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


def to_audit_hex(value):
    """auditd's own encoding for a field it cannot leave as a bare token: the raw UTF-8
    bytes as uppercase hex, no quotes. Unlike the two escapers above this needs no
    escaping at all -- hex cannot split a line, which is exactly why auditd reaches for
    it -- so the payload goes in byte for byte, newlines and double-quotes included.
    Safe to apply to a marker sitting INSIDE a larger hex blob (an a2= holding a whole
    `bash -c '...' "<marker>"` command line), because hex distributes over
    concatenation: substituting the marker's hex there gives the same bytes as
    substituting in the decoded string and re-encoding the whole thing."""
    return value.encode('utf-8').hex().upper()


def inject_line(line, next_stage, hex_variant='payload'):
    """Replace every marker in one log line with this stage's injection text, rendered for
    the surface it lands on so it stays a single line -- an injected file must keep the same
    line count as its source log. Which of the three variants applies is decided by the
    MARKER FORM that matched, not by the line type, because the form is what records how
    auditd treated the original placeholder:

      hex marker      -> to_audit_hex(stage[hex_variant]): uppercase hex of the raw text.
                         The placeholder had spaces, so auditd hexed it; hex holds anything,
                         so the text goes in unsanitized. `hex_variant` picks WHICH text:
                         'payload' (the default, and what a real attacker typing spaces
                         produces) or 'tag' -- see --hex-variant.
      quoted auditd   -> the underscored `tag`, dropped verbatim into the a3=/a4=/acct=
                         field the marker already sits in -- so a tag has to stay one token,
                         no spaces and no double-quote.
      free text       -> the `payload`, Apache-escaped on an access line, control-char
                         sanitized on an sshd/mysql line.

    The rest of the line is copied through byte for byte: hex fields stay hex and quoted
    fields stay quoted, exactly as the log holds them."""
    if line.startswith('type='):
        variant, escape = 'tag', sanitize_single_line
    elif IPV4_RE.match(line):
        variant, escape = 'payload', escape_apache         # Apache access line -> quoted UA
    else:
        variant, escape = 'payload', sanitize_single_line  # sshd username line, or free text

    # A callable repl, so re.sub uses the return value as-is instead of reinterpreting the
    # backslash escapes escape_apache emits.
    def repl(match):
        stage = next_stage()
        if match.group(0) == MARKER_HEX:
            return to_audit_hex(stage[hex_variant])
        return escape(stage[variant])

    return MARKER_RE.sub(repl, line)


def inject_text(text, injection, hex_variant='payload'):
    """Inject every marker in one file's text, occurrence-order-aware. Returns
    (new_text, marker_count, hex_marker_count) -- the split is per marker FORM, so a
    corpus captured wrong (an all-text count on the hex corpus, say) shows up in the inject
    log instead of only after a full benchmark run."""
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
            injected, count, hex_count = inject_text(text, inj, args.hex_variant)
            out_path = os.path.join(sub_dir, out_name(in_path, inj['id']))
            # newline='\n' so a run on Windows emits the same LF-terminated bytes a run on
            # the cluster does -- real auditd/syslog is LF, and CRLF would both inflate the
            # token count and hand the classifier an artifact no SOC pipeline produces.
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
