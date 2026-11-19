#!/usr/bin/env python3
"""Regenerate tests/unit/qwen/qwen_encode_vectors.h (differential encode vectors).

Self-contained: reads the tokenizer arrays (tokenizer.ggml.tokens / .merges /
.token_type) DIRECTLY from the GGUF checkpoint with a minimal built-in GGUF v3
metadata reader -- no intermediate JSON, no cached inputs. The reference
encoder is independent of the C++ tokenizer: the third-party `regex` module
provides the exact qwen35 pre-tokenization pattern (Unicode categories), and
BPE is plain rank-based merging over the GGUF merge table.

Usage (from the repository root):
    python tests/unit/qwen/generate_qwen_encode_vectors.py \
        [--gguf data/Qwen3.5-4B-BF16.gguf] [--output tests/unit/qwen/qwen_encode_vectors.h] [--check]

--check regenerates in memory and exits nonzero if the committed header would
change (vector drift). Requires: pip install regex
"""
import argparse
import pathlib
import struct
import sys

try:
    import regex as re
except ImportError:
    sys.exit("the third-party 'regex' module is required: python -m pip install regex")

# Exact qwen35 pre-tokenizer pattern (llama.cpp unicode.cpp "qwen35" rule).
PAT = re.compile(
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
    r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

# The differential sample set (fixed; changing it changes the header).
SAMPLES = [
    "Hello, world!", "The quick brown fox jumps over the lazy dog.",
    "line1\nline2\r\n\tindented", "  spaces  everywhere  ",
    "na\u00efve caf\u00e9 r\u00e9sum\u00e9",
    "nai\u0308ve",  # decomposed
    "\u4f60\u597d\uff0c\u4e16\u754c\uff01",
    "\U0001f600 emoji \U0001f389",
    "mix 12345 of 67890 digits and symbols!!!",
    "don't stop believing", "DON'T STOP", "it's IT'S we're WE'RE they'll THEY'LL",
    "a\tb\u00a0c", "a \t\n b", "a!\r\nb", "a  b", "a  ", "!!!", " !!!",
    "\u0661\u0662", "\u0915\u093e", "\u0915\u093e\u092e",
    "x\t y", "\u4f60\u597d", "hello", " hello", "hello world", "123",
    "a\nb", "a \nb", "a!!!b", "", " ", "\n", "\r\n", " \r\n ",
    "tab\t\tend", "v1.2.3", "end.", "end. ", "(paren) [bracket]",
    "\u200bzero\u200bwidth", "\U0001f600\U0001f601",
    "mixed\u4e2d\u6587text123", "e\u0301cole", "\u00a0\u00a0",
    "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n\n\n",
]


# ---------------------------------------------------------------------------
# Minimal GGUF v3 metadata reader (little-endian; header + KV section only --
# tensor infos are skipped without reading any tensor payload).
# ---------------------------------------------------------------------------
class Reader:
    def __init__(self, f):
        self.f = f

    def take(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise ValueError("truncated GGUF metadata")
        return b

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        return self.take(self.u64()).decode("utf-8")

    def value(self, t):
        if t == 0: return struct.unpack("<B", self.take(1))[0]   # u8
        if t == 1: return struct.unpack("<b", self.take(1))[0]   # i8
        if t == 2: return struct.unpack("<H", self.take(2))[0]   # u16
        if t == 3: return struct.unpack("<h", self.take(2))[0]   # i16
        if t == 4: return self.u32()                             # u32
        if t == 5: return struct.unpack("<i", self.take(4))[0]   # i32
        if t == 6: return struct.unpack("<f", self.take(4))[0]   # f32
        if t == 7: return struct.unpack("<B", self.take(1))[0]   # bool
        if t == 8: return self.string()                          # string
        if t == 9:                                               # array
            et = self.u32()
            n = self.u64()
            return [self.value(et) for _ in range(n)]
        if t == 10: return self.u64()                            # u64
        if t == 11: return struct.unpack("<q", self.take(8))[0]  # i64
        if t == 12: return struct.unpack("<d", self.take(8))[0]  # f64
        raise ValueError(f"unknown GGUF metadata type {t}")


def read_tokenizer_meta(path):
    want = {"tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.token_type"}
    meta = {}
    with open(path, "rb") as f:
        r = Reader(f)
        if r.take(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        version = r.u32()
        if version not in (2, 3):
            raise ValueError(f"{path}: unsupported GGUF version {version}")
        _tensor_count = r.u64()
        kv_count = r.u64()
        for _ in range(kv_count):
            key = r.string()
            t = r.u32()
            if key in want:
                meta[key] = r.value(t)
            else:
                r.value(t)  # skip (still parsed to advance the stream)
            if want <= meta.keys():
                break
    missing = want - meta.keys()
    if missing:
        raise ValueError(f"{path}: missing tokenizer metadata: {sorted(missing)}")
    return meta


# ---------------------------------------------------------------------------
# Independent reference encoder.
# ---------------------------------------------------------------------------
def pretokenize(text):
    # finditer skips unmatched chars (llama.cpp emits them alone); replicate.
    out = []
    pos = 0
    for m in PAT.finditer(text):
        if m.start() > pos:
            out.extend(text[pos:m.start()])
        out.append(m.group())
        pos = m.end()
    if pos < len(text):
        out.extend(text[pos:])
    return out


def byte_encoder():
    # GPT-2 bytes-to-unicode table.
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(0xa1, 0xad)) + list(range(0xae, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, cs))


class Encoder:
    def __init__(self, meta):
        toks = meta["tokenizer.ggml.tokens"]
        merges = meta["tokenizer.ggml.merges"]
        types = meta["tokenizer.ggml.token_type"]
        self.id_of = {t: i for i, t in enumerate(toks)}
        self.rank = {}
        for r, m in enumerate(merges):
            a, b = m.split(' ')
            self.rank[(a, b)] = r
        self.b2c = byte_encoder()
        # Specials (types 3=control, 4=user-defined): longest content first,
        # ties broken by id (matches the C++ matcher).
        self.specials = sorted(
            [(i, toks[i]) for i in range(len(toks)) if types[i] in (3, 4)],
            key=lambda t: (-len(t[1]), t[0]))

    def byte_encode(self, s):
        return ''.join(chr(self.b2c[b]) for b in s.encode('utf-8'))

    def bpe(self, piece):
        syms = list(piece)
        while len(syms) > 1:
            best = None
            for i in range(len(syms) - 1):
                r = self.rank.get((syms[i], syms[i + 1]))
                if r is not None and (best is None or r < best[0]):
                    best = (r, i)
            if best is None:
                break
            i = best[1]
            syms[i:i + 2] = [syms[i] + syms[i + 1]]
        return [self.id_of[s] for s in syms]

    def encode(self, text):
        ids = []
        pos = 0
        normal = ''

        def flush():
            nonlocal normal
            if not normal:
                return
            for pre in pretokenize(normal):
                ids.extend(self.bpe(self.byte_encode(pre)))
            normal = ''

        while pos < len(text):
            matched = False
            for sid, content in self.specials:
                if text.startswith(content, pos):
                    flush()
                    ids.append(sid)
                    pos += len(content)
                    matched = True
                    break
            if not matched:
                normal += text[pos]
                pos += 1
        flush()
        return ids


# ---------------------------------------------------------------------------
# Header emission (byte-for-byte the committed format).
# ---------------------------------------------------------------------------
def c_escape(text):
    out = []
    for b in text.encode('utf-8'):
        c = chr(b)
        if c == '"':
            out.append('\\"')
        elif c == '\\':
            out.append('\\\\')
        elif c == '\n':
            out.append('\\n')
        elif c == '\r':
            out.append('\\r')
        elif c == '\t':
            out.append('\\t')
        elif 32 <= b < 127:
            out.append(c)
        else:
            # Octal, fixed 3 digits (MSVC hex escapes are greedy).
            out.append('\\%03o' % b)
    return ''.join(out)


def render_header(vectors):
    lines = [
        "// Differential encode vectors: independent Python reference (regex module",
        "// with the exact qwen35 pattern + rank-based BPE over the real GGUF vocab).",
        "// Generated by tests/unit/qwen/generate_qwen_encode_vectors.py (reads",
        "// data/Qwen3.5-4B-BF16.gguf directly); do not edit by hand.",
        "// Non-ASCII bytes use octal escapes (MSVC hex escapes are greedy).",
        "#pragma once",
        "namespace luisa::qwen::test_vectors {",
        "struct EncodeVector { const char *text; const unsigned int *ids; unsigned int count; };",
    ]
    for i, (_, ids) in enumerate(vectors):
        if ids:
            body = ','.join(str(x) for x in ids)
        else:
            body = '0u'
        suffix = '// empty (count 0)' if not ids else ''
        lines.append(f"inline constexpr unsigned int kIds{i}[] = {{{body}}};{suffix}")
    lines.append("inline constexpr EncodeVector kEncodeVectors[] = {")
    for i, (text, ids) in enumerate(vectors):
        lines.append(f'    {{"{c_escape(text)}", kIds{i}, {len(ids)}}},')
    lines.append("};")
    lines.append("}// namespace luisa::qwen::test_vectors")
    return '\n'.join(lines) + '\n'


def main():
    root = pathlib.Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", default=str(root / "data" / "Qwen3.5-4B-BF16.gguf"))
    ap.add_argument("--output", default=str(root / "tests" / "unit" / "qwen" / "qwen_encode_vectors.h"))
    ap.add_argument("--check", action="store_true",
                    help="fail if regeneration would change the committed header")
    args = ap.parse_args()

    meta = read_tokenizer_meta(args.gguf)
    enc = Encoder(meta)
    vectors = [(s, enc.encode(s)) for s in SAMPLES]
    content = render_header(vectors)
    out = pathlib.Path(args.output)
    if args.check:
        old = out.read_text(encoding="utf-8") if out.exists() else ""
        if old.replace('\r\n', '\n') != content:
            print(f"DRIFT: {out} differs from regeneration over {args.gguf}", file=sys.stderr)
            return 1
        print(f"no drift: {out} matches regeneration ({len(vectors)} vectors)")
        return 0
    out.write_bytes(content.encode("utf-8"))
    print(f"wrote {out} ({len(vectors)} vectors, vocab={len(enc.id_of)}, merges={len(enc.rank)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
