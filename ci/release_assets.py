#!/usr/bin/env python3
"""CI helper for the windows-x64 workflow.

Versions come from the repo, never from the workflow YAML:
  VERSION                            release version; a release tag must be v<VERSION>
  vcpkg-ports/ggml/vcpkg.json        ggml version
  vcpkg-ports/whisper-cpp/vcpkg.json whisper.cpp version
  ci/build-matrix.json               CUDA toolkits, whisper source variant, Vulkan SDK

Subcommands:
  matrix [--tag REF]    write the build matrix and versions to $GITHUB_OUTPUT (or stdout)
  verify DIR            check DIR holds exactly the expected archives with the expected
                        layout, and write DIR/SHA256SUMS
  notes DIR TAG OUT     write release notes (versions, sha256, external_versions.json entries)
"""
import argparse
import fnmatch
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = os.environ.get("GITHUB_REPOSITORY", "MinLL/ggml-whisper")


def load_json(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def versions():
    matrix = load_json(ROOT / "ci" / "build-matrix.json")
    ggml = load_json(ROOT / "vcpkg-ports" / "ggml" / "vcpkg.json")
    whisper = load_json(ROOT / "vcpkg-ports" / "whisper-cpp" / "vcpkg.json")
    cuda = []
    for full in matrix["cuda"]:
        major, minor = full.split(".")[:2]
        cuda.append({"full": full, "short": f"{major}.{minor}"})
    shorts = [c["short"] for c in cuda]
    if matrix["whisper_from_cuda"] not in shorts:
        sys.exit(f"whisper_from_cuda {matrix['whisper_from_cuda']} is not one of {shorts}")
    return {
        "release": (ROOT / "VERSION").read_text(encoding="utf-8").strip(),
        "ggml": ggml.get("version") or ggml.get("version-semver"),
        "whisper": whisper.get("version-semver") or whisper.get("version"),
        "cuda": cuda,
        "whisper_from_cuda": matrix["whisper_from_cuda"],
        "vulkan_sdk": matrix["vulkan_sdk"],
    }


GGML_LAYOUT = {
    "bin": ["ggml.dll", "ggml-base.dll", "ggml-cpu.dll", "ggml-cuda.dll", "ggml-vulkan.dll", "vulkan-1.dll"],
    "include": ["ggml.h", "ggml-alloc.h", "ggml-backend.h", "ggml-cpu.h", "ggml-cuda.h", "ggml-vulkan.h"],
    "lib": ["ggml.lib", "ggml-base.lib"],
}
WHISPER_LAYOUT = {
    "bin": ["whisper.dll", "parakeet.dll"],
    "include": ["whisper.h", "parakeet.h"],
    "lib": ["whisper.lib", "parakeet.lib"],
}
CUDART_LAYOUT = {
    "bin": ["cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll", "cufft64_*.dll"],
}


def expected_archives(v):
    out = {}
    for c in v["cuda"]:
        out[f"ggml-windows-x64-{v['ggml']}-cuda{c['short']}"] = GGML_LAYOUT
        out[f"cudart-windows-x64-v{c['short']}"] = CUDART_LAYOUT
    out[f"whisper-windows-x64-{v['whisper']}"] = WHISPER_LAYOUT
    return out


def cmd_matrix(args):
    v = versions()
    if args.tag and args.tag.startswith("refs/tags/"):
        tag = args.tag[len("refs/tags/"):]
        if tag != f"v{v['release']}":
            sys.exit(f"Tag {tag} does not match VERSION ({v['release']}); bump VERSION and tag v<VERSION>.")
    include = [
        {"cuda": c["full"], "cuda_short": c["short"], "whisper": c["short"] == v["whisper_from_cuda"]}
        for c in v["cuda"]
    ]
    outputs = {
        "matrix": json.dumps({"include": include}),
        "release_version": v["release"],
        "ggml_version": v["ggml"],
        "whisper_version": v["whisper"],
        "vulkan_sdk": v["vulkan_sdk"],
    }
    target = os.environ.get("GITHUB_OUTPUT")
    with (open(target, "a", encoding="utf-8") if target else sys.stdout) as f:
        for k, val in outputs.items():
            f.write(f"{k}={val}\n")
    for k, val in outputs.items():
        print(f"{k}: {val}", file=sys.stderr)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_verify(args):
    d = Path(args.dir)
    expected = expected_archives(versions())
    found = {p.stem: p for p in d.glob("*.zip")}
    errors = []
    for name in sorted(set(expected) - set(found)):
        errors.append(f"missing archive {name}.zip")
    for name in sorted(set(found) - set(expected)):
        errors.append(f"unexpected archive {name}.zip")

    for name in sorted(set(expected) & set(found)):
        with zipfile.ZipFile(found[name]) as z:
            files = [n for n in z.namelist() if not n.endswith("/")]
        tops = {n.split("/", 1)[0] for n in files}
        if tops != {name}:
            errors.append(f"{name}.zip: top-level entries {sorted(tops)}, want only {name}/")
        rel = {n.split("/", 1)[1] for n in files if "/" in n}
        matched = set()
        for sub, patterns in expected[name].items():
            for pat in patterns:
                hits = fnmatch.filter(rel, f"{sub}/{pat}")
                if not hits:
                    errors.append(f"{name}.zip: missing {sub}/{pat}")
                matched.update(hits)
        for extra in sorted(rel - matched):
            print(f"note: {name}.zip has extra file {extra}")
        print(f"ok: {name}.zip ({len(files)} files)")

    if errors:
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    lines = [f"{sha256(found[n])}  {n}.zip" for n in sorted(found)]
    (d / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def cmd_notes(args):
    v = versions()
    d = Path(args.dir)
    sums = {}
    for line in (d / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1)
        sums[name.strip()] = digest
    base = f"https://github.com/{REPO}/releases/download/{args.tag}"

    entries = {}
    for c in v["cuda"]:
        folder = f"ggml-windows-x64-{v['ggml']}-cuda{c['short']}"
        entries[f"ggml-cuda{c['short']}"] = {
            "version": v["ggml"], "folder": folder,
            "url": f"{base}/{folder}.zip", "sha256": sums[f"{folder}.zip"],
        }
    folder = f"whisper-windows-x64-{v['whisper']}"
    entries["whisper"] = {
        "version": v["whisper"], "folder": folder,
        "url": f"{base}/{folder}.zip", "sha256": sums[f"{folder}.zip"],
    }

    cuda_list = ", ".join(c["full"] for c in v["cuda"])
    lines = [
        f"ggml {v['ggml']} and whisper.cpp {v['whisper']} for Windows x64, built by GitHub Actions.",
        "",
        f"- ggml: CPU, Vulkan (SDK {v['vulkan_sdk']}) and CUDA backends, `GGML_BACKEND_DL`; one archive per CUDA toolkit ({cuda_list})",
        f"- whisper.cpp / parakeet: linked against the separate ggml (built in the CUDA {v['whisper_from_cuda']} job)",
        "- cudart: the CUDA runtime, cuBLAS and cuFFT DLLs for each toolkit",
        "",
        "### SHA-256",
        "",
        "```",
        *[f"{sums[n]}  {n}" for n in sorted(sums)],
        "```",
        "",
        "### external_versions.json entries",
        "",
        "```json",
        json.dumps(entries, indent=2)[1:-1].strip("\n"),
        "```",
        "",
    ]
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("matrix")
    m.add_argument("--tag", default="")
    m.set_defaults(func=cmd_matrix)
    vf = sub.add_parser("verify")
    vf.add_argument("dir")
    vf.set_defaults(func=cmd_verify)
    n = sub.add_parser("notes")
    n.add_argument("dir")
    n.add_argument("tag")
    n.add_argument("out")
    n.set_defaults(func=cmd_notes)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
