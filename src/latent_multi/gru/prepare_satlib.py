"""Convert a user-obtained UF20-91 archive. No solutions or raw data are bundled."""
import argparse
import json
import tarfile
from pathlib import Path, PurePosixPath

from .io import content_crc32, validate_rows, write_json


def parse_dimacs(text):
    clauses, pending, header = [], [], None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("c"):
            continue
        if line.startswith("%"):
            break
        if line.startswith("p"):
            if header is not None or line.split()[:2] != ["p", "cnf"] or len(line.split()) != 4:
                raise ValueError("Malformed DIMACS header")
            header = tuple(map(int, line.split()[2:]))
            continue
        if header is None:
            raise ValueError("Missing DIMACS header")
        for token in line.split():
            value = int(token)
            if value:
                pending.append(value)
            elif pending:
                clauses.append(pending)
                pending = []
            else:
                raise ValueError("Empty or repeated clause terminator")
    if header != (20, 91) or pending or len(clauses) != 91:
        raise ValueError("Expected a complete UF20-91 formula")
    validate_rows([{"id": 0, "n": 20, "clauses": clauses}], "sat")
    return clauses


def convert(archive):
    entries = {}
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or member.issym() or member.islnk():
                raise ValueError("Unsafe archive entry")
            if not member.isfile() or name.suffix != ".cnf":
                continue
            if name.name in entries or member.size > 1024 * 1024:
                raise ValueError("Duplicate or oversized formula")
            entries[name.name] = parse_dimacs(handle.extractfile(member).read().decode("ascii"))
    if len(entries) != 1000:
        raise ValueError(f"Expected the full 1000 instances, found {len(entries)}")
    # Source filenames are preserved. These IDs are conversion-local, not assumed
    # to be the original project's integer IDs without an explicit binding.
    return [{"id": i, "n": 20, "source_filename": name, "clauses": entries[name]}
            for i, name in enumerate(sorted(entries))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = convert(args.archive)
    write_json(args.output, rows)
    print(json.dumps({"rows": len(rows), "archive_crc32": content_crc32(args.archive),
                      "output_crc32": content_crc32(args.output), "solutions_enumerated": 0,
                      "identity": "sorted source filename; project-ID equivalence requires separate binding"}))


if __name__ == "__main__":
    main()
