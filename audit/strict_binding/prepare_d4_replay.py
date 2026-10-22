#!/usr/bin/env python3
"""Prepare only the exact pinned D4 replay inputs, checking bytes before use."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--destination",type=Path,required=True)
    parser.add_argument("--from-root",type=Path)
    args=parser.parse_args()
    ledger=json.loads((HERE/"source_ledger.json").read_text())
    for row in ledger["d4_replay_inputs"]:
        relative=Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid pinned path")
        if args.from_root:
            data=(args.from_root/relative).read_bytes()
        else:
            url=f"https://raw.githubusercontent.com/{row['repository']}/{row['revision']}/{row['path']}"
            with urllib.request.urlopen(url,timeout=45) as response:data=response.read()
        if hashlib.sha256(data).hexdigest()!=row["sha256"]:
            raise ValueError("D4 source hash mismatch: "+str(relative))
        destination=args.destination/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(data)
    print(f"PASS: {len(ledger['d4_replay_inputs'])} exact D4 replay/source blobs prepared")


if __name__=="__main__":main()
