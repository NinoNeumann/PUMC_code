"""Cross-platform SHA-256 and atomic completion verification (standard library)."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def verify_complete(directory, recursive=True):
    directory=Path(directory).resolve()
    marker=directory/'COMPLETE.json'
    record=json.loads(marker.read_text(encoding='utf-8'))
    files=record.get('files')
    if not isinstance(files,dict) or not files:
        raise ValueError(f'Completion marker has no signed files: {marker}')
    for relative,digest in files.items():
        path=(directory/relative).resolve()
        if not path.is_relative_to(directory):
            raise ValueError(f'Completion path escapes its folder: {relative}')
        if not path.is_file() or sha256(path)!=digest:
            raise ValueError(f'Missing or changed completed file: {path}')
        if recursive and path.name=='COMPLETE.json':
            verify_complete(path.parent)
    if 'manifest_sha256' in record and sha256(directory/'manifest.json')!=record['manifest_sha256']:
        raise ValueError(f'Manifest hash mismatch: {directory}')
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input_csv',nargs='?');p.add_argument('output_csv',nargs='?')
    p.add_argument('--root',default='.')
    p.add_argument('--verify-complete');p.add_argument('--verify-sha');p.add_argument('--expected')
    a=p.parse_args()
    if a.verify_complete:
        verify_complete(a.verify_complete);print('COMPLETE_SHA256_VERIFIED');return
    if a.verify_sha:
        if not a.expected or sha256(a.verify_sha)!=a.expected:
            raise ValueError('File SHA-256 mismatch')
        print('FILE_SHA256_VERIFIED');return
    if not a.input_csv or not a.output_csv:p.error('Provide input.csv output.csv --root ROOT')
    root=Path(a.root).resolve()
    with Path(a.input_csv).open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    values=[]
    for row in rows:
        path=Path(row['path'])
        path=path if path.is_absolute() else root/path
        values.append(dict(path=row['path'],bytes=path.stat().st_size,sha256=sha256(path)))
    with Path(a.output_csv).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['path','bytes','sha256'])
        writer.writeheader();writer.writerows(values)


if __name__=='__main__':
    try:main()
    except Exception as error:
        print(f'HASH_VERIFICATION_FAILED: {error}',file=sys.stderr);sys.exit(1)
