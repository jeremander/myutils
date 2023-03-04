#!/usr/bin/env python3
"""Identifies groups of files that are likely to be "versions" of the same file, based on the filenames and extensions."""

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from humanize import naturalsize


TIME_FMT = '%Y-%m-%d %H:%M:%S'

def identify_groups(path: Path, min_size: int, prefix_length: int, verbose: bool = False) -> Dict[str, Any]:
    subgroups = {}
    groups = defaultdict(list)
    paths = path.glob('*')
    for p in paths:
        if p.is_file():
            # group files by (name prefix, extension)
            key = (p.stem[:prefix_length], p.suffix[1:])
            groups[key].append(p)
        else:  # directory (issue recursive call)
            subgroups[str(p)] = identify_groups(p, min_size, prefix_length, verbose)
    # filter out groups with one element, and any files less than min_size bytes
    new_groups: Dict[Tuple[str, str], List[Tuple[str, int, str]]] = {}
    for (key, group) in groups.items():
        if (len(group) > 1):
            new_group = []
            for p in group:
                stat = p.stat()
                size = stat.st_size
                if (size >= (min_size << 20)):
                    ctime = datetime.fromtimestamp(stat.st_ctime).strftime(TIME_FMT)
                    new_group.append((str(p), size, ctime))
            if (len(new_group) > 1):
                new_groups[key] = sorted(new_group)
    if verbose and new_groups:
        print(str(path))
        for new_group in new_groups.values():
            print('-' * 20)
            for (filename, size, _) in new_group:
                # print(f'\t{naturalsize(size)} | {ctime} | {filename}')
                print(f'\t{naturalsize(size)} | {filename}')
        print('-' * 20 + '\n')
    return {'groups' : new_groups, 'subgroups' : subgroups}


def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('input_dir', help = 'input directory')
    parser.add_argument('-s', '--min-size', type = int, default = 1, help = 'min size in MB of files to consider')
    parser.add_argument('-p', '--prefix-length', type = int, default = 8, help = 'length of prefix to match')
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    assert input_dir.exists() and input_dir.is_dir(), 'Must provide an existing directory.'
    identify_groups(input_dir, args.min_size, args.prefix_length, verbose = True)


if __name__ == '__main__':
    main()
