#!/usr/bin/env python3
"""Downloads audio from YouTube and splits it according to a given file containing track names and start times."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from pytimeparse.timeparse import timeparse
import subprocess_tee
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple


Tracks = List[Tuple[str, float]]
Metadata = Dict[str, Any]

METADATA_FIELDS = ['artist', 'composer', 'album', 'genre', 'description']

def check_output(cmd: List[str], verbose: bool = False) -> str:
    print(' '.join(cmd))
    return subprocess_tee.run(cmd, tee = verbose).stdout

@dataclass
class YoutubeSplitter:
    track_list: str
    output_dir: Optional[str] = None
    proxy: Optional[str] = None
    trim: float = 1.0
    metadata: Optional[Metadata] = None
    def read_track_list(self) -> Tracks:
        pairs = []
        with open(self.track_list) as f:
            for line in f:
                if (line := line.strip()):
                    time_str, track_name = line.split(maxsplit = 1)
                    pairs.append((track_name, timeparse(time_str)))
        return pairs
    def download_track(self, url: str) -> str:
        print(f'Downloading {url} as MP3...')
        fmt = './%(title)s.%(ext)s'
        cmd = ['youtube-dl', '-x', '--audio-format', 'mp3', '-o', fmt]
        if self.proxy:
            cmd += ['--proxy', self.proxy]
        cmd.append(url)
        output = check_output(cmd)
        prefix = '[ffmpeg] Destination:'
        for line in output.splitlines():
            if line.startswith(prefix):
                return line.removeprefix(prefix).strip()
        raise ValueError('could not retrieve download destination path')
    def split_tracks(self, track_path: str, tracks: Tracks) -> None:
        output_dir = Path(self.output_dir) if self.output_dir else Path(track_path).parent
        output_dir.mkdir(exist_ok = True)
        ext = Path(track_path).suffix
        num_tracks = len(tracks)
        for (i, (track_name, start)) in enumerate(tqdm(tracks)):
            output_path = str(output_dir / (track_name + ext))
            if (i < num_tracks - 1):  # the end is the next track's start, minus the trim duration
                end: Optional[float] = tracks[i + 1][1] - self.trim
            else:
                end = None
            cmd = ['ffmpeg', '-y', '-ss', str(start), '-i', track_path, '-c', 'copy']
            if (end is not None):
                cmd += ['-t', str(end - start)]
            if self.metadata:
                if self.metadata.get('label_tracks', False):
                    cmd += ['-metadata', f'track={i + 1}/{num_tracks}']
                for field in METADATA_FIELDS:
                    if (field in self.metadata):
                        cmd += ['-metadata', field + '=' + self.metadata[field]]
            cmd.append(output_path)
            check_output(cmd, verbose = False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    input_group = parser.add_mutually_exclusive_group(required = True)
    input_group.add_argument('-u', '--url', help = 'URL of YouTube video to download')
    input_group.add_argument('-i', '--input-file', help = 'path to an MP3 file to split')
    parser.add_argument('tracks', help = 'file containing track list (each line contains start time, then track name)')
    parser.add_argument('-o', '--output-dir', help = 'output directory name')
    parser.add_argument('--proxy', help = 'proxy HOST:PORT to use')
    parser.add_argument('-t', '--trim', default = 1.0, type = float, help = 'trim this many seconds from the end of each track')
    metadata_gp = parser.add_argument_group(title = 'metadata arguments')
    metadata_gp.add_argument('--label-tracks', action = 'store_true', help = 'whether to label tracks in order')
    for field in METADATA_FIELDS:
        metadata_gp.add_argument('--' + field)
    args = parser.parse_args()

    metadata = {'label_tracks' : getattr(args, 'label_tracks', False)}
    for field in METADATA_FIELDS:
        val = getattr(args, field, None)
        if val:
            metadata[field] = val

    splitter = YoutubeSplitter(args.tracks, args.output_dir, proxy = args.proxy, trim = args.trim, metadata = metadata)

    if args.url:
        track_path = splitter.download_track(args.url)
    else:
        track_path = args.input_file
    tracks = splitter.read_track_list()
    splitter.split_tracks(track_path, tracks)
