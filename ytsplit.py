#!/usr/bin/env python3
"""Downloads audio from YouTube and splits it according to a given file containing track names and start times."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional

from pytimeparse.timeparse import timeparse
import subprocess_tee
from tqdm import tqdm


YDL = 'yt-dlp'

@dataclass
class Track:
    name: str
    album: Optional[str]
    start: float
    duration: Optional[float] = None

Tracks = List[Track]
Metadata = Dict[str, Any]

METADATA_FIELDS = ['artist', 'composer', 'album', 'genre', 'description']

def check_output(cmd: List[str], verbose: bool = False) -> str:
    print(' '.join(cmd))
    try:
        return subprocess_tee.run(cmd, tee = verbose, check = True).stdout
    except subprocess.CalledProcessError as e:
        print(e.stderr, file = sys.stderr)
        raise e

def fix_name(name: str) -> str:
    return name.replace('/', ' - ')  # slash conflicts with paths

@dataclass
class YoutubeSplitter:
    track_list: str
    output_dir: Optional[str] = None
    proxy: Optional[str] = None
    trim: float = 1.0
    metadata: Optional[Metadata] = None
    track_path: str = ''
    time_first: bool = False  # if True, the time comes before the name
    def read_track_list(self, headers_as_albums = True) -> Tracks:
        tracks = []
        with open(self.track_list) as f:
            album = None
            start = 0.0
            for line in f:
                if (line := line.strip()):
                    # print(line)
                    try:
                        toks = line.split()
                        if self.time_first:
                            time_str, track_toks = toks[0], toks[1:]
                        else:
                            track_toks, time_str = toks[:-1], toks[-1]
                        track_name = ' '.join(track_toks)
                        start = timeparse(time_str)
                        assert (start is not None), f'invalid time: {time_str}'
                        tracks.append(Track(track_name, album, start))
                    except (AssertionError, ValueError):
                        # assume the line is an album title
                        if headers_as_albums:
                            album = line
        for (i, track) in enumerate(tracks[:-1]):
            # the end is the next track's start, minus the trim duration
            end = tracks[i + 1].start - self.trim
            track.duration = end - track.start
        return tracks
    def download_track(self, url: str) -> str:
        print(f'Downloading {url} as MP3...')
        fmt = './%(title)s.%(ext)s'
        cmd = [YDL, '-x', '--audio-format', 'mp3', '-o', fmt]
        if self.proxy:
            cmd += ['--proxy', self.proxy]
        cmd.append(url)
        output = check_output(cmd)
        prefix = re.compile(r'\[(ffmpeg|ExtractAudio)\] Destination: (.*)')
        for line in output.splitlines():
            if (match := prefix.match(line)):
                return match.group(2).strip()
        raise ValueError('could not retrieve download destination path')
    def group_tracks_by_album(self, tracks: Tracks) -> Dict[Optional[str], List[Track]]:
        """Groups tracks by album."""
        grouped = defaultdict(list)
        for track in tracks:
            grouped[track.album].append(track)
        for group in grouped.values():
            names = [track.name for track in group]
            assert (len(names) == len(set(names))), 'duplicate track name'
        return grouped
    def _split_track_group(self, output_dir: Path, tracks: Tracks) -> None:
        output_dir.mkdir(exist_ok = True)
        ext = Path(self.track_path).suffix
        num_tracks = len(tracks)
        for (i, track) in enumerate(tqdm(tracks)):
            output_path = str(output_dir / (fix_name(track.name) + ext))
            cmd = ['ffmpeg', '-y', '-ss', str(track.start), '-i', self.track_path, '-c', 'copy']
            if (track.duration is not None):
                cmd += ['-t', str(int(track.duration))]
            if self.metadata:
                if self.metadata.get('label_tracks', False):
                    cmd += ['-metadata', f'track={i + 1}/{num_tracks}']
                for field in METADATA_FIELDS:
                    val = None
                    if (field == 'album') and track.album:
                        val = track.album
                    elif (field in self.metadata):
                        val = self.metadata[field]
                    if val:
                        cmd += ['-metadata', field + '=' + val]
            cmd.append(output_path)
            check_output(cmd, verbose = False)
    def split_tracks(self, tracks: Tracks) -> None:
        output_dir = Path(self.output_dir or Path.cwd())
        grouped = self.group_tracks_by_album(tracks)
        if (len(grouped) > 1):
            for (album, group) in grouped.items():
                # make a subdirectory for each album
                if (album is not None):
                    print('\n' + album)
                album_dir = output_dir if (album is None) else (output_dir / fix_name(album))
                self._split_track_group(album_dir, group)
        else:
            self._split_track_group(output_dir, tracks)

def main() -> None:
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    input_group = parser.add_mutually_exclusive_group(required = True)
    input_group.add_argument('-u', '--url', help = 'URL of YouTube video to download')
    input_group.add_argument('-i', '--input-file', help = 'path to an MP3 file to split')
    parser.add_argument('tracks', help = 'file containing track list (each line contains start time, then track name)')
    parser.add_argument('-o', '--output-dir', help = 'output directory name')
    parser.add_argument('--proxy', help = 'proxy HOST:PORT to use')
    parser.add_argument('-t', '--trim', default = 0.5, type = float, help = 'trim this many seconds from the end of each track')
    parser.add_argument('--time-first', action = 'store_true', help = 'time comes before track name in track list')
    metadata_gp = parser.add_argument_group(title = 'metadata arguments')
    metadata_gp.add_argument('--label-tracks', action = 'store_true', help = 'whether to label tracks with numbers in order')
    metadata_gp.add_argument('--headers-as-albums', action = 'store_true', help = 'whether to use section headers as album titles')
    for field in METADATA_FIELDS:
        metadata_gp.add_argument('--' + field)
    args = parser.parse_args()

    metadata = {'label_tracks' : getattr(args, 'label_tracks', False)}
    for field in METADATA_FIELDS:
        val = getattr(args, field, None)
        if val:
            metadata[field] = val

    splitter = YoutubeSplitter(args.tracks, args.output_dir, proxy = args.proxy, trim = args.trim, metadata = metadata, time_first = args.time_first)

    if args.url:
        splitter.track_path = splitter.download_track(args.url)
    else:
        splitter.track_path = args.input_file

    output_dir = args.output_dir if args.output_dir else Path(splitter.track_path).parent
    splitter.output_dir = output_dir

    tracks = splitter.read_track_list(headers_as_albums = args.headers_as_albums)
    splitter.split_tracks(tracks)


if __name__ == '__main__':

    main()
