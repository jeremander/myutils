#!/usr/bin/env python3

import os
import subprocess


def mkdir(dirname, remote_url):
    """Makes a directory at the remote URL."""
    subprocess.check_call(['duck', '-y', '-c', os.path.join(remote_url, dirname)])

def delete(remote_url):
    """Deletes a folder or file at the remote URL."""
    subprocess.check_call(['duck', '-y', '-D', remote_url])

def copy(local_files, remote_url):
    """Copies files to a remote URL."""
    subprocess.check_call(['duck', '-y', '--upload', remote_url])

def synchronize(local_dirname, remote_url):
    """Synchronize two directories."""
    #