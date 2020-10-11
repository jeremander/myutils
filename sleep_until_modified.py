#!/usr/bin/env python3

import getopt
import os
import os.path
import sys
import time


available_parameters = [
    ("h", "help", "Print help"),
    ("i:", "interval=", "Defines the polling interval, in seconds (default=1.0)"),
]

class ProgramOptions():
    """Holds the program options, after they are parsed by parse_options()"""
    def __init__(self):
        self.poll_interval = 1
        self.args = []

def print_help():
    scriptname = os.path.basename(sys.argv[0])
    print("Usage: %s [options] filename" % scriptname)
    print("Sleeps until 'filename' has been modified.\nOptions:")
    long_length = 2 + max(len(lng) for x, lng, y in available_parameters)
    for short, lng, desc in available_parameters:
        if short and lng:
            comma = ", "
        else:
            comma = "  "
        if short == "":
            short = "  "
        else:
            short = "-" + short[0]
        if lng:
            lng = "--" + lng
        print("  {0}{1}{2:{3}}  {4}".format(short, comma, lng, long_length, desc))
    print("\nCurrently, it is implemented using polling.\nSample usage command-line:")
    print(" while sleep_until_modified.py myfile.tex || sleep 1; do make ; done")

def parse_options(argv, opt):
    """argv should be sys.argv[1:]
    opt should be an instance of ProgramOptions()"""
    try:
        opts, args = getopt.getopt(argv, "".join(short for (short, x, y) in available_parameters), [lng for (x, lng, y) in available_parameters])
    except getopt.GetoptError as e:
        print(str(e))
        print("Use --help for usage instructions.")
        sys.exit(2)
    for (o, v) in opts:
        if o in ("-h", "--help"):
            print_help()
            sys.exit(0)
        elif o in ("-i", "--interval"):
            opt.poll_interval = float(v)
        else:
            print("Invalid parameter: {0}".format(o))
            print("Use --help for usage instructions.")
            sys.exit(2)
    opt.args = args
    if len(args) == 0:
        print("Missing filename")
        print("Use --help for usage instructions.")
        sys.exit(2)
    if len(args) > 1:
        print("Currently, this script monitors only one file, but {0} files were given. Aborting.".format(len(args)))
        sys.exit(2)

def main():
    opt = ProgramOptions()
    parse_options(sys.argv[1:], opt)
    filename = opt.args[0]
    prev_time = os.stat(filename).st_mtime
    try:
        while True:
            time.sleep(opt.poll_interval)
            new_time = os.stat(filename).st_mtime
            if new_time != prev_time:
                break
    except KeyboardInterrupt:
        sys.exit(2)

if __name__ == "__main__":
    main()