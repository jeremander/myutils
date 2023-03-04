#!/usr/bin/env python3

"""
Generates an xkcd-style password using a cryptographically secure random source.
The password is a sequence of words sampled randomly from a dictionary.
User provides a length (in words) and a dictionary file (optional).
"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, RawDescriptionHelpFormatter
import json
import math
from pathlib import Path
import secrets


# words of length 1-8, all lowercase
DEFAULT_DICTFILE = Path(__file__).with_name('mydict.json')


def make_xkcd_password(n, dictfile = DEFAULT_DICTFILE):
    """Makes a password by randomly sampling n words from Webster's dictionary and concatenating them, using camel case."""
    with open(dictfile) as f:
        dictionary = json.load(f)
    words = sorted(dictionary.keys())
    pw_words = []
    pw = ''
    for _ in range(n):
        word = secrets.choice(words)
        pw_words.append(word)
        pw += word[0].upper() + word[1:]  # camel case
    entropy = n * math.log2(len(words))
    for word in pw_words:
        print(word + ':')
        meaning = dictionary[word]
        if isinstance(meaning, str):
            print('\t' + meaning)
        else:
            print(json.dumps(meaning, indent = 4))
        print("")
    return (pw, entropy)

def main():
    class Formatter(ArgumentDefaultsHelpFormatter, RawDescriptionHelpFormatter):
        pass
    p = ArgumentParser(description = __doc__, formatter_class = Formatter)
    p.add_argument('length', help = 'length of password in words', type = int)
    p.add_argument('-d', '--dictfile', default = DEFAULT_DICTFILE, help = 'dictionary file')
    args = p.parse_args()

    try:
        (pw, entropy) = make_xkcd_password(args.length, args.dictfile)
    except FileNotFoundError as e:
        p.error(e)

    print("\nGenerated %d-long password with %.3f bits of entropy:\n" % (len(pw), entropy))
    print(pw + '\n')


if __name__ == "__main__":

    main()

