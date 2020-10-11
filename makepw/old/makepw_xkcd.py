#!/usr/bin/env python3

"""
Generates an xkcd-style password using a cryptographically secure random source.\n
The password is a sequence of words sampled randomly from a dictionary.\n
User provides a length (in words) and a dictionary file (optional).\n
"""

import secrets
import math
import argparse
import io
from contextlib import redirect_stdout, redirect_stderr

DEFAULT_DICTFILE = '/Users/jeremander/Desktop/Programming/scripts/mydict.txt'

def make_xkcd_password(n, dictfile = DEFAULT_DICTFILE):
    """Makes a password by randomly sampling n words from Webster's dictionary and concatenating them, using camel case."""
    # Webster words of length 1-8, all lowercase
    with open(dictfile) as f:  
        words = [word.strip() for word in f]
    pw_words = []
    pw = ''
    for i in range(n):
        word = secrets.choice(words)
        pw_words.append(word)
        pw += word[0].upper() + word[1:]  # camel case
    entropy = n * math.log2(len(words))
    try:
        from PyDictionary import PyDictionary
        dictionary = PyDictionary()
        for word in pw_words:
            print(word + ':')
            f = io.StringIO()
            with redirect_stdout(f):
                with redirect_stderr(f):
                    meanings = dictionary.meaning(word)
            if (meanings is not None):
                for pos in meanings:
                    print('\t' + pos + ':')
                    for meaning in meanings[pos]:
                        print('\t\t' + meaning)
    except:
        print("Warning: could not load PyDictionary.")
    return (pw, entropy)


if __name__ == "__main__":

    p = argparse.ArgumentParser(description = __doc__)
    p.add_argument('length', help = 'length of password in words', type = int)
    p.add_argument('--dictfile', default = DEFAULT_DICTFILE, help = "dictionary file")
    args = p.parse_args()

    (pw, entropy) = make_xkcd_password(args.length, args.dictfile)

    print("\nGenerated %d-long password with %.3f bits of entropy:\n" % (len(pw), entropy))
    print(pw + '\n')
