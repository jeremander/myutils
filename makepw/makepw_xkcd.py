#!/usr/bin/env python3

"""
Generates an xkcd-style password using a cryptographically secure random source.\n
The password is a sequence of words sampled randomly from a dictionary.\n
User provides a length (in words) and a dictionary file (optional).\n
"""

import secrets
import math
import argparse
import json

# words of length 1-8, all lowercase
DEFAULT_DICTFILE = '/Users/jeremander/Programming/scripts/makepw/mydict.json'

def make_xkcd_password(n, dictfile = DEFAULT_DICTFILE):
    """Makes a password by randomly sampling n words from Webster's dictionary and concatenating them, using camel case."""
    dictionary = json.load(open(dictfile, 'r'))
    words = sorted(dictionary.keys())
    pw_words = []
    pw = ''
    for i in range(n):
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

if __name__ == "__main__":

    p = argparse.ArgumentParser(description = __doc__)
    p.add_argument('length', help = 'length of password in words', type = int)
    p.add_argument('--dictfile', default = DEFAULT_DICTFILE, help = "dictionary file")
    args = p.parse_args()

    (pw, entropy) = make_xkcd_password(args.length, args.dictfile)

    print("\nGenerated %d-long password with %.3f bits of entropy:\n" % (len(pw), entropy))
    print(pw + '\n')
