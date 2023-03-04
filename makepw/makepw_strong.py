#!/usr/bin/env python3

"""
Generates a strong password using a cryptographically secure random source.\n
The password is a sequence of words sampled randomly from a dictionary.\n
User provides a length (in characters) and an alphabet (subset of 'luno'), optionally.\n
"""

from math import log2
import secrets
import string
import sys


ALPHABETS = {
    'l' : string.ascii_lowercase,
    'u' : string.ascii_uppercase,
    'n' : string.digits,
    'o' : string.punctuation,
    'c' : 'BCDFGHJKLMNPQRSTVWXZbcdfghjklmnpqrstvwxz',
    'v' : 'AEIOUYaeiouy',
    't' : string.ascii_lowercase + string.ascii_uppercase,
    'a' : string.ascii_lowercase + string.ascii_uppercase + string.digits,
    'p' : string.ascii_lowercase + string.ascii_uppercase + string.digits + string.punctuation,
    'f' : '!#$%12345@ABCDEFGQRSTVWXZ`abcdefgqrstvwxz~',
    'r' : '"&\'()*+,-./06789:;<=>?HIJKLMNOPUY[\\]^_hijklmnopuy{|}',
    'F' : 'ABCDEFGQRSTVWXZabcdefgqrstvwxz',
    'R' : 'HIJKLMNOPUYhijklmnopuy'
}


def make_strong_password(seq: str):
    """Makes a password by randomly sampling from a sequence of alphabets.
    Each letter in seq refers to an alphabet."""
    alphabets = [ALPHABETS[c] for c in seq]
    pw = ''.join(secrets.choice(alpha) for alpha in alphabets)
    entropy = sum([log2(len(alpha)) for alpha in alphabets])
    return (pw, entropy)

if __name__ == "__main__":

    desc = """
        USAGE:
            makepw_strong.py [LENGTH] [CLASS]
            makepw_strong.py [SEQUENCE]
        Generates a strong password using a cryptographically secure random source.\n
        The password is a sequence of random characters sampled from a sequence of character classes.\n
        User provides either a length (in characters) and a class, OR a sequence of classes.\n
        The valid classes are:\n
            l: lowercase
            u: uppercase
            n: numbers
            o: others (punctuation)
            c: consonants
            v: vowels
            t: letters
            a: alphanumeric (letters & numbers)
            f: left side of the keyboard
            r: right side of the keyboard
            F: left side of the keyboard (letters only)
            R: right side of the keyboard (letters only)
        """

    def show_usage():
        print(desc)

    try:
        arg1 = sys.argv[1]
        if arg1.isdigit():  # a length
            alpha = sys.argv[2] if (len(sys.argv) >= 3) else 'p'
            assert (len(alpha) == 1)
            seq = alpha * int(arg1)
        else:  # a sequence
            seq = arg1
        (pw, entropy) = make_strong_password(seq)
        print("\nGenerated %d-long password with %.3f bits of entropy:\n" % (len(pw), entropy))
        print(pw + '\n')
    except Exception:
        show_usage()
