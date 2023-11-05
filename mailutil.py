#!/usr/bin/env python3

from argparse import ArgumentParser, Namespace
from contextlib import contextmanager
import csv
from datetime import datetime
from functools import reduce
import getpass
import logging
import sys
from typing import Iterator, Optional, Union

import imap_tools
from imap_tools import AND, OR, MailMessage
from imap_tools.mailbox import Criteria
from tqdm import tqdm


logging.basicConfig(level = logging.INFO)
LOGGER = logging.getLogger('emails')

FIELDS = {
    'Subject' : 'subject',
    'From' : 'from_',
    'To' : 'to',
    'Date' : 'date_str'
}

class MailBox(imap_tools.MailBox):
    def fetch(self, criteria: Criteria = 'ALL', charset: str = 'US-ASCII', limit: Optional[Union[int, slice]] = None, mark_seen = True, reverse = False, headers_only = False, chunk_size: int = 1) -> Iterator[MailMessage]:
        message_parts = "(BODY{}[{}] UID FLAGS RFC822.SIZE)".format('' if mark_seen else '.PEEK', 'HEADER' if headers_only else '')
        limit_range = slice(0, limit) if type(limit) is int else limit or slice(None)
        assert type(limit_range) is slice
        uids = tuple((reversed if reverse else iter)(self.uids(criteria, charset)))[limit_range]
        for i in range(0, len(uids), chunk_size):
            uid_chunk = uids[i : i + chunk_size]
            for fetch_item in self._fetch_in_bulk(uid_chunk, message_parts, reverse):
                yield self.email_message_class(fetch_item)

@contextmanager
def login(server: str, address: str, password: Optional[str] = None) -> Iterator[MailBox]:
    password = getpass.getpass() if (password is None) else password
    LOGGER.info(f'Logging into {address} on {server}...')
    with MailBox(server).login(address, password) as mailbox:
        yield mailbox

# DELETE

def make_parser_delete(parser: ArgumentParser) -> None:
    parser.add_argument('--from', nargs = '+', help = 'sender e-mail address(es), can also be substrings')
    parser.add_argument('--before', help = 'date before which to delete (YYYY-mm-dd format)')

def run_delete(args: Namespace) -> None:
    queries = []
    if getattr(args, 'from'):
        queries.append(OR(from_ = getattr(args, 'from')))
    if args.before:
        before = datetime.strptime(args.before, '%Y-%m-%d').date()
        queries.append(AND(date_lt = before))
    query = reduce(AND, queries)
    LOGGER.info(f'Delete query: {query}')
    with login(args.server, args.address) as mailbox:
        uids = [uid for msg in tqdm(mailbox.fetch(query)) if (uid := msg.uid)]
        LOGGER.info(f'Deleting {len(uids)} message(s)...')
        mailbox.delete(uids)

# EXPORT

def make_parser_export(parser: ArgumentParser) -> None:
    parser.add_argument('--chunk-size', type = int, default = 50, help = 'number of e-mail headers to download in each request')
    parser.add_argument('--start', type = int, default = 0, help = 'start index of first e-mail')
    parser.add_argument('--stop', type = int, help = 'start index of first e-mail')

def run_export(args: Namespace) -> None:
    with login(args.server, args.address) as mailbox:
        writer = csv.writer(sys.stdout, delimiter = '\t', quoting = csv.QUOTE_MINIMAL)
        header = list(FIELDS) + ['FromDomain']
        writer.writerow(header)
        ctr = 0
        from_idx = list(FIELDS.values()).index('from_')
        limit = slice(args.start, args.stop)
        for msg in tqdm(mailbox.fetch(headers_only = True, mark_seen = False, limit = limit, chunk_size = args.chunk_size)):
            ctr += 1
            row = []
            for field in FIELDS.values():
                val = getattr(msg, field)
                if (field == 'to'):
                    val = ','.join(val)
                row.append(val)
            from_ = row[from_idx]
            from_domain = from_.split('@')[1] if ('@' in from_) else ''
            row.append(from_domain)
            writer.writerow(row)
        LOGGER.info(f'Fetched {ctr} email(s)')

# MAIN

def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('-s', '--server', required = True, help = 'server name')
    parser.add_argument('-a', '--address', required = True, help = 'full e-mail address')
    subparsers = parser.add_subparsers(dest = 'subcommand')
    p_delete = subparsers.add_parser('delete', help = 'delete e-mails')
    make_parser_delete(p_delete)
    p_export = subparsers.add_parser('export', help = 'export e-mail metadata to TSV file')
    make_parser_export(p_export)
    args = parser.parse_args()
    func = globals()[f'run_{args.subcommand}']
    func(args)


if __name__ == '__main__':
    main()