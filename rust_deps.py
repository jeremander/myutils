#!/usr/bin/env python3
"""Generates a directed graph of internal dependencies for a Rust crate."""

from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from typing import Iterator, Self

import networkx as nx
import parsec
import pydot


def fix_name(s: str) -> str:
    return s.replace('::', '_')

def digraph_to_dag(dg: nx.DiGraph) -> nx.DiGraph:
    dg = deepcopy(dg)
    while True:
        try:
            cycle = nx.find_cycle(dg)
            dg.remove_edge(*cycle[-1])
        except nx.exception.NetworkXNoCycle:
            break
    return dg

def crate_parser() -> parsec.Parser:
    p_name = parsec.regex(r'\w+')
    p_identifier = parsec.sepBy1(p_name, parsec.string('::')).parsecmap('::'.join)
    p_comma = parsec.regex(r',\s*')
    p_identifiers = parsec.regex(r'{\s*') >> parsec.sepEndBy1(p_identifier, p_comma) << parsec.regex(r'\s*}')
    p_import = p_identifier + parsec.optional(parsec.string('::') >> p_identifiers)  # type: ignore
    p_imports = parsec.regex(r'{\s*') >> parsec.sepEndBy1(p_import, p_comma) << parsec.regex(r'\s*}')
    p_import_or_imports = p_imports ^ p_import.parsecmap(lambda x : None if (x is None) else [x])
    # p_prefix = parsec.regex(r'\s*') >> (parsec.regex(r'use\s+crate::') ^ parsec.regex(r'pub\s+mod\s+'))
    p_prefix = parsec.regex(r'\s*use\s+crate::')
    p_crate = p_prefix >> p_import_or_imports << parsec.optional(parsec.string(';'))  # type: ignore
    p_crate_search = parsec.many(p_crate ^ parsec.any())
    return p_crate_search.parsecmap(lambda results : [res for res in results if isinstance(res, list)])

class DepGraph(nx.DiGraph):
    @classmethod
    def from_crate_root(cls, crate_root: str) -> Self:
        parser = crate_parser()
        dg = cls()
        paths = list(Path(crate_root).glob('**/*.rs'))
        crate_name_by_path = {}
        for path in paths:
            crate1_name = str(path).removeprefix(crate_root.rstrip('/') + '/').removesuffix('.rs').replace('/', '::')
            crate_name_by_path[path] = crate1_name
            dg.add_node(crate1_name)
        for path in paths:
            crate1_name = crate_name_by_path[path]
            with open(path) as f:
                s = f.read()
                # if 'rfc3161' in crate1_name:
                #     breakpoint()
                results = parser.parse(s)
                for result in results:
                    for (crate2, identifiers) in result:
                        crate2_segs = crate2.split('::')
                        if (identifiers is None):
                            it = [crate2_segs]
                        else:
                            it = [crate2_segs + identifier.split('::') for identifier in identifiers]
                        for crate2_segs in it:
                            # module::self refers to module/mod.rs?
                            crate2_segs = ['mod' if (seg == 'self') else seg for seg in crate2_segs]
                            for n in range(len(crate2_segs), 0, -1):
                                # find longest prefix of matching filename crate
                                segs = crate2_segs[:n]
                                name = '::'.join(segs)
                                if (name in dg):
                                    dg.add_edge(crate1_name, name)
                                    break
        return dg
    def to_pydot(self) -> pydot.Dot:
        dg = nx.DiGraph()
        dg.add_nodes_from(map(fix_name, self.nodes))
        dg.add_edges_from((fix_name(u), fix_name(v)) for (u, v) in self.edges)
        return nx.drawing.nx_pydot.to_pydot(dg)
    def reverse_arrows(self) -> nx.DiGraph:
        dg = nx.DiGraph()
        dg.add_nodes_from(self.nodes)
        for (u, v) in self.edges:
            dg.add_edge(v, u)
        return dg
    def dependency_order(self) -> Iterator[tuple[int, int, str]]:
        dag = digraph_to_dag(self.reverse_arrows())
        for gen in nx.topological_generations(dag):
            # prioritize nodes that depend on less, are depended on by more, then tiebreak alphabetically
            tups = [(self.out_degree(node), -self.in_degree(node), node) for node in gen]
            yield from sorted(tups)
    def write_file(self, path: str) -> None:
        p = Path(path)
        suff = p.suffix.lstrip('.')
        method = f'write_{suff}'
        dot = self.to_pydot()
        if hasattr(dot, method):
            getattr(dot, method)(path)
        else:
            raise ValueError(f'invalid file extension {suff!r}')

def main() -> None:
    parser = ArgumentParser(description = __doc__)
    parser.add_argument('crate_root', help = 'root directory of crate')
    parser.add_argument('-o', '--output-file', help = 'output image file for dependency graph')
    args = parser.parse_args()
    dg = DepGraph.from_crate_root(args.crate_root)
    for (out_deg, neg_in_deg, name) in dg.dependency_order():
        print(f'{out_deg} (out) {-neg_in_deg} (in) {name}')
    if (args.output_file):
        print(f'Writing dependency graph to {args.output_file}')
        dg.write_file(args.output_file)


if __name__ == '__main__':
    main()
