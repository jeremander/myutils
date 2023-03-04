from argparse import ArgumentParser
from pathlib import Path

import networkx as nx
import parsec


def fix_name(s):
    return s.replace('::', '_')

def crate_parser():
    p_identifier = parsec.regex(r'\w+(::\w+)*')
    p_name = parsec.regex(r'\w+')
    p_comma = parsec.regex(r',\s*')
    p_names = parsec.string('{') >> parsec.sepEndBy1(p_name, p_comma) << parsec.string('}')
    p_import = p_identifier + parsec.optional(parsec.string('::') >> p_names)
    p_imports = parsec.sepEndBy1(p_import, p_comma)
    p_crate = parsec.regex(r'\s*use\s+crate::{\s*') >> p_imports << parsec.regex(r'\s*};')
    p_crate_search = parsec.many(p_crate ^ parsec.any())
    return p_crate_search.parsecmap(lambda results : [res for res in results if isinstance(res, list)])

def make_depgraph(crate_root):
    parser = crate_parser()
    dg = nx.DiGraph()
    paths = list(Path(crate_root).glob('**/*.rs'))
    crate_name_by_path = {}
    for path in paths:
        file_crate = str(path).removeprefix(crate_root.rstrip('/') + '/').removesuffix('.rs').replace('/', '::')
        crate1_segs = file_crate.split('::')
        crate1_name = '_'.join(crate1_segs)
        crate_name_by_path[path] = crate1_name
        dg.add_node(crate1_name)
    for path in paths:
        crate1_name = crate_name_by_path[path]
        with open(path) as f:
            s = f.read()
            results = parser.parse(s)
            for result in results:
                for (crate2, names) in result:
                    crate2_segs = crate2.split('::')
                    if (names is None):
                        it = [crate2_segs]
                    else:
                        it = [crate2_segs + [name] for name in names]
                    for crate2_segs in it:
                        # module::self refers to module/mod.rs?
                        crate2_segs = ['mod' if (seg == 'self') else seg for seg in crate2_segs]
                        for n in range(len(crate2_segs), 0, -1):
                            # find longest prefix of matching filename crate
                            segs = crate2_segs[:n]
                            name = '_'.join(segs)
                            if (name in dg):
                                dg.add_edge(crate1_name, name)
                                break
    return dg


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('crate_root', help = 'root directory of crate')
    args = parser.parse_args()

    dg = make_depgraph(args.crate_root)

    d = nx.drawing.nx_pydot.to_pydot(dg)
    print(d.to_string())
