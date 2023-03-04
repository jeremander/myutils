#!/usr/bin/env python3

"""Does a grid search parameter sweep, plotting the results.
Takes a config file whose fields are:
data (path to a .csv file containing the data)
[distinguisher] (specifies name of variable for the distinguisher)
y_var (specifies name of variable for the y values)
label_suppress_vars (names of variables for which to show values only, not variable name)
outer_vars (names of variables to create separate plots for)"""

import argparse
from configparser import ConfigParser
import importlib
import inspect
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# from pylib
from rstyle import *


distinguishers = ['x', 'xfacet', 'yfacet', 'linestyle', 'color']
max_numvals_by_distinguisher = {'x' : np.inf, 'xfacet' : 5, 'yfacet' : 5, 'linestyle' : 4, 'color' : 16}
linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
cmap = plt.cm.gist_ncar

def dict_union(*dicts):
    return dict(itertools.chain(*map(lambda dct: list(dct.items()), dicts)))

def legend_str(var, param, suppress_var):
    return str(param) if suppress_var else ('{}={}'.format(var, param))

def try_eval(expression):
    try:
        return eval(expression)
    except NameError:
        return expression

def main():
    p = argparse.ArgumentParser()
    p.add_argument('config', type = str, help = 'config filename')
    p.add_argument('--section', '-s', type = str, default = 'main', help = 'section header of config file')
    p.add_argument('--prefix', '-p', type = str, default = None, help = 'prefix of path to write files')
    p.add_argument('--width', type = int, default = 12, help = 'width of plot (inches)')
    p.add_argument('--height', type = int, default = 8, help = 'height of plot (inches)')
    args = p.parse_args()

    # get various things from the config file
    cfg = ConfigParser()
    cfg.read(args.config)
    params = cfg._sections[args.section]
    df = pd.read_csv(params['data'])
    assert ('y_var' in params), "Parameter file must contain variable 'y_var'"
    yvar = params['y_var']
    assert (yvar in df.columns), "variable '{}' must be a data column".format(yvar)

    # get the path prefix
    if (args.prefix is None):
        outfile_prefix = os.path.splitext(params['data'])[0]
    else:
        outfile_prefix = args.prefix

    # associate variables with distinguishers
    vars_by_distinguisher = dict()
    for dist in distinguishers:
        var_key = dist + '_var'
        if (dist == 'x'):
            assert (var_key in params), "Parameter file must contain variable '{}'".format(var_key)
            xvar = params[var_key]
        if var_key in params:
            var = params[var_key]
            vars_by_distinguisher[dist] = var
            assert (var in df.columns), "variable '{}' must be a data column".format(var)

    # acquire any constant variables
    constant_dict = dict()
    for arg in params:
        if (arg not in vars_by_distinguisher.values()) and (arg in df.columns):
            val = try_eval(params[arg])
            df = df[df[arg] == val]  # filter the data by the constraint
            constant_dict[arg] = val

    # construct the function as a lookup table
    input_cols = sorted(vars_by_distinguisher.values())
    lookup = dict()
    for (inputs, output) in zip(zip(*(df[col] for col in input_cols)), df[yvar]):
        if (inputs in lookup):
            raise ValueError("Can't have multiple outputs for the same set of inputs.")
        lookup[inputs] = output
    def func(**kwargs):
        try:
            return lookup[tuple(val for (_, val) in sorted(kwargs.items()))]
        except KeyError:
            return None

    # get the iterables of each input variable
    kwargs = dict()
    for var in input_cols:
        if (var in params):  # get values from the config file
            kwargs[var] = try_eval(params[var.lower()])
        else:  # consider all possible values in the table
            kwargs[var] = sorted(set(df[var]))
    xvals = sorted(set(df[xvar]))

    # organize the different types of variables
    vars_to_suppress = try_eval(params.get('label_suppress_vars', '[]'))
    inner_vars = set(vars_by_distinguisher.values())  # vars for plotting
    outer_vars = set(try_eval(params.get('outer_vars', '[]')))  # vars for iterating separate plots
    iter_vars = inner_vars.union(outer_vars)  # vars to be iterated over

    outer_dict_iter = ({var : val for (var, val) in zip(outer_vars, param_tuple)} for param_tuple in itertools.product(*(kwargs[var1] for var1 in outer_vars)))
    # loop over the outer variables
    for outer_dict in outer_dict_iter:
        #print(outer_dict)
        numvals_by_distinguisher = {dist : len(kwargs[var]) for (dist, var) in vars_by_distinguisher.items()}
        for dist in vars_by_distinguisher:
            if (numvals_by_distinguisher[dist] > max_numvals_by_distinguisher[dist]):
                raise ValueError("Maximum number of values for {} is {}.".format(dist, max_numvals_by_distinguisher[dist]))
        for dist in distinguishers:
            if (dist not in numvals_by_distinguisher):
                numvals_by_distinguisher[dist] = 1

        colors = {j : cmap(int((j + 1) * cmap.N / (numvals_by_distinguisher['color'] + 1.0))) for j in range(numvals_by_distinguisher['color'])} if ('color' in vars_by_distinguisher) else {0 : 'blue'}

        if ('title' in params):
            plot_title = params['title']
        else:  # create the title
            def getval(var):
                if (var in outer_vars):
                    return outer_dict[var]
                else:  # inner_vars
                    return None
            plot_title = '{}({})'.format(yvar, ', '.join(var if (getval(var) is None) else '{} = {}'.format(var, getval(var)) for var in input_cols))

        # prepare the plot
        plt.close('all')
        fig, axis_grid = plt.subplots(numvals_by_distinguisher['yfacet'], numvals_by_distinguisher['xfacet'], sharex = 'col', sharey = 'row', figsize = (args.width, args.height), facecolor = 'white')
        axis_grid = np.array(axis_grid).reshape((numvals_by_distinguisher['yfacet'], numvals_by_distinguisher['xfacet']))
        plots_for_legend = []
        keys_for_legend = []

        # loop over the inner variables
        inner_dict = dict()
        for x in range(numvals_by_distinguisher['xfacet']):
            if ('xfacet' in vars_by_distinguisher):
                inner_dict[vars_by_distinguisher['xfacet']] = kwargs[vars_by_distinguisher['xfacet']][x]
            for y in range(numvals_by_distinguisher['yfacet']):
                if ('yfacet' in vars_by_distinguisher):
                    inner_dict[vars_by_distinguisher['yfacet']] = kwargs[vars_by_distinguisher['yfacet']][y]
                ax = axis_grid[y, x]
                for i in range(numvals_by_distinguisher['color']):
                    if ('color' in vars_by_distinguisher):
                        inner_dict[vars_by_distinguisher['color']] = kwargs[vars_by_distinguisher['color']][i]
                    for j in range(numvals_by_distinguisher['linestyle']):
                        if ('linestyle' in vars_by_distinguisher):
                            inner_dict[vars_by_distinguisher['linestyle']] = kwargs[vars_by_distinguisher['linestyle']][j]
                        input_dict = dict_union(inner_dict, outer_dict)
                        xs, ys = [], []
                        for xval in xvals:
                            input_dict[xvar] = xval
                            yval = func(**input_dict)
                            if (yval is not None):
                                xs.append(xval)
                                ys.append(yval)
                        if all(yval is None for yval in ys):  # skip this plot if there are invalid values
                            print("warning: invalid value occurred for input_dict = {}".format(input_dict))
                        plot, = ax.plot(xs, ys, color = colors[i], linestyle = linestyles[j], linewidth = 2)
                        plot.set_dash_capstyle('projecting')
                        if (('linestyle' in vars_by_distinguisher) or ('color' in vars_by_distinguisher)):
                            if (((x == 0) and (y == 0)) and ((i == numvals_by_distinguisher['color'] - 1) or (j == 0))):
                                plots_for_legend.append(plot)
                                key = ', '.join([legend_str(vars_by_distinguisher[dist], kwargs[vars_by_distinguisher[dist]][k], vars_by_distinguisher[dist] in vars_to_suppress) for (dist, k) in [('color', i), ('linestyle', j)] if (dist in vars_by_distinguisher)])
                                keys_for_legend.append(key)
                if ((numvals_by_distinguisher['yfacet'] > 1) and (x == 0)):
                    ax.annotate(legend_str(vars_by_distinguisher['yfacet'], str(kwargs[vars_by_distinguisher['yfacet']][y]), vars_by_distinguisher['yfacet'] in vars_to_suppress), xy = (0, 0.5), xytext = (-ax.yaxis.labelpad, 0), xycoords = ax.yaxis.label, textcoords = 'offset points', ha = 'right', va = 'center')
                if ((numvals_by_distinguisher['xfacet'] > 1) and (y == 0)):
                    ax.annotate(legend_str(vars_by_distinguisher['xfacet'], str(kwargs[vars_by_distinguisher['xfacet']][x]), vars_by_distinguisher['xfacet'] in vars_to_suppress), xy = (0.5, 1.01), xytext = (0, 0), xycoords = 'axes fraction', textcoords = 'offset points', ha = 'center', va = 'baseline')

        for row in axis_grid:
            for ax in row:
                rstyle(ax)
                ax.patch.set_facecolor('0.89')

        fig.text(0.5, 0.04, xvar, ha = 'center', fontsize = 14, fontweight = 'bold')
        fig.text(0.03, 0.5, yvar, va = 'center', rotation = 'vertical', fontsize = 14, fontweight = 'bold')
        plt.figlegend(plots_for_legend, keys_for_legend, 'right', fontsize = 10)
        plt.suptitle(plot_title, fontsize = 16, fontweight = 'bold')
        plt.subplots_adjust(left = 0.11, right = 0.84, top = 0.9)

        plot_params = {var : val for (var, val) in dict_union(outer_dict, constant_dict).items() if (var in params)}
        #print(plot_params)

        plot_path = '_'.join([outfile_prefix] + ['='.join(map(str, pair)) for pair in sorted(plot_params.items())]) + '.png'
        print(plot_path)
        plt.savefig(plot_path)
        #plt.show()

if __name__ == "__main__":
    main()
