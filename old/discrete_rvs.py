# MASTER COPY
#from scipy.stats import rv_discrete
import itertools
import numpy as np
import matplotlib.pyplot as plt
from operator import mul

ZERO_THRESH = .000000001


def get_shape(x, root_types = None):
    """Returns the shape of a list as a list of integers. If root_types is set to some list of types, it enforces that the lowest-level elements must all belong to one of those types."""
    if ((root_types is not None) and (not isinstance(root_types, list))):
        root_types = [root_types]
    if isinstance(x, np.ndarray):
        return x.shape
    elif isinstance(x, list):
        if (len(x) == 0):
            return (0,)
        else:
            first = x[0]
            shape = get_shape(first, root_types)
            for elt in x[1:]:
                if (type(elt) != type(first)):
                    raise ValueError("List elements must be of uniform type.")
                if (get_shape(elt, root_types) != shape):
                    raise ValueError("List elements must be of uniform shape.")
            return (len(x),) + shape
    else:
        if ((root_types is not None) and isinstance(root_types, list) and (type(x) not in root_types)):
            raise ValueError("Root elements must be in this set of types: %s" % str(root_types))
        return ()

def all_unique(l):
    """Returns True if all elements of a list are unique."""
    return (len(l) == len(set(l)))

def fold(arr, binary_op, zero_val = 0):
    """Folds a binary operation along a one-dimensional array. (We assume the operation is symmetric and associative."""
    x = zero_val
    for elt in arr:
        x = binary_op(x, elt)
    return x

def fold_arrays(arr_list, binary_op):
    """Folds a binary operation element-wise across list of arrays. The binary operation must be defined as the corresponding element-wise operation for numpy arrays. Arrays must all be the same shape."""
    if (len(arr_list) == 0):
        raise ValueError("Must have at least one array in list.")
    result = np.array(arr_list[0])
    for i in range(1, len(arr_list)):
        result = binary_op(result, arr_list[i])
    return result

def add_arrays(arr_list):
    """Adds arrays element-wise. Arrays must be the same shape."""
    return fold_arrays(arr_list, lambda x, y : x + y)

def and_arrays(arr_list):
    """Logical ANDs Boolean arrays element-wise. Arrays must be the same shape."""
    return fold_arrays(arr_list, lambda x, y : x & y)

def flatten(l, ltypes = (list, tuple)):
   """Fully flattens a list or tuple."""
   ltype = type(l)
   l = list(l)
   i = 0
   while (i < len(l)):
       while isinstance(l[i], ltypes):
           if (not l[i]):
               l.pop[i]
               i -= 1
               break
           else:
               l[i:i + 1] = l[i]
       i += 1
   return ltype(l)

class JointDiscreteTable(object):
    """Multidimensional array of arbitrary type with labeled variables and values."""
    def __init__(self, entries = None, shape = None, var_labels = None, value_labels = None, dtype = float, normalize = False, forbidden = None):
        if (entries is None):
            if (var_labels is not None):
                shape = tuple(map(len, value_labels))
            if (shape is None):
                self.entries = None
                self.shape = ()
                self.var_labels = []
                self.value_labels = []
            else:
                self.shape = shape
                self.entries = np.ndarray(shape = self.shape, dtype = dtype)
                if (dtype is int):
                    self.entries.fill(0)  # all zeroes
                else:
                    self.entries.fill(0.0)  # should this be default behavior?
                    #prod = reduce(mul, list(shape))
                    #self.entries.fill(1.0 / prod)  # uniform distribution
        else:
            if (shape is None):
                self.shape = get_shape(entries, [float, int, np.float64])
            else:
                if (shape != get_shape(entries, [float, int, np.float64])):
                    raise ValueError("Specified shape does not match actual shape of entries.")
                self.shape = shape
            self.entries = np.array(entries, dtype = dtype)
        self.forbidden = forbidden  # entries that must be 0
        if (self.forbidden is None):
            self.forbidden = np.zeros(shape = self.shape, dtype = bool)
        self.dtype = dtype
        self.do_norm = normalize
        if (self.do_norm):
            self.normalize()
        self.dimensions = len(self.shape)
        self.handle_labels(var_labels, value_labels)
    def _get_table_indices_from_value_list(self, value_list):
        """Converts list of variable values to the proper indices for accessing the corresponding element."""
        indices = []
        for i in xrange(self.dimensions):
            j = 0
            while (j < len(value_list)):
                if (value_list[j] in self.value_labels[i]):
                    indices.append(self.value_labels[i].index(value_list[j]))
                    break
                j += 1
            if (j == len(value_list)):
                raise ValueError("Must specify values of all variables.")
        return indices
    def get_entry(self, value_list):
        """Gets the entry at the given variable values."""
        if (not isinstance(value_list, list)):
            value_list = [value_list]
        indices = self._get_table_indices_from_value_list(value_list)
        return self.entries[tuple(indices)]
    def set_entry(self, entry, value_list):
        """Sets an entry at the given variable values."""
        indices = self._get_table_indices_from_value_list(value_list)
        if self.forbidden[tuple(indices)]:
            raise ValueError("Cannot set entry with forbidden flag.")
        self.entries[tuple(indices)] = entry
    def set_forbidden_at_indices(self, indices, flag = True):
        """Sets the forbidden flag for an entry, given indices."""
        if (len(indices) != self.dimensions):
            raise ValueError("Number of indices must equal number of dimensions.")
        self.forbidden[tuple(indices)] = flag
        if ((flag == True) and (abs(float(self.entries[tuple(indices)])) > ZERO_THRESH)):
            raise ValueError("Cannot set forbidden flag for nonzero entry.")
    def set_forbidden(self, value_list, flag = True):
        """Sets the forbidden flag for an entry, given variable values."""
        indices = self._get_table_indices_from_value_list(value_list)
        self.set_forbidden_at_indices(indices, flag)
    def normalize(self):
        """Normalizes entries of table so that they all some to 1 (probability distribution). Only does this if the array has a floating-point data type."""
        if (self.dtype == int):  # might want some other integer types here
            raise ValueError("Cannot normalize array of integer data type.")
        nrm_inv = np.sum(self.entries)
        if (nrm_inv < ZERO_THRESH):
            self.entries.fill(0.0)
        else:
            self.entries = (1.0 / nrm_inv) * self.entries
    def handle_labels(self, var_labels, value_labels):
        """Checks to make sure all variable labels and value labels are unique and that there are the proper amount. Applies default label convention if labels are not provided."""
        if (var_labels is None):
            self.var_labels = ['x' + str(i) for i in range(self.dimensions)]
        else:
            if (len(var_labels) != self.dimensions):
                raise ValueError("var_labels should have %d dimension labels" % (self.dimensions))
            if (not all_unique(var_labels)):
                raise ValueError("All var_labels should be unique.")
            self.var_labels = var_labels
        if (value_labels is None):
            self.value_labels = [[self.var_labels[i] + '_' + str(j) for j in range(self.shape[i])] for i in range(self.dimensions)]
        else:
            if (len(value_labels) != self.dimensions):
                raise ValueError("value_labels should have length %d" % (self.dimensions))
            for i in range(self.dimensions):
                if (len(value_labels[i]) != self.shape[i]):
                    raise ValueError("value_labels[%d] should have length %d" % (i, self.shape[i]))
            if (not all_unique(flatten(value_labels, ltypes = (list)))):
                raise ValueError("All value_labels should be unique.")
            self.value_labels = value_labels
    def var_index(self, var):
        """Returns the numerical index corresponding to a given variable name."""
        if (var not in self.var_labels):
            raise ValueError("%s is not a variable." % var)
        return self.var_labels.index(var)
    def var_val_indices(self, var, val):
        """Returns the ordered pair of indices corresponding to a given variable and value."""
        i = self.var_index(var)
        if (val not in self.value_labels[i]):
            raise ValueError("%s is not a value of variable %s" % (str(val), var))
        j = self.value_labels[i].index(val)
        return (i, j)
    def values_of_var(self, var):
        """Returns list of values for given variable."""
        return self.value_labels[self.var_index(var)]
    def marginalize_out(self, var_labels, normalize = None):
        """Marginalizes out a list of variables. If normalize is True, re-normalizes the resulting PDF."""
        if (normalize is None):
            normalize = self.do_norm
        if (not isinstance(var_labels, list)):
            return self.marginalize_out([var_labels], normalize)
        if (len(var_labels) == 0):
            return self
        elif (len(var_labels) == 1):
            if (self.dimensions <= 1):
                return JointDiscreteTable()
            dim = self.var_labels.index(var_labels[0])
            entries_t = self.entries.swapaxes(dim, 0)
            forbidden_t = self.forbidden.swapaxes(dim, 0)
            entries_t = add_arrays([entries_t[i] for i in xrange(self.shape[dim])])
            forbidden_t = and_arrays([forbidden_t[i] for i in xrange(self.shape[dim])])
            for i in reversed(xrange(dim - 1)):
                entries_t = entries_t.swapaxes(i, i + 1)
                forbidden_t = forbidden_t.swapaxes(i, i + 1)
            new_shape = list(self.shape)
            del(new_shape[dim])
            new_var_labels = list(self.var_labels)
            del(new_var_labels[dim])
            new_value_labels = list(self.value_labels)
            del(new_value_labels[dim])
            return JointDiscreteTable(entries_t, tuple(new_shape), new_var_labels, new_value_labels, self.dtype, normalize, forbidden_t)
        else:
            if (not all_unique(var_labels)):
                raise ValueError("All var_labels should be unique.")
            for var in var_labels:
                if (var not in self.var_labels):
                    raise ValueError("Invalid variable name: %s" % var)
            var_labels.sort(key = lambda var : self.var_labels.index(var))
            rv = self.marginalize_out([var_labels[0]], normalize)
            return rv.marginalize_out(var_labels[1:], normalize)
    def marginalize_over(self, var_labels, normalize = None):
        """Marginalizes out any variables not listed in var_labels."""
        if (normalize is None):
            normalize = self.do_norm
        if (not isinstance(var_labels, list)):
            return self.marginalize_over([var_labels], normalize)
        if (not all_unique(var_labels)):
            raise ValueError("All var_labels should be unique.")
        out_vars = [var for var in self.var_labels if (var not in var_labels)]
        return self.marginalize_out(out_vars, normalize)
    def extract(self, values, normalize = None):
        """Given list of values, return the sub-table that conditions on those values. The number of dimensions remains the same. If normalize is True, renormalizes PDF."""
        if (normalize is None):
            normalize = self.do_norm
        if (not isinstance(values, list)):
            return self.extract([values], normalize)
        if (len(values) == 0):
            return self.__class__([])
        all_vals = flatten(self.value_labels, ltypes = (list))
        for val in values:
            if (val not in all_vals):
                raise ValueError("No value: %s" % val)
        values_by_var = [[val for val in values if val in self.values_of_var(var)] for var in self.var_labels]
        for i in range(len(values_by_var)):
            if (len(values_by_var[i]) == 0):
                values_by_var[i] = self.value_labels[i]  # no variable specified means condition on all the variables
        compress_conditions = [[(self.value_labels[i][j] in values_by_var[i]) for j in range(self.shape[i])] for i in range(self.dimensions)]
        new_entries = np.array(self.entries, copy = True)
        new_forbidden = np.array(self.forbidden, copy = True)
        for i in range(self.dimensions):
            new_entries = np.compress(compress_conditions[i], new_entries, axis = i)
            new_forbidden = np.compress(compress_conditions[i], new_forbidden, axis = i)
        if isinstance(self, DiscreteRV):
            return DiscreteRV(new_entries, var_label = self.var_labels[0], value_labels = values_by_var, forbidden = new_forbidden)
        if isinstance(self, JointDiscreteRV):
            return JointDiscreteRV(new_entries, var_labels = self.var_labels, value_labels = values_by_var, forbidden = new_forbidden)
        if isinstance(self, DiscreteTable):
            return DiscreteTable(new_entries, var_label = self.var_labels[0], value_labels = values_by_var, dtype = self.dtype, normalize = normalize, forbidden = new_forbidden)
        return JointDiscreteTable(new_entries, var_labels = self.var_labels, value_labels = values_by_var, dtype = self.dtype, normalize = normalize, forbidden = new_forbidden)
    def extract_events_to_univariate_table(self, events, normalize = None):
        """Given events, that is, a list of lists of values, return the univariate table on those events. (Note: the events should be disjoint if the result is to be interpreted as a relative probability distribution on the events). Since the labeling can get complicated, that is just left to the default."""
        if (normalize is None):
            normalize = self.do_norm
        entries = [self.entry_sum(events[i]) for i in xrange(len(events))]
        if isinstance(self, JointFreqTable):
            return FreqTable(entries)
        forbidden = [self.forbidden_and(events[i]) for i in xrange(len(events))]
        if isinstance(self, JointDiscreteRV):
            return DiscreteRV(entries, forbidden = forbidden)
        return DiscreteTable(entries, normalize = normalize, forbidden = forbidden)
    def condition_on(self, values):
        """Given list of values, return the r.v. conditional on those values. The distribution marginalizes out all variables that have values given. If multiple values are given for the same variable, it is interpreted as the union of those events. If no values are provided, condition on the union of all values."""
        if (not isinstance(values, list)):
            return self.condition_on([values])
        sub_rv = self if (len(values) == 0) else self.extract(values)
        values_by_var = [[val for val in values if val in self.values_of_var(var)] for var in self.var_labels]
        out_vars = [self.var_labels[i] for i in range(self.dimensions) if (len(values_by_var[i]) > 0)]
        return sub_rv.marginalize_out(out_vars)
    def entry_sum(self, values, conditions = []):
        """Returns sum of entries at the specified values."""
        return self._arr_fold(values, conditions, 'entries')
    def forbidden_and(self, values, conditions = []):
        """Returns logical AND of the forbidden entries at the specified values."""
        return self._arr_fold(values, conditions, 'forbidden')
    def _arr_fold(self, values, conditions, array):
        """Returns sum of entries at values, given conditions. The inputs are lists of the values), taken to be union if on the same axis, intersection otherwise. Unspecified variables are marginalized out."""
        if ((not isinstance(values, list)) or (not isinstance(conditions, list))):
            if ((not isinstance(values, list))):
                values = [values]
            if (not isinstance(conditions, list)):
                conditions = [conditions]
            return self.entry_sum(values, conditions)
        for val in (values + conditions):
            if ((val not in flatten(self.value_labels, ltypes = (list)))):
                raise ValueError("%s is not a value." % str(val))
        if (len(values) == 0):  # empty event
            return False if (array == 'forbidden') else self.dtype(0)
        values_by_var = [[val for val in values if val in self.values_of_var(var)] for var in self.var_labels]
        conditions_by_var = [[val for val in conditions if val in self.values_of_var(var)] for var in self.var_labels]
        for i in xrange(self.dimensions):
            if ((len(conditions_by_var[i]) > 0) and (len(values_by_var[i]) > 0) and ([val for val in values_by_var[i] if val not in conditions_by_var[i]] == values_by_var[i])):
                return False if (array == 'forbidden') else self.dtype(0)
        condition_var_indices = [i for i in xrange(self.dimensions) if (len(conditions_by_var[i]) > 0)]
        cond_table = self.condition_on(conditions)
        values_by_var = [values_by_var[i] for i in xrange(self.dimensions) if i not in condition_var_indices]
        new_var_labels = [self.var_labels[i] for i in xrange(self.dimensions) if i not in condition_var_indices]
        new_dimensions = len(values_by_var)
        marg_table = cond_table.marginalize_over([new_var_labels[i] for i in xrange(new_dimensions) if (len(values_by_var[i]) > 0)])
        indices = [[self.var_val_indices(new_var_labels[i], values_by_var[i][j])[1] for j in range(len(values_by_var[i]))] for i in range(new_dimensions)]
        indices = [inds for inds in indices if (len(inds) > 0)]
        if (array == 'forbidden'):
            return JointDiscreteTable._arr_and_with_all_vars_specified(marg_table.forbidden, indices)
        return self.dtype(JointDiscreteTable._arr_sum_with_all_vars_specified(marg_table.entries, indices))
    @staticmethod
    def _arr_sum_with_all_vars_specified(arr, indices):
        return JointDiscreteTable._fold_operation_on_array_with_all_vars_specified(arr, indices, lambda x, y : x + y)
    @staticmethod
    def _arr_and_with_all_vars_specified(arr, indices):
        return JointDiscreteTable._fold_operation_on_array_with_all_vars_specified(arr, indices, lambda x, y : x & y)
    @staticmethod
    def _fold_operation_on_array_with_all_vars_specified(arr, indices, operation):
        if (len(indices) != len(arr.shape)):
            raise ValueError("Index list must have same dimension as input array.")
        if (len(indices) == 0):
            return self.dtype(0)
        if (len(indices[0]) == 0):
           raise ValueError("All variables must be specified.")
        if (len(indices) == 1):
            if (len(indices[0]) == 1):
                return np.array(arr[indices[0][0]], dtype = arr.dtype)
            else:
                return np.array(fold([arr[i] for i in indices[0]], operation, 0))
        else:
            if (len(indices[0]) == 1):
                return JointDiscreteTable._fold_operation_on_array_with_all_vars_specified(arr[indices[0][0]], indices[1:], operation)
            else:
                folded_arr = fold_arrays([arr[i] for i in indices[0]], operation)
                return JointDiscreteTable._fold_operation_on_array_with_all_vars_specified(folded_arr, indices[1:], operation)
    @staticmethod
    def flattened_index_to_index_list(index, shape_list):
        if (index < 0):
            raise KeyError("Index must be non-negative.")
        prod = reduce(mul, shape_list)
        if (index >= prod):
            raise KeyError("Index must be less than %d" % prod)
        if (len(shape_list) == 1):
            return [index]
        if (len(shape_list) > 1):
            return [index / reduce(mul, shape_list[1:])] + JointDiscreteTable.flattened_index_to_index_list(index % reduce(mul, shape_list[1:]), shape_list[1:])
    def __getitem__(self, index):
        return self.entries[index]
    def __setitem__(self, index, item):
        if not (self.forbidden[index].all()):
            raise ValueError("Cannot set entry with forbidden flag.")
        self.entries[index] = item
    def __len__(self):
        return len(self.entries)
    def __str__(self):
        return str(self.entries)
    def str2(self):
        """Shows the distribution with value labels."""
        s = '['
        for i in range(len(self.value_labels[0])):
            s += str(self.value_labels[0][i]) + ': '
            if (len(self.shape) == 1):
                s += str(self.entries[i])
            else:
                rv = self.extract(self.value_labels[0][i], False)
                rv = rv.marginalize_out(self.var_labels[0], False)
                s += rv.str2()
            if (i < len(self.value_labels[0]) - 1):
                s += ', '
        s += ']'
        return s
    def __repr__(self):
        return self.__str__()


class DiscreteTable(JointDiscreteTable):
    """Discrete univariate table."""
    def __init__(self, entries = None, var_label = None, value_labels = None, dtype = float, normalize = False, forbidden = None):
        """entries is a 1-D array or list, var_label is a string (variable name), value_labels is a 1-D list of value names."""
        if (isinstance(var_label, list) and (len(var_label) > 0)):
            var_label = var_label[0]
        if (isinstance(value_labels, list) and (len(value_labels) > 0) and isinstance(value_labels[0], list)):
            value_labels = value_labels[0]
        var_labels = [var_label] if (var_label is not None) else None
        value_labels = [value_labels] if (value_labels is not None) else None
        super(DiscreteTable, self).__init__(entries = entries, shape = None, var_labels = var_labels, value_labels = value_labels, dtype = dtype, normalize = normalize, forbidden = forbidden)
    def set_var_label(self, var_label):
        self.var_labels = [var_label]
    def set_value_labels(self, value_labels):
        self.value_labels = [value_labels]


class JointFreqTable(JointDiscreteTable):
    """Multivariate frequency table."""
    def __init__(self, freqs = None, shape = None, var_labels = None, value_labels = None):
        super(JointFreqTable, self).__init__(freqs, shape, var_labels, value_labels, dtype = int, normalize = False)
    def sum(self):
        """Sum of all the frequencies."""
        return np.sum(self.entries)
    def get_prob_dist(self):
        """Returns probability distribution induced by the frequencies in the table. No values are explicitly forbidden even if their frequencies are zero."""
        probs = np.array(self.entries, dtype = float)
        total = self.sum()
        if (total != 0):
            probs /= total
        if (len(self.shape) > 1):
            return JointDiscreteRV(probs = probs, var_labels = self.var_labels, value_labels = self.value_labels)
        return DiscreteRV(probs = probs, var_label = self.var_labels[0], value_labels = self.value_labels[0])
    def __add__(self, other):
        """Entry-wise adds another JointFreqTable with same dimensions. If any labels mismatch, resets to the default."""
        if ((not isinstance(other, JointFreqTable)) or (other.shape != self.shape)):
            raise ValueError("Both addends must be JointFreqTables with identical dimensions.""")
        var_labels = self.var_labels if (self.var_labels == other.var_labels) else None
        value_labels = self.value_labels if (self.value_labels == other.value_labels) else None
        if isinstance(self, FreqTable):
            return FreqTable(self.entries + other.entries, var_labels[0], value_labels[0])
        return JointFreqTable(self.entries + other.entries, None, var_labels, value_labels)
    def __mul__(self, c):
        """Multiplies table by a constant (not by another frequency table -- why would one ever do that?)."""
        if isinstance(self, FreqTable):
            return FreqTable(c * self.entries, self.var_labels[0], self.value_labels[0])
        return JointFreqTable(c * self.entries, None, self.var_labels, self.value_labels)
    def __rmul__(self, c):
        return self * c


class FreqTable(DiscreteTable, JointFreqTable):
    """Univariate frequency table."""
    def __init__(self, freqs = None, var_label = None, value_labels = None):
        JointFreqTable.__init__(self, freqs = freqs, var_labels = (None if (var_label is None) else [var_label]), value_labels = (None if (value_labels is None) else [value_labels]))


# for later: compute log probs alongside/instead of probs
class JointDiscreteRV(JointDiscreteTable):
    """Multivariate discrete random variable."""
    def __init__(self, probs = None, shape = None, var_labels = None, value_labels = None, forbidden = None):
        super(JointDiscreteRV, self).__init__(probs, shape, var_labels, value_labels, dtype = float, normalize = True, forbidden = forbidden)
        self.cumsums = np.cumsum(self.entries)
        self.has_support = not ((len(self.cumsums) == 0) or (self.cumsums[-1] < .999))
    def prob(self, values, conditions = []):
        """Returns probability of values given conditions. The inputs are lists of the values), taken to be union if on the same axis, intersection otherwise. Unspecified variables are marginalized out."""
        return super(JointDiscreteRV, self).entry_sum(values, conditions)
    def generate(self, seed = None):
        """Randomly generates element of the distribution as list of value labels."""
        if (seed is not None):
            np.random.seed(seed)
        r = np.random.random()
        num_elements = reduce(mul, list(self.shape))
        if (num_elements == 0):
            return []
        if (not self.has_support):
            return []
            #raise RuntimeError("Distribution has no support.")
        for i in range(num_elements):
            if (r <= self.cumsums[i]):
                index_list = JointDiscreteRV.flattened_index_to_index_list(i, list(self.shape))
                return [self.value_labels[j][index_list[j]] for j in range(self.dimensions)]
    def to_conditional_distribution(self, condition_var_labels):
        if (not isinstance(condition_var_labels, list)):
            condition_var_labels = [condition_var_labels]
        condition_var_labels = list(set(condition_var_labels))  # de-dupe
        indices = [self.var_labels.index(var) for var in condition_var_labels]
        assert(len(indices) == len(condition_var_labels))
        rvs = []
        prior = []
        for condition in itertools.product(*[self.value_labels[i] for i in indices]):
            rvs.append(self.condition_on(list(condition)))
            prior.append(self.prob(list(condition)))
        condition_value_labels = [self.value_labels[i] for i in indices]
        crv =  ConditionalRV(rvs, condition_var_labels, condition_value_labels)
        crv.set_prior(prior)
        return crv
    def __add__(self, other):
        """Entry-wise averages another JointDiscreteRV with same dimensions. If any labels mismatch, resets to the default. Forbidden tables taken to be the AND of those of the two addends."""
        if ((not isinstance(other, JointDiscreteRV)) or (other.shape != self.shape)):
            raise ValueError("Both addends must be JointDiscreteRVs with identical dimensions.""")
        var_labels = self.var_labels if (self.var_labels == other.var_labels) else None
        value_labels = self.value_labels if (self.value_labels == other.value_labels) else None
        forbidden = self.forbidden & other.forbidden
        if isinstance(self, DiscreteRV):
            return DiscreteRV(self.entries + other.entries, var_labels[0], value_labels[0], forbidden)
        return JointDiscreteRV(self.entries + other.entries, None, var_labels, value_labels, forbidden)


class JointUniformRV(JointDiscreteRV):
    """Joint distribution in which all probabilities are equal."""
    def __init__(self, shape, var_labels = None, value_labels = None):
        super(JointUniformRV, self).__init__(shape = shape, var_labels = var_labels, value_labels = value_labels)


class DiscreteRV(JointDiscreteRV, DiscreteTable):
    """Discrete univariate distribution."""
    def __init__(self, probs, var_label = None, value_labels = None, forbidden = None):
        """probs is a 1-D array or list, var_label is a string (variable name), value_labels is a 1-D list of value names."""
        DiscreteTable.__init__(self, entries = probs, var_label = var_label, value_labels = value_labels, dtype = float, normalize = True, forbidden = forbidden)
        self.cumsums = np.cumsum(self.entries)
        self.has_support = not ((len(self.cumsums) == 0) or (self.cumsums[-1] < .999))
    def generate(self, seed = None):
        gen = JointDiscreteRV.generate(self, seed)
        if (len(gen) > 0):
            return gen[0]
        else:
            return None
    def generate_ordering(self, seed = None):
        if (seed is not None):
            np.random.seed(seed)
        if (len(self) == 0):
            return []
        first_var = self.generate()
        if (first_var is None):
            return []
        new_rv = self.extract([v for v in self.value_labels[0] if (v != first_var)])
        return [first_var] + new_rv.generate_ordering()


class UniformRV(DiscreteRV):
    """Discrete univariate uniform distribution."""
    def __init__(self, num_vals, var_label = None, value_labels = None):
        if (num_vals > 0):
            probs = [1.0 / num_vals for i in range(num_vals)]
        else:
            probs = []
        super(UniformRV, self).__init__(probs, var_label = var_label, value_labels = value_labels)


class JointDiscreteIndependentRV(JointDiscreteRV):
    """Joint discrete independent random variable. More efficient memory-wise."""
    def __init__(self, univariate_rvs):
        """univariate_rvs is a list of 1-D JointDiscreteRV objects, or a list of lists of probabilities."""
        if ((len(univariate_rvs) > 0) and isinstance(univariate_rvs[0], list)):
            self.univariate_rvs = [DiscreteRV(rv) for rv in univariate_rvs]
            for i in range(len(univariate_rvs)):
                self.univariate_rvs[i].set_var_label('x' + str(i))
                self.univariate_rvs[i].set_value_labels(['x' + str(i) + '_' + str(j) for j in range(len(self.univariate_rvs[i]))])
        else:
            self.univariate_rvs = univariate_rvs
        self.dimensions = len(self.univariate_rvs)
        self.shape = tuple([len(self.univariate_rvs[i]) for i in range(self.dimensions)])
        self.var_labels = [self.univariate_rvs[i].var_labels[0] for i in range(self.dimensions)]
        self.value_labels = [self.univariate_rvs[i].value_labels[0] for i in range(self.dimensions)]
        if (not all_unique(flatten(self.value_labels, ltypes = (list)))):
            raise ValueError("All value labels should be unique.")
        self.dtype = float
        self.do_norm = True
    def marginalize_out(self, var_labels, normalize = True):
        if (not isinstance(var_labels, list)):
            return self.marginalize_out([var_labels], normalize)
        if (not all_unique(var_labels)):
            raise ValueError("All var_labels should be unique.")
        new_univariate_rvs = [rv for rv in self.univariate_rvs if (rv.var_labels[0] not in var_labels)]
        return JointDiscreteIndependentRV(new_univariate_rvs)
    def extract(self, values):
        """Given list of values, return the (normalized) sub-distribution that conditions on those values. The number of dimensions remains the same."""
        if (not isinstance(values, list)):
            return self.extract([values])
        if (len(values) == 0):
            return JointDiscreteIndependentRV([])
        all_vals = flatten(self.value_labels, ltypes = (list))
        for val in values:
            if (val not in all_vals):
                raise ValueError("%s is not a value." % val)
        values_by_var = [[val for val in values if val in self.values_of_var(var)] for var in self.var_labels]
        for i in range(len(values_by_var)):
            if (len(values_by_var[i]) == 0):
                values_by_var[i] = self.value_labels[i]  # no variable specified means condition on all the variables
        new_univariate_rvs = [self.univariate_rvs[i].extract(values_by_var[i]) for i in range(self.dimensions)]
        return JointDiscreteIndependentRV(new_univariate_rvs)
    def entry_sum(self, values, conditions = []):
        """Returns probability of values given conditions. The inputs are lists of the values), taken to be union if on the same axis, intersection otherwise. Unspecified variables are marginalized out."""
        if ((not isinstance(values, list)) or (not isinstance(conditions, list))):
            if ((not isinstance(values, list))):
                values = [values]
            if (not isinstance(conditions, list)):
                conditions = [conditions]
            return self.prob(values, conditions)
        for val in (values + conditions):
            if ((val not in flatten(self.value_labels, ltypes = list))):
                raise ValueError("%s is not a value." % str(val))
        values_by_var = [[val for val in values if val in self.values_of_var(var)] for var in self.var_labels]
        conditions_by_var = [[val for val in conditions if val in self.values_of_var(var)] for var in self.var_labels]
        for i in range(self.dimensions):
            if ((len(conditions_by_var[i]) > 0) and (len(values_by_var[i]) > 0) and ([val for val in values_by_var[i] if val not in conditions_by_var[i]] == values_by_var[i])):
                return 0.0
        condition_var_indices = [i for i in range(self.dimensions) if (len(conditions_by_var[i]) > 0)]
        cond_rv = self.condition_on(conditions)
        values_by_var = [values_by_var[i] for i in range(self.dimensions) if i not in condition_var_indices]
        new_var_labels = [self.var_labels[i] for i in range(self.dimensions) if i not in condition_var_indices]
        new_dimensions = len(values_by_var)
        marg_rv = cond_rv.marginalize_over([new_var_labels[i] for i in range(new_dimensions) if (len(values_by_var[i]) > 0)])
        indices = [[self.var_val_indices(new_var_labels[i], values_by_var[i][j])[1] for j in range(len(values_by_var[i]))] for i in range(new_dimensions)]
        indices = [inds for inds in indices if (len(inds) > 0)]
        return JointDiscreteIndependentRV._prob_with_all_vars_specified(marg_rv.univariate_rvs, indices)
    def prob(self, values, conditions = []):
        return self.entry_sum(values, conditions)
    @staticmethod
    def _prob_with_all_vars_specified(univariate_rvs, indices):
        """Given list of univariate rv's and 2D list of indices from each dimension to sum over, returns the probability of the event."""
        if (len(indices) != len(univariate_rvs)):
            raise ValueError("Length of index list must be equal to number of variables.")
        if (len(indices) == 0):
            return 1.0
        if (len(indices[0]) == 0):
           raise ValueError("All variables must be specified.")
        first_event_prob = sum([univariate_rvs[0][i] for i in indices[0]])
        return (first_event_prob * JointDiscreteIndependentRV._prob_with_all_vars_specified(univariate_rvs[1:], indices[1:]))
    def generate(self, seed = None):
        if (seed is not None):
            np.random.seed(seed)
        return [self.univariate_rvs[i].generate() for i in range(self.dimensions)]
    def __getitem__(self, i):
        return self.univariate_rvs[i]
    def __len__(self):
        return len(self.univariate_rvs)
    def __str__(self):
        return str(self.univariate_rvs)
    def __repr__(self):
        return self.__str__()
    def __add__(self, other):
        if ((not isinstance(other, JointDiscreteIndependentRV)) or (len(self) != len(other))):
            raise ValueError("Both addends must be JointDiscreteIndependentRVs with identical dimensions.""")
        return JointDiscreteIndependentRV([self[i] + other[i] for i in xrange(len(self))])

class ConditionalRV(object):
    def __init__(self, rvs, condition_var_labels = None, condition_value_labels = None):
        """Takes a list of JointDiscreteRVs. Each element of the list represents the conditional distribution of some other set of variables conditioned on some values of the condition variables. condition_value_labels should be a list of lists of value labels, each list corresponding to each variable."""
        self.rvs = rvs
        self.condition_var_labels = condition_var_labels
        self.condition_value_labels = condition_value_labels
        num_values = len(self.rvs)
        if (num_values > 0):
            self.var_labels = self.rvs[0].var_labels
            self.value_labels = self.rvs[0].value_labels
            for i in xrange(1, num_values):
                if ((self.rvs[i].var_labels != self.var_labels) or (self.rvs[i].value_labels != self.value_labels)):
                    raise ValueError("All input random variables must have same set of variables and values.")
        if (self.condition_var_labels is None):
            self.condition_var_labels = ["y0"]
        all_var_labels = list(self.condition_var_labels)
        if (num_values > 0):
            all_var_labels += self.var_labels
        if (not all_unique(self.var_labels)):
            raise ValueError("All var_labels should be unique.")
        assert(num_values % len(self.condition_var_labels) == 0)
        if ((self.condition_value_labels is None) and (len(self.condition_var_labels) == 1)):
            self.condition_value_labels = [[("%s_%d" % (self.condition_var_labels[0], i)) for i in xrange(num_values)]]
        assert(np.prod(map(len, self.condition_value_labels)) == num_values)
        self.shape = (tuple(map(len, self.condition_value_labels)), self.rvs[0].shape)
    def set_prior(self, prior):
        """Set a prior distribution on the conditions. Input can either be a list of probabilities or a DiscreteRV of the appropriate length."""
        if isinstance(prior, list):
            prior = DiscreteRV(prior)
        if (len(prior) != len(self)):
            raise ValueError("Number of entries must match number of conditions.")
        self.prior = prior
    def condition_on(self, values):
        """values is a set of values (tuples) on which to condition. Returns the probability distribution conditioning on these. If values is a list of multiple values, there must be a prior distribution on the conditions, in which case the resulting distribution is the weighted sum of distributions on each condition."""
        if (not isinstance(values, list)):
            values = [values]
        for i in xrange(len(values)):
            if (not isinstance(values[i], tuple)):
                values[i] = (values[i], )
        if (len(values) == 0):
            return None  # placeholder?
        elif (len(values) == 1):
            for pair in enumerate(itertools.product(*self.condition_value_labels)):
                if (pair[1] == values[0]):
                    return self.rvs[pair[0]]
            raise KeyError("Invalid condition values.")
        else:
            if (not hasattr(self, 'prior')):
                raise ValueError("Must have prior on conditions in order to obtain conditional distribution on multiple events.")
            entries = np.zeros(self.rvs[0].entries.shape, dtype = float)
            for pair in enumerate(itertools.product(*self.condition_value_labels)):
                if (pair[1] in values):
                    entries += self.prior[pair[0]] * self.rvs[pair[0]].entries
            rv = self.rvs[0].__class__(entries)
            rv.var_labels = self.rvs[0].var_labels
            rv.value_labels = self.rvs[0].value_labels
            return rv
    def to_joint_distribution(self):
        if (not hasattr(self, 'prior')):
            raise ValueError("Must have prior on conditions in order to obtain conditional distribution on multiple events.")
        entries = []
        for i in xrange(len(self)):
            entries.append(self.rvs[i] * self.prior[i])
        entries = np.array(entries)
        entries = entries.reshape(self.shape[0] + self.shape[1])
        var_labels = self.condition_var_labels + self.rvs[0].var_labels
        value_labels = self.condition_value_labels + self.rvs[0].value_labels
        return JointDiscreteRV(entries, var_labels = var_labels, value_labels = value_labels)
    def __len__(self):
        return len(self.rvs)
    def __getitem__(self, i):
        return self.rvs[i]
    def __str__(self):
        s = ''
        for pair in enumerate(itertools.product(*self.condition_value_labels)):
            s += '\n'
            if (len(self.condition_var_labels) == 1):
                s += (str(pair[1][0]) + '\n')
            else:
                s += (str(pair[1]) + '\n')
            s += (str(self.rvs[pair[0]]) + '\n')
        return s
    def __repr__(self):
        return str(self)


class ModRV(DiscreteRV):
    """Probability distribution over the integers mod M."""
    def __init__(self, probs, var_label = None):
        self.M = len(probs)
        DiscreteRV.__init__(self, probs, var_label, [i for i in xrange(self.M)], None)
    def binary_op_with(self, other, op):
        """Takes a binary function on mod-M integers and applies the operation to two probability distributions."""
        probs = np.zeros(self.M, dtype = float)
        if isinstance(other, ModRV):
            if (self.M != other.M):
                raise ValueError("Random variables must have the same modulus.")
            for i in xrange(self.M):
                for j in xrange(self.M):
                    probs[op(i, j) % self.M] += self[i] * other[j]
        else:
            for i in xrange(self.M):
                probs[op(i, other) % self.M] += self[i]
        return ModRV(probs)
    def __add__(self, other):
        return self.binary_op_with(other, lambda x, y : x + y)
    def __sub__(self, other):
        return self.binary_op_with(other, lambda x, y : x - y)
    def __mul__(self, other):
        return self.binary_op_with(other, lambda x, y : x * y)
    def __neg__(self):
        probs = np.zeros(self.M, dtype = float)
        for i in xrange(self.M):
            probs[i] = self[-i % self.M]
        return ModRV(probs)
    @classmethod
    def uniform(cls, M):
        return cls([1.0 / M for i in xrange(M)])

class RealRV(JointDiscreteRV):
    def check_real(self):
        """Checks whether all value labels are real numbers."""
        for label in flatten(self.value_labels):
            if (not isinstance(label, (float, int))):
                raise ValueError("All values must be real numbers.")

class DiscreteRealRV(DiscreteRV, RealRV):
    """Represents a real-valued univariate discrete random variable."""
    def __init__(self, probs, vals = None, var_label = None, forbidden = None):
        if (vals is None):
            vals = range(len(probs))
        probs_vals = zip(probs, vals)
        probs_vals.sort(key = (lambda x : x[1]))
        probs = [x[0] for x in probs_vals]
        vals = [x[1] for x in probs_vals]
        DiscreteRV.__init__(self, probs, var_label, vals, forbidden)
        self.check_real()
    def moment(self, n):
        """Computes the nth moment."""
        total = 0.0
        for val in self.value_labels[0]:
            total += self.prob(val) * (val ** n)
        return total
    def expectation(self):
        return self.moment(1)
    def variance(self):
        return self.moment(2) - (self.expectation()) ** 2
    def eval(self, x, comparison = 'le'):
        """For any of the comparison functions < ('lt'), > ('gt'), <= ('le'), >= ('ge'), == ('eq'), returns the probability that the random variable will have that relation to x."""
        if (comparison == 'eq'):
            return self.prob(x)
        i = 0
        cmp_fn = (lambda x, y : x >= y) if (comparison in ['le', 'gt']) else (lambda x, y : x > y)
        while ((i < len(self.value_labels[0])) and (cmp_fn(x, self.value_labels[0][i]))):
            i += 1
        if (comparison in ['le', 'lt']):
            return 0.0 if (i == 0) else self.cumsums[i - 1]
        if (comparison in ['ge', 'gt']):
            return 1.0 if (i == 0) else 1 - self.cumsums[i - 1]
    def plot_cmf(self, samps = 1000):
        vals = np.linspace(self.value_labels[0][0], self.value_labels[0][-1], samps)
        Fs = np.array([self.eval(val) for val in vals])
        plt.plot(vals, Fs)
        plt.xlabel('x')
        plt.ylabel('F(x)')
        plt.show()






jdt_float = JointDiscreteTable([[[.1,.05],[.05,.1]],[[.3,.0],[.2,.05]],[[.0, .05],[.05, .05]]], dtype = float, normalize = True)
jdt_int = JointDiscreteTable([[[3, 7],[2,4]],[[5,16],[9,11]],[[13,1],[0,5]]], dtype = int)
dt_int = DiscreteTable([3, 7, 17, 9, 19], dtype = int)
jdrv = JointDiscreteRV([[[.1,.05],[.05,.1]],[[.3,.0],[.2,.05]],[[.0, .05],[.05, .05]]])
jurv = JointUniformRV((2, 3, 4))
drv = DiscreteRV([.2,.4,.1,.3])
urv = UniformRV(8)
jdirv = JointDiscreteIndependentRV([[.25,.75],[.4,.6]])
jft = JointFreqTable([[[3, 7],[2,4]],[[5,16],[9,11]],[[13,1],[0,5]]])
ft = FreqTable([7,2,0,7])
crv = ConditionalRV([jdrv.condition_on('x0_0'), jdrv.condition_on('x0_1'), jdrv.condition_on('x0_2'), jdrv.condition_on('x0_0'), jdrv.condition_on('x0_2'), jdrv.condition_on('x0_2')], ['y1', 'y2'], [['y1_0', 'y1_1', 'y1_2'], ['y2_0', 'y2_1']])
