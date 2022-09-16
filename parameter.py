import numpy as np
import argparse

def parse_parameters(args, default):
    parser = argparse.ArgumentParser()
    for p in default.keys():
        t = type(default[p])
        if t is list:
            parser.add_argument('--'+p, type=type(default[p][0]), nargs='*')
        elif t is bool:
            if default[p]:
                parser.add_argument('--'+p, action='store_false')
            else:
                parser.add_argument('--'+p, action='store_true')
        else:
            parser.add_argument('--'+p, type=type(default[p]), metavar='')

    parser.add_argument('--sweep', nargs=4, metavar='', 
        help='param start stop step')

    parsed_args, _ = parser.parse_known_args(args[1:])
    parsed_args = vars(parsed_args)

    params = Parameters(default)
    for p in parsed_args.keys():
        if p in default and parsed_args[p] is not None:
            params[p] = parsed_args[p]

    sweep = parsed_args['sweep']
    if sweep is not None:
        t = type(default[sweep[0]])
        for i in range(1, len(sweep)):
            sweep[i] = t(sweep[i])
        params.set_sweep(*parsed_args['sweep'])

    return params

class Parameters:
    def __init__(self, params):
        self.params = params
        self.sweep = None
        self.default = None
        self.stopiter = False
        self.start, self.stop, self.step = 0, 1, 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.stopiter:
            raise StopIteration
        if self.sweep is None:
            self.stopiter = True
        else:
            self.params[self.sweep] = self._next
            self._next += self.step
            self.stopiter = self._next > self.stop
        return self

    def set_sweep(self, name, start, stop, step):
        if self.sweep is not None:
            self.params[self.sweep] = self.default
        if name not in self.params.keys():
            raise RuntimeError(name + ' is not a parameter.')
        self.sweep = name
        self.start = start
        self.stop = stop
        self.step = step
        self.default = self.params[self.sweep]
        self.reset_sweep()

    def reset_sweep(self):
        self.params[self.sweep] = self.start
        self._next = self.start

    def sweep_range(self):
        num = int((self.stop - self.start) / self.step) + 1
        return np.linspace(self.start, self.stop, num)

    def __getattr__(self, name):
        if str(name) not in self.params.keys():
            raise AttributeError(str(name) + ' is not a parameter.')
        return self.params[str(name)]

    def __getitem__(self, key):
        if key not in self.params.keys():
            raise AttributeError(key + ' is not a parameter.')
        return self.params[key]

    def __setitem__(self, key, value):
        if key not in self.params.keys():
            raise AttributeError(key + ' is not a parameter.')
        self.params[key] = value

    def __str__(self):
        return str(self.params)
