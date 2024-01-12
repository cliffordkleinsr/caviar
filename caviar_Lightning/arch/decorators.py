import inspect, re
from torch import nn


def merge(*ds):
    "Merge all dictionaries in `ds`"
    return {k:v for d in ds if d is not None for k,v in d.items()}

def basic_repr(flds=None):
    "Minimal `__repr__`"
    if isinstance(flds, str): flds = re.split(', *', flds)
    flds = list(flds or [])
    def _f(self):
        res = f'{type(self).__module__}.{type(self).__name__}'
        if not flds: return f'<{res}>'
        sig = ', '.join(f'{o}={getattr(self,o)!r}' for o in flds)
        return f'{res}({sig})'
    return _f

def module(*flds, **defaults):
    "Decorator to create an `nn.Module` using `f` as `forward` method"
    pa = [inspect.Parameter(o, inspect.Parameter.POSITIONAL_OR_KEYWORD) for o in flds]
    pb = [inspect.Parameter(k, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=v)
          for k,v in defaults.items()]
    params = pa+pb
    all_flds = [*flds,*defaults.keys()]

    def _f(f):
        class c(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                for i,o in enumerate(args): kwargs[all_flds[i]] = o
                kwargs = merge(defaults,kwargs)
                for k,v in kwargs.items(): setattr(self,k,v)
            __repr__ = basic_repr(all_flds)
            forward = f
        c.__signature__ = inspect.Signature(params)
        c.__name__ = c.__qualname__ = f.__name__
        c.__doc__  = f.__doc__
        return c
    return _f