from typing import Union, Optional, Collection, Any, Tuple
from collections.abc import Iterable,Iterator,Generator,Sequence,MutableSequence,MutableMapping

ListOrItem = Union[Collection[Any],int,float,str]
ListSizes = Collection[Tuple[int,int]]
OptListOrItem = Optional[ListOrItem]

@DeprecationWarning
def listify(p:OptListOrItem=None, q:OptListOrItem=None):
    "Make `p` listy and the same length as `q`."
    if p is None: p=[]
    elif isinstance(p, str):          p = [p]
    elif not isinstance(p, Iterable): p = [p]
    #Rank 0 tensors in PyTorch are Iterable but don't have a length.
    else:
        try: a = len(p)
        except: p = [p]
    n = q if type(q)==int else len(p) if q is None else len(q)
    if len(p)==1: p = p * n
    assert len(p)==n, f'List len mismatch ({len(p)} vs {n})'
    return list(p)

def is_listy(x:Any)->bool:
    return isinstance(x, (tuple,list))

# DF PROC
def FillMissing(data, continuous_cols):
    data[continuous_cols] = data[continuous_cols].fillna(0)
    return data

def Categorify(data, categorical_cols):
    for col in categorical_cols:
        data[col] = data[col].astype('category').cat.as_ordered()
    return data

def Normalize(data, continuous_cols):
    for col in continuous_cols:
        mean = data[col].mean()
        std = data[col].std()
        data[col] = (data[col] - mean) / std
    return data

def Preprocess(data, categorical_cols, continuous_cols, functions):
    for func in functions:
        if func in [FillMissing, Normalize]:
            data = func(data, continuous_cols=continuous_cols)
        elif func in [Categorify]:
            data = func(data, categorical_cols=categorical_cols)
    return data

# EMB SZ PROC
def emb_sz_rule(n_cat:int)->int: return min(600, round(1.6 * n_cat**0.56))

def def_emb_sz(classes, n, sz_dict=None):
    "Pick an embedding size for `n` depending on `classes` if not given in `sz_dict`."
    sz_dict = {} if sz_dict is None else sz_dict 
    n_cat = len(classes[n])
    sz = sz_dict.get(n, int(emb_sz_rule(n_cat)))  # rule of thumb
    return n_cat,sz

