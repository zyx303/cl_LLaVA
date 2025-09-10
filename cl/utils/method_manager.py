from methods.er_baseline import ER
from methods.ewc import EWCpp
from methods.mir import MIR
from methods.clib import CLIB
from methods.der import DER
from methods.xder import XDER
from methods.cama_nodc import CAMA_NODC
from methods.cama import CAMA
from methods.base import BaseCLMethod
"""
Note: sdlora is imported lazily to avoid import errors if the optional file is incomplete.
We also lazy-import olora.
"""


def select_method(args, n_classes, model):
    kwargs = vars(args)

    methods = {
        'er': ER,
        'ewc++': EWCpp,
        'mir': MIR,
        'clib': CLIB,
        'der': DER,
        'xder': XDER,
        'cama_nodc': CAMA_NODC,
        'cama': CAMA,
        'base': BaseCLMethod
    }

    # Lazy imports for optional methods
    if args.mode == 'sdlora':
        from methods.sdlora import SDLoRA  # type: ignore
        methods['sdlora'] = SDLoRA
    if args.mode == 'olora':
        from methods.olora import OLoRA  # type: ignore
        methods['olora'] = OLoRA

    if args.mode in methods:
        method = methods[args.mode](n_classes, model, **kwargs)
    else:
        raise NotImplementedError(f"Choose the args.mode in {list(methods.keys())}")

    return method
