from . import fdetr

def build_model(args):
    return fdetr.build(args)
