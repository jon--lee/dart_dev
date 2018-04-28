import numpy as np

def enforce(params):
    assert params['trials'] == 20
    assert params['epochs'] == 100
    assert params['lr'] == .01
    assert params['arch'] == [64, 64]
    assert params['t'] == 500
