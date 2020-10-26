import numpy as np

def f(ep): return ep - ep**2/np.pi * np.log(ep) + ep**2/np.pi * np.log(2)
def k(sig): return (4*sig) / (np.pi - 4 * np.sqrt(sig))

def multi_escp(r, D, N, ep, ignore_error=False):
    # if ep <= 0:
    #     return 0
    D0 = D
    Ep0 = ep
    r0 = r
    D = (np.sqrt(D) / r)**2
    ep = ep / r
    sig = (N * ep**2)/4
    t = (f(ep)/(3*D*k(sig))) + 1/(15*D)

    return t
