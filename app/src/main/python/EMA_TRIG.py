import numpy as np
import pandas as pd


#%%EMA TRIGGER Function
def EMA_TRIG(Features, bndrs=0, dstrs=0): 
    """
    Feautrues: Comes from the Feature_Extraction Function
    bndrs and dstrs: Are two constant M*N and M*(N+1) np arrays which are directly read from the phone memory. Set as constants here for now.
    """
    bndrs = np.zeros((7,4))
    dstrs = np.zeros((7,5))
    
    """ BODY OF CODE  """
    
    t=0 # Output is either 0 or 1:       0: Trigger, 1: Not trigger
    return t