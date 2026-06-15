
print('oLIMpus_v2 version June 15, 2026')
from zeus21 import * 
from .inputs_LIM import * 
from .luminosities_LIM import * 
from .coefficients_LIM import * 
from .correlations_LIM import * 
from .maps_LIM import * 

import warnings
warnings.filterwarnings("ignore", category=UserWarning) #to silence unnecessary warning in mcfit
