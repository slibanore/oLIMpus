
from importlib.metadata import version

from zeus21 import *

print(f"oLIMpus version {version('oLIMpus')}")
print(f"zeus21 version {version('zeus21')}")

from .inputs_LIM import *
from .luminosities_LIM import *
from .coefficients_LIM import *
from .correlations_LIM import *
from .maps_LIM import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning) #to silence unnecessary warning in mcfit
warnings.filterwarnings("ignore", category=RuntimeWarning)
