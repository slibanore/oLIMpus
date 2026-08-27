from importlib.metadata import version as _installed_version

from ._version import __version__

print(f"oLIMpus version {__version__}")
print(f"zeus21 version {_installed_version('zeus21')}")

from .inputs_LIM import *
from .luminosities_LIM import *
from .burstiness_LIM import *
from .coefficients_LIM import *
from .correlations_LIM import *
from .maps_LIM import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning) #to silence unnecessary warning in mcfit
warnings.filterwarnings("ignore", category=RuntimeWarning)
