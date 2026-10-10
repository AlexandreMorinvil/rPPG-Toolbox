"""Heart-rate estimation for the launcher; implemented in evaluation/signal_analysis.py."""

import sys

from .paths import TOOLBOX_ROOT

if str(TOOLBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLBOX_ROOT))

from evaluation.signal_analysis import *  # noqa: E402,F401,F403
from evaluation.signal_analysis import _detrend_operator  # noqa: E402,F401
