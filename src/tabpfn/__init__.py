from tabpfn.classifier import TabPFNClassifier
from tabpfn.regressor import TabPFNRegressor
from importlib.metadata import version

try:
    __version__ = version(__name__)
except ImportError:
    __version__ = "unknown"

__all__ = [
    "TabPFNClassifier",
    "TabPFNRegressor",
    "__version__",
]
