# 'import evolvekit.examples'
from evolvekit.examples.advanced import *
from evolvekit.examples.inspectors import *

# Combine __all__
from evolvekit.examples.basic import __all__ as basic_all
from evolvekit.examples.advanced import __all__ as advanced_all
from evolvekit.examples.inspectors import __all__ as inspectors_all

__all__ = basic_all + advanced_all + inspectors_all
