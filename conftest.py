try:
    from pytest_astropy_header.display import (
        PYTEST_HEADER_MODULES,
        TESTED_VERSIONS,
    )
except ImportError:
    PYTEST_HEADER_MODULES = {}
    TESTED_VERSIONS = {}

try:
    from rocky_worlds_data_challenge import __version__ as version
except ImportError:
    version = 'unknown'


TESTED_VERSIONS['rocky_worlds_data_challenge'] = version
