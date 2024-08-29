from importlib import metadata
from semver import Version

__version__ = metadata.version("smplcodec")
SEMVER = Version.parse(__version__)
MAJOR = SEMVER.major
MINOR = SEMVER.minor
PATCH = SEMVER.patch
