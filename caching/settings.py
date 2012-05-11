import warnings

from django.conf import settings

CACHEMACHINE_ENABLED = getattr(settings, 'CACHEMACHINE_ENABLED', True)
CACHE_PREFIX = getattr(settings, 'CACHEMACHINE_PREFIX', None)
CACHE_DEBUG = getattr(settings, 'CACHEMACHINE_DEBUG', None)
NO_INVALIDATION = getattr(settings, 'CACHEMACHINE_NO_INVALIDATION', False)

def warn_deprecated_setting(settings_suffix):
    message = "The setting CACHE_%s has been changed to CACHEMACHINE_%s"
    warnings.warn(DeprecationWarning(message % (settings_suffix, settings_suffix)))

# TODO: Remove these deprecated getattrs
if CACHE_PREFIX is None:
    CACHE_PREFIX = getattr(settings, 'CACHE_PREFIX', '')
    if CACHE_PREFIX == '' and hasattr(settings, 'CACHE_PREFIX'):
        warn_deprecated_setting("PREFIX")

if CACHE_DEBUG is None:
    CACHE_DEBUG = getattr(settings, 'CACHE_DEBUG', False)
    if hasattr(settings, 'CACHE_DEBUG'):
        warn_deprecated_setting("DEBUG")