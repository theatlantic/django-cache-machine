from django.db import models

from .settings import CACHEMACHINE_ENABLED
from .db import CachingQuerySet

if CACHEMACHINE_ENABLED:
    from .db import CachingManager, CachingMixin
else:
    def cached_with(obj, f, f_key, timeout=None):
        return f()

    class CachingMixin(object):
        cache_enabled = False

    class CachingManager(models.Manager):
        pass
