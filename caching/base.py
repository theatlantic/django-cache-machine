from django.db import models

from .settings import CACHEMACHINE_ENABLED


if CACHEMACHINE_ENABLED:
    from .db import CachingManager, CachingMixin, CachingQuerySet
else:
    def cached_with(obj, f, f_key, timeout=None):
        return f()

    class CachingMixin(object):
        cache_enabled = False

    class CachingManager(models.Manager):

        def no_cache(self):
            return self.get_query_set()

    class CachingQuerySet(models.query.QuerySet):

        def no_cache(self):
            return self


__all__ = ['CachingManager', 'CachingMixin', 'CachingQuerySet']
