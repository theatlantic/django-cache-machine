import collections
import functools
import hashlib
import logging
import socket
import sys
from itertools import chain

from django.conf import settings
from django.core.cache import cache, parse_backend_uri
from django.utils import encoding, translation
import caching.backends.redis_backend
from .settings import CACHE_PREFIX, NO_INVALIDATION, CACHE_DEBUG

try:
    import redis as redislib
except ImportError:
    redislib = None

FLUSH = CACHE_PREFIX + ':flush:'

class NullHandler(logging.Handler):

    def emit(self, record):
        pass

debug_log = logging.getLogger('caching')
debug_log.addHandler(NullHandler())

log = logging.getLogger('caching.invalidation')

try:
    from sentry.client.handlers import SentryHandler

    sentry_logger = logging.getLogger('root')
    if SentryHandler not in map(lambda x: x.__class__, sentry_logger.handlers):
        sentry_logger.addHandler(SentryHandler())
except ImportError:
    sentry_logger = None

def make_key(k, with_locale=True):
    """Generate the full key for ``k``, with a prefix."""
    key = encoding.smart_str('%s:%s' % (CACHE_PREFIX, k))
    if with_locale:
        key += encoding.smart_str(translation.get_language())
    # memcached keys must be < 250 bytes and w/o whitespace, but it's nice
    # to see the keys when using locmem.
    return hashlib.md5(key).hexdigest()


def flush_key(obj):
    """We put flush lists in the flush: namespace."""
    key = obj if isinstance(obj, basestring) else obj.cache_key
    return FLUSH + make_key(key, with_locale=False)


def safe_redis(return_type):
    """
    Decorator to catch and log any redis errors.

    return_type (optionally a callable) will be returned if there is an error.
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kw):
            try:
                return f(*args, **kw)
            except (socket.error, redislib.RedisError), e:
                log.error('redis error: %s' % e)
                if sentry_logger is not None:
                    sentry_logger.warning(
                        'RedisError: %s' % e,
                        exc_info=sys.exc_info()
                    )
                # log.error('%r\n%r : %r' % (f.__name__, args[1:], kw))
                if hasattr(return_type, '__call__'):
                    return return_type()
                else:
                    return return_type
        return wrapper
    return decorator



class Invalidator(object):

    def get_query_string(self, key):
        sql = cache.get('sql:%s' % key)
        if not sql:
            return key
        return sql

    def invalidate_keys(self, keys):
        """Invalidate all the flush lists named by the list of ``keys``."""
        if not keys:
            return
        flush, flush_keys = self.find_flush_lists(keys)

        if flush:
            if CACHE_DEBUG:
                for k in keys:
                    sql = self.get_query_string(k)
                    debug_log.debug("Flushing object    %s" % sql)
                    assoc_flush, assoc_flush_keys = self.find_flush_lists([k])
                    if assoc_flush:
                        for k in assoc_flush:
                            sql = self.get_query_string(k)
                            debug_log.debug("Flushing query     %s" % sql)
                    if assoc_flush_keys:
                        for k in assoc_flush_keys:
                            sql = self.get_query_string(k)
                            debug_log.debug("Flushing flush_key %s" % sql)
            if hasattr(cache, 'set_many_ex'):
                cache.set_many_ex(dict((k, None) for k in flush), 5)
            else:
                cache.set_many(dict((k, None) for k in flush), 5)
        if flush_keys:
            self.clear_flush_lists(flush_keys)

    def cache_objects(self, objects, query_key, query_flush, model_flush_keys=None):
        # Add this query to the flush list of each object.  We include
        # query_flush so that other things can be cached against the queryset
        # and still participate in invalidation.
        flush_keys = list(chain.from_iterable(
            [[o.flush_key(), o.model_flush_key()] for o in objects]
        ))
        if model_flush_keys is not None:
            flush_keys.extend(list(model_flush_keys))

        flush_lists = collections.defaultdict(set)
        for key in flush_keys:
            flush_lists[key].add(query_flush)
        flush_lists[query_flush].add(query_key)

        # Add each object to the flush lists of its foreign keys.
        for obj in objects:
            obj_flush = obj.flush_key()
            for key in map(flush_key, obj._cache_keys()):
                if key != obj_flush:
                    flush_lists[key].add(obj_flush)
        self.add_to_flush_list(flush_lists, watch_key=query_flush)

    def find_flush_lists(self, keys):
        """
        Recursively search for flush lists and objects to invalidate.

        The search starts with the lists in `keys` and expands to any flush
        lists found therein.  Returns ({objects to flush}, {flush keys found}).
        """
        new_keys = keys = set(map(flush_key, keys))
        flush = set(k for k in keys if not k.startswith(FLUSH))

        # Add other flush keys from the lists, which happens when a parent
        # object includes a foreign key.
        while 1:
            to_flush = self.get_flush_lists(new_keys)
            new_keys = set([])
            for k in to_flush:
                if k.startswith(FLUSH):
                    new_keys.add(k)
                else:
                    flush.add(k)
            diff = new_keys.difference(keys)
            if diff:
                keys.update(new_keys)
            else:
                return flush, keys

    def add_to_flush_list(self, mapping, **kwargs):
        """Update flush lists with the {flush_key: [query_key,...]} map."""
        flush_lists = collections.defaultdict(set)
        flush_lists.update(cache.get_many(mapping.keys()))
        for key, list_ in mapping.items():
            if flush_lists[key] is None:
                flush_lists[key] = set(list_)
            else:
                flush_lists[key].update(list_)
        cache.set_many(flush_lists)

    def get_flush_lists(self, keys):
        """Return a set of object keys from the lists in `keys`."""
        return set(e for flush_list in
                   filter(None, cache.get_many(keys).values())
                   for e in flush_list)

    def clear_flush_lists(self, keys):
        """Remove the given keys from the database."""
        cache.delete_many(keys)

    def clear(self):
        """Clears all"""
        cache.clear()

class RedisInvalidator(Invalidator):

    def safe_key(self, key):
        if ' ' in key or '\n' in key:
            log.warning('BAD KEY: "%s"' % key)
            return ''
        return key

    @safe_redis(None)
    def add_to_flush_list(self, mapping, watch_key=None):
        """Update flush lists with the {flush_key: [query_key,...]} map."""
        if not mapping or not len(mapping):
            return
        pipe = redis.pipeline()
        while 1:
            try:
                if watch_key is not None:
                    pipe.watch(watch_key)
                pipe.multi()
                for key, list_ in mapping.items():
                    for query_key in list_:
                        pipe.sadd(self.safe_key(key), query_key)
                pipe.execute()
                break
            except redislib.WatchError:
                continue
            finally:
                pipe.reset()
    
    @safe_redis(set)
    def get_flush_lists(self, keys):
        return redis.sunion(map(self.safe_key, keys))

    @safe_redis(None)
    def clear_flush_lists(self, keys):
        redis.delete(*map(self.safe_key, keys))

    @safe_redis(None)
    def clear(self):
        """Clears all"""
        redis.flushdb()

class NullInvalidator(Invalidator):

    def add_to_flush_list(self, mapping, **kwargs):
        return


if NO_INVALIDATION:
    invalidator = NullInvalidator()
elif isinstance(cache, caching.backends.redis_backend.CacheClass):
    redis = cache.redis
    invalidator = RedisInvalidator()
else:
    invalidator = Invalidator()
