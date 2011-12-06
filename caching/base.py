import collections
import functools
import logging

from django.conf import settings
from django.core.cache import cache, parse_backend_uri
from django.db import models
from django.db.models import signals
from django.utils import encoding

from .invalidation import invalidator, flush_key, make_key

from datetime import timedelta
second_delta = timedelta(seconds=1)

class NullHandler(logging.Handler):

    def emit(self, record):
        pass


log = logging.getLogger('caching')
log.addHandler(NullHandler())

FOREVER = 0
NO_CACHE = -1
CACHE_PREFIX = getattr(settings, 'CACHE_PREFIX', '')
CACHE_DEBUG = getattr(settings, 'CACHE_DEBUG', False)

class StopCaching(Exception):
    """Raised when a query is determined to be uncacheable"""
    pass

class CachingManager(models.Manager):

    # Tell Django to use this manager when resolving foreign keys.
    use_for_related_fields = True

    def get_query_set(self):
        return CachingQuerySet(self.model)

    def contribute_to_class(self, cls, name):
        signals.pre_save.connect(self.pre_save, sender=cls)
        signals.post_save.connect(self.post_save, sender=cls)
        signals.post_delete.connect(self.post_delete, sender=cls)
        return super(CachingManager, self).contribute_to_class(cls, name)

    def pre_save(self, sender, instance, raw, **kwargs):
        """
        Flush all cached queries associated with a model if a field has been
        changed that is a known constraint in an already cached query.
        
        TODO: Associate flush lists with constraint columns, so that we don't
        need to flush the whole table.
        """
        # The raw boolean means we're loading the database from a fixture, so
        # we don't want to mess with it.
        if raw:
            return

        # We only need to flush the model if the post already exists; when new
        # instances are created it flushes the model cache, so calling flush
        # here would be redundant
        if not instance.id:
            return
        
        cls = instance.__class__
        if not hasattr(cls.objects, 'invalidate_model'):
            return
        
        # Grab the original object, before the to-be-saved changes
        orig = cls.objects.no_cache().get(pk=instance.id)
        
        constraint_key = 'cols:%s' % instance.model_key
        flush_cols = invalidator.get_flush_lists([constraint_key])
        if len(flush_cols) == 0:
            return
        
        for col in flush_cols:
            if not hasattr(orig, col) or not hasattr(instance, col):
                continue
            if getattr(orig, col) != getattr(instance, col):
                instance.invalidate_model = True
                return

    def post_save(self, instance, created, **kwargs):
        if instance.invalidate_model or created:
            self.invalidate_model()
        else:
            self.invalidate(instance)
        
    def post_delete(self, instance, **kwargs):
        self.invalidate(instance)
    
    def m2m_changed(self, instance, action, *args, **kwargs):
        if action[:4] != "post":
            return
        self.invalidate(instance)
    
    def invalidate(self, *objects):
        keys = []
        for o in objects:
            if hasattr(o, '_cache_keys'):
                keys += list(o._cache_keys())
        """Invalidate all the flush lists associated with ``objects``."""
        if len(keys) > 0:
            invalidator.invalidate_keys(keys)

    def invalidate_model(self):
        """
        Invalidate all the flush lists associated with the models of ``objects``.
        
        This effectively flushes all queries linked to a given model.
        """
        model_key = self.model._model_key()
        if CACHE_DEBUG:
            log.debug("Invalidating model %s" % model_key[2:])
        if not hasattr(self.model, '_model_key'):
            raise Exception((
                "The model Manager of %s uses caching, but the " + \
                "model does not. Needs CachingMixIn."
            ) % ".".join([self.model.__module__, self.model.__name__]))
        invalidator.invalidate_keys([model_key])
        # Set cols to none for 5 seconds before expiring (prevents race condition)
        cache.delete(u'cols:%s' % model_key)

    def cache(self, timeout=None):
        return self.get_query_set().cache(timeout)

    def no_cache(self):
        return self.cache(NO_CACHE)


class CacheMachine(object):
    """
    Handles all the cache management for a QuerySet.

    Takes the string representation of a query and a function that can be
    called to get an iterator over some database results.
    """

    def __init__(self, queryset, cached=None):
        self.query_string = queryset.query_string()
        self.iter_function = queryset.iterator(skip_cache=True)
        self.timeout = getattr(queryset, 'timeout', None)
        self.queryset = queryset
        self.query = self.queryset.query
        self.cached = cached

    def query_key(self):
        """Generate the cache key for this query."""
        return make_key('qs:%s' % self.query_string, with_locale=False)

    def __iter__(self):
        try:
            query_key = self.query_key()
        except query.EmptyResultSet:
            raise StopIteration

        # If anything has been passed to the queryset via extra, we can't
        # ensure that the proper fields will be associated, so we don't cache
        if len(self.query.extra) > 0:
            # Put a persistent lock on the query key to prevent it from going
            # through the cache again
            cache.set(query_key, None, timeout=0)
            self.cached = None

        if self.cached is not None and not isinstance(self.cached, int):
            if CACHE_DEBUG:
                log.debug('cache hit: %s' % self.query_string)
            for obj in self.cached:
                obj.from_cache = True
                yield obj
            return

        iterator = self.iter_function

        # Do the database query, cache it once we have all the objects.
        to_cache = []
        try:
            while True:
                obj = iterator.next()
                obj.from_cache = False
                to_cache.append(obj)
                yield obj
        except StopIteration:
            if to_cache:
                self.cache_objects(to_cache)
            raise

    def cache_objects(self, objects):
        """Cache query_key => objects, then update the flush lists."""
        query_key = self.query_key()
        query_flush = flush_key(self.query_string)
        try:
            constraints = self.get_constraints()
        except StopCaching:
            # Put a persistent lock on the query key to prevent it from going
            # through the cache again
            cache.set(query_key, None, timeout=0)
        else:
            cache.add(query_key, objects, timeout=self.timeout)
            model_flush_keys = set([flush_key(k[5:]) for k in constraints.keys()])
            model_flush_keys.add(flush_key(self.queryset.model._model_key()))
            invalidator.cache_objects(objects, query_key, query_flush, model_flush_keys)
            invalidator.add_to_flush_list(constraints, watch_key=query_flush)

    column_map = {}
    table_map = {}
    _compiler = None

    @property
    def compiler(self):
        if self._compiler is None:
            compiler = self.query.get_compiler(using=self.queryset.db)
            compiler.pre_sql_setup()
            compiler.get_columns()
            self._compiler = compiler
        return self._compiler

    def get_columns_for_order_by(self, opts=None):
        columns = tuple()
        if opts is None:
            opts = self.query.model._meta
        if opts.db_table not in self.column_map:
            self.column_map[opts.db_table] = {}
        only_load = self.compiler.deferred_to_columns()
        for field, model in opts.get_fields_with_model():
            alias = self.query.included_inherited_models[model]
            table = self.query.alias_map[alias][models.sql.constants.TABLE_NAME]
            if table in only_load and field.column not in only_load[table]:
                continue
            columns += ((table, field.name),)
            self.column_map[opts.db_table][field.column] = field.name
        return columns

    def map_column_to_field_name(self, table, col):
        field = col
        
        if table in self.table_map and table not in self.column_map:
            model = self.table_map[table]
            # Update column_map for table
            self.get_columns_for_order_by(model._meta)
        if table in self.column_map and col in self.column_map[table]:
            field = self.column_map[table][col]
        
        return field

    def has_offset_or_limit(self):
        """Whether the query has a LIMIT or OFFSET defined."""
        return self.query.low_mark != 0 or self.query.high_mark is not None

    def get_ordering_fields(self):
        compiler = self.compiler
        
        if self.query.extra_order_by:
            ordering = self.query.extra_order_by
        elif not self.query.default_ordering:
            ordering = self.query.order_by
        else:
            ordering = self.query.order_by or self.query.model._meta.ordering
        
        select_aliases = compiler._select_aliases
        all_columns = self.get_columns_for_order_by()
        
        # used to check whether we've already seen something
        processed_pairs = set()
        order_fields = tuple()
        for field in ordering:
            if field == '?':
                # order_by('?') means order at random. Can't cache that obvs
                raise StopCaching
            if isinstance(field, int):
                field = abs(field)
                if field < len(all_columns):
                    table, field_name = all_columns[field]
                    if (table, field_name) not in processed_pairs:
                        processed_pairs.add((table, field_name))
                        order_fields += ((table, field_name),)
                continue
            col, order = models.sql.query.get_order_dir(field)
            # This is something like a HAVING COUNT(*) > X, not relevant
            if col in self.query.aggregate_select:
                continue
            if '.' in field:
                # This came in through an extra(order_by=...) addition.
                # order_by can't be parsed, so we can't cache
                raise StopCaching
            elif col not in self.query.extra_select:
                # 'col' is of the form 'field' or 'field1__field2' or
                # '-field1__field2__field', etc.
                for table, col, _ in compiler.find_ordering_name(field, self.query.model._meta):
                    field_name = self.map_column_to_field_name(table, col)
                    if (table, field_name) not in processed_pairs:
                        processed_pairs.add((table, field_name))
                        order_fields += ((table, field_name),)
            else:
                # order_by is a column in an extra_select, no good, abort
                raise StopCaching
        
        return order_fields
            
    def get_constraints(self):
        """
        Get the table/column constraints associated with the queryset's query.
        
        TODO: Look at join information.
        """
        constraints = collections.defaultdict(set)
        stack = [self.query.where]
        while stack:
            curr_where = stack.pop()
            for k, v in curr_where.__dict__.items():
                if isinstance(v, (list, tuple)):
                    for i, item in enumerate(v):
                        if isinstance(item, models.sql.where.WhereNode):
                            stack.append(item)
                        elif isinstance(item, (tuple)):
                            if len(item) > 0 and isinstance(item[0], models.sql.where.Constraint):
                                constraint = item[0]
                                model = constraint.field.model
                                name = constraint.field.name
                                if not hasattr(model, '_model_key'):
                                    continue
                                # If the primary key, don't add to list
                                if model._meta.pk and model._meta.pk.name == name:
                                    continue
                                constraint_key = u'cols:%s' % model._model_key()
                                if model._meta.db_table not in self.table_map:
                                    self.table_map[model._meta.db_table] = model
                                constraints[constraint_key].add(name)
        if self.has_offset_or_limit():
            order_fields = self.get_ordering_fields()
            for table, name in order_fields:
                constraint_key = u'cols:m:%s' % table
                constraints[constraint_key].add(name)
        return constraints
    

class CachingQuerySet(models.query.QuerySet):
    
    cache_machine = None

    def __init__(self, *args, **kw):
        super(CachingQuerySet, self).__init__(*args, **kw)
        self.timeout = None

    def flush_key(self):
        return flush_key(self.query_string())

    def query_string(self):
        sql, params = self.query.get_compiler(using=self.db).as_sql()
        return sql % params

    def iterator(self, skip_cache=False):
        iterator = super(CachingQuerySet, self).iterator
        if self.timeout == NO_CACHE or skip_cache:
            return iterator()
        try:
            # Work-around for Django #12717.
            query_string = self.query_string()
        except models.sql.query.EmptyResultSet:
            return iterator()

        if self.cache_machine is not None:
            return iter(self.cache_machine)

        self.cache_machine = CacheMachine(self)
        query_key = self.cache_machine.query_key()
        cached = cache.get(query_key, default=-1)
        # If the value is None, that means it has a lock on it after
        # being cleared (if the key doesn't exist, we would get -1).
        # We return the regular queryset iterator, which yields an
        # uncached result set.
        if cached is None:
            return iterator()
        else:
            self.cache_machine.cached = cached
        return iter(self.cache_machine)

    def cache(self, timeout=None):
        qs = self._clone()
        qs.timeout = timeout
        return qs

    def no_cache(self):
        return self.cache(NO_CACHE)

    def _clone(self, *args, **kw):
        qs = super(CachingQuerySet, self)._clone(*args, **kw)
        qs.timeout = self.timeout
        return qs


class CachingMixin:
    """Inherit from this class to get caching and invalidation helpers."""
    
    """Whether to invalidate the model in the post_save. Set in the pre_save"""
    invalidate_model = False
    
    def flush_key(self):
        return flush_key(self)

    @property
    def cache_key(self):
        """Return a cache key based on the object's primary key."""
        return self._cache_key(self.pk)

    @classmethod
    def _cache_key(cls, pk):
        """
        Return a string that uniquely identifies the object.

        For the Addon class, with a pk of 2, we get "o:addons.addon:2".
        """
        key_parts = ('o', cls._meta, pk)
        return ':'.join(map(encoding.smart_unicode, key_parts))

    def model_flush_key(self):
        return flush_key(self.model_key)

    @property
    def model_key(self):
        """Returns a cache key based on the object's model."""
        return self._model_key()

    @classmethod
    def _model_key(cls):
        """
        Return a string that uniquely identifies the model the object
        belongs to.
        
        For the Addon class, we get "m:addons.addon".
        """
        key_parts = ('m', cls._meta.db_table)
        return ':'.join(map(encoding.smart_unicode, key_parts))

    def _model_keys(self):
        """
        Return the model cache key for self plus all related foreign keys.
        """
        return (self.model_key,) + self._cache_keys()

    def _cache_keys(self):
        """Return the cache key for self plus all related foreign keys."""
        fks = dict((f, getattr(self, f.attname)) for f in self._meta.fields
                    if isinstance(f, models.ForeignKey))

        keys = [fk.rel.to._cache_key(val) for fk, val in fks.items()
                if val is not None and hasattr(fk.rel.to, '_cache_key')]
        return (self.cache_key,) + tuple(keys)


def _function_cache_key(key):
    return make_key('f:%s' % key, with_locale=True)


def cached(function, key_, duration=None):
    """Only calls the function if ``key`` is not already in the cache."""
    key = _function_cache_key(key_)
    val = cache.get(key)
    if val is None:
        if CACHE_DEBUG:
            log.debug('cache miss for %s' % key)
        val = function()
        cache.setex(key, val, duration)
    elif CACHE_DEBUG:
        log.debug('cache hit for %s' % key)
    return val


def cached_with(obj, f, f_key, timeout=None):
    """Helper for caching a function call within an object's flush list."""
    try:
        obj_key = (obj.query_key() if hasattr(obj, 'query_key')
                   else obj.cache_key)
    except AttributeError:
        log.warning(u'%r cannot be cached.' % obj)
        return f()

    key = '%s:%s' % tuple(map(encoding.smart_str, (f_key, obj_key)))
    # Put the key generated in cached() into this object's flush list.
    func_cache_key = _function_cache_key(key)
    invalidator.add_to_flush_list({
        obj.flush_key(): [func_cache_key],
        obj.model_flush_key(): [func_cache_key],
    }, watch_key=obj.flush_key())
    return cached(f, key, timeout)


class cached_method(object):
    """
    Decorator to cache a method call in this object's flush list.

    The external cache will only be used once per (instance, args).  After that
    a local cache on the object will be used.

    Lifted from werkzeug.
    """
    def __init__(self, func):
        self.func = func
        functools.update_wrapper(self, func)

    def __get__(self, obj, type=None):
        if obj is None:
            return self
        _missing = object()
        value = obj.__dict__.get(self.__name__, _missing)
        if value is _missing:
            w = MethodWrapper(obj, self.func)
            obj.__dict__[self.__name__] = w
            return w
        return value


class MethodWrapper(object):
    """
    Wraps around an object's method for two-level caching.

    The first call for a set of (args, kwargs) will use an external cache.
    After that, an object-local dict cache will be used.
    """
    def __init__(self, obj, func):
        self.obj = obj
        self.func = func
        functools.update_wrapper(self, func)
        self.cache = {}

    def __call__(self, *args, **kwargs):
        k = lambda o: o.cache_key if hasattr(o, 'cache_key') else o
        arg_keys = map(k, args)
        kwarg_keys = [(key, k(val)) for key, val in kwargs.items()]
        key = 'm:%s:%s:%s:%s' % (self.obj.cache_key, self.func.__name__,
                                 arg_keys, kwarg_keys)
        if key not in self.cache:
            f = functools.partial(self.func, self.obj, *args, **kwargs)
            self.cache[key] = cached_with(self.obj, f, key)
        return self.cache[key]
