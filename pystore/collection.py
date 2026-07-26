#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# PyStore: Flat-file datastore for timeseries data
# https://github.com/ranaroussi/pystore
#
# Copyright 2018-2020 Ran Aroussi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import os
import time
import shutil
import uuid
import dask.dataframe as dd
import multitasking

from . import utils
from .item import Item
from . import config


class Collection(object):
    def __repr__(self):
        return "PyStore.collection <%s>" % self.collection

    def __init__(self, collection, datastore, engine="pyarrow"):
        self.engine = engine
        self.datastore = datastore
        self.collection = collection
        self.items = self.list_items()
        self.snapshots = self.list_snapshots()

    def _item_path(self, item, as_string=False):
        p = utils.make_path(self.datastore, self.collection, item)
        if as_string:
            return str(p)
        return p

    @multitasking.task
    def _list_items_threaded(self, **kwargs):
        self.items = self.list_items(**kwargs)

    def list_items(self, **kwargs):
        dirs = utils.subdirs(utils.make_path(self.datastore, self.collection))
        if not kwargs:
            return set(dirs)

        matched = []
        for d in dirs:
            meta = utils.read_metadata(utils.make_path(
                self.datastore, self.collection, d))
            del meta["_updated"]

            m = 0
            keys = list(meta.keys())
            for k, v in kwargs.items():
                if k in keys and meta[k] == v:
                    m += 1

            if m == len(kwargs):
                matched.append(d)

        return set(matched)

    def item(self, item, snapshot=None, filters=None, columns=None, calculate_divisions=True):
        return Item(item, self.datastore, self.collection,
                    snapshot, filters, columns, engine=self.engine, calculate_divisions=calculate_divisions)

    def index(self, item, last=False):
        data = dd.read_parquet(self._item_path(item, as_string=True),
                               columns="index", engine=self.engine)
        if not last:
            return data.index.compute()

        return float(str(data.index).split(
                     "\nName")[0].split("\n")[-1].split(" ")[0])

    def delete_item(self, item, reload_items=False):
        shutil.rmtree(self._item_path(item))
        self.items.remove(item)
        if reload_items:
            self.items = self._list_items_threaded()
        return True

    @multitasking.task
    def write_threaded(self, item, data, metadata={},
                       npartitions=None,
                       overwrite=False, epochdate=False,
                       reload_items=False, append=False, **kwargs):
        return self.write(item, data, metadata,
                          npartitions, overwrite,
                          epochdate, reload_items, append,
                          **kwargs)

    def write(self, item, data, metadata={},
              npartitions=None, overwrite=False,
              epochdate=False, reload_items=False, append=False,
              **kwargs):

        if utils.path_exists(self._item_path(item)) and not overwrite:
            raise ValueError("""
                Item already exists. To overwrite, use `overwrite=True`.
                Otherwise, use `<collection>.append()`""")

        if isinstance(data, Item):
            data = data.to_pandas()
        else:
            # work on copy
            data = data.copy()

        if epochdate or "datetime" in str(data.index.dtype):
            data = utils.datetime_to_int64(data)
            if (1 == data.index.nanosecond).any() and "times" not in kwargs:
                kwargs["times"] = "int96"

        if data.index.name == "":
            data.index.name = "index"

        # if npartitions is None:
        #     memusage = data.memory_usage(deep=True).sum()
        #     if isinstance(data, dd.DataFrame):
        #         npartitions = int(
        #             1 + memusage.compute() // config.PARTITION_SIZE)
        #         data.repartition(npartitions=npartitions)
        #     else:
        #         npartitions = int(
        #             1 + memusage // config.PARTITION_SIZE)
        if isinstance(data, dd.DataFrame):
            # repartition()'s first positional argument is `divisions`, so the
            # count has to be passed by keyword. Leaving the frame alone when no
            # count is asked for keeps concatenated frames from being collapsed
            # into a single in-memory partition.
            if npartitions:
                data = data.repartition(npartitions=npartitions)
        else:
            data = dd.from_pandas(data, npartitions=npartitions or 1)

        dd.to_parquet(data, self._item_path(item, as_string=True),
                      compression="snappy", engine=self.engine, append=append, ignore_divisions=True, **kwargs)

        utils.write_metadata(utils.make_path(
            self.datastore, self.collection, item), metadata)

        # update items
        self.items.add(item)
        if reload_items:
            self._list_items_threaded()

    def append(self, item, data, npartitions=None, epochdate=False,
               threaded=False, reload_items=False, skip_existing=True, **kwargs):

        if not utils.path_exists(self._item_path(item)):
            raise ValueError(
                """Item do not exists. Use `<collection>.write(...)`""")

        # work on copy
        data = data.copy()

        if skip_existing:
            try:
                if epochdate or ("datetime" in str(data.index.dtype) and
                                 any(data.index.nanosecond) > 0):
                    data = utils.datetime_to_int64(data)
                old_index = dd.read_parquet(self._item_path(item, as_string=True),
                                            columns=[], engine=self.engine
                                            ).index.compute()
                data = data[~data.index.isin(old_index)]
            except Exception as e:
                print(str(e))
                return

        # if data.empty:
        if len(data.index) == 0:
            return

        if data.index.name == "":
            data.index.name = "index"

        if not isinstance(data, dd.DataFrame):
            new = dd.from_pandas(data, npartitions=npartitions or 1)
        elif npartitions:
            new = data.repartition(npartitions=npartitions)
        else:
            new = data

        # Store the new rows as extra files inside the existing dataset instead
        # of concatenating them with what is already stored and rewriting the
        # whole item: the cost is the size of `data`, not the size of the item.
        # `skip_existing` has already dropped rows whose index is present, so
        # nothing is gained by rewriting. (The old concat path cannot run on
        # dask >= 2025 in any case: dd.concat() of a parquet read and an
        # in-memory frame fails to generate metadata for categorical dtypes.)
        #
        # Filenames have to be unique, or a second append would overwrite the
        # `part.0.parquet` that the first one wrote. Callers that pass their own
        # deterministic `name_function` get idempotent re-writes instead, which
        # is what makes re-ingesting the same day safe.
        #
        # Because each append writes its own files rather than rewriting the item,
        # `data` must carry the same dtypes as what is already stored: a column
        # that is categorical in one file and a string in another makes the
        # dataset unreadable. Build every batch the same way.
        kwargs.setdefault(
            "name_function",
            lambda i, uid=uuid.uuid4().hex[:12]: "%s.part.%s.parquet" % (uid, i))

        # `threaded` is still accepted for backwards compatibility, but the write
        # is now a single dask call and no longer needs the threaded wrapper.
        dd.to_parquet(new, self._item_path(item, as_string=True),
                      compression="snappy", engine=self.engine,
                      ignore_divisions=True, **kwargs)

        # update items
        self.items.add(item)
        if reload_items:
            self._list_items_threaded()

    def create_snapshot(self, snapshot=None):
        if snapshot:
            snapshot = "".join(
                e for e in snapshot if e.isalnum() or e in [".", "_"])
        else:
            snapshot = str(int(time.time() * 1000000))

        src = utils.make_path(self.datastore, self.collection)
        dst = utils.make_path(src, "_snapshots", snapshot)

        shutil.copytree(src, dst,
                        ignore=shutil.ignore_patterns("_snapshots"))

        self.snapshots = self.list_snapshots()
        return True

    def list_snapshots(self):
        snapshots = utils.subdirs(utils.make_path(
            self.datastore, self.collection, "_snapshots"))
        return set(snapshots)

    def delete_snapshot(self, snapshot):
        if snapshot not in self.snapshots:
            # raise ValueError("Snapshot `%s` doesn't exist" % snapshot)
            return True

        shutil.rmtree(utils.make_path(self.datastore, self.collection,
                                      "_snapshots", snapshot))
        self.snapshots = self.list_snapshots()
        return True

    def delete_snapshots(self):
        snapshots_path = utils.make_path(
            self.datastore, self.collection, "_snapshots")
        shutil.rmtree(snapshots_path)
        os.makedirs(snapshots_path)
        self.snapshots = self.list_snapshots()
        return True
