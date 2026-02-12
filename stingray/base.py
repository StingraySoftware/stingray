"""Base classes"""

from __future__ import annotations

from collections.abc import Iterable
from collections import OrderedDict

import pickle
import warnings
import copy

import numpy as np
from astropy.table import Table
from astropy.time import Time, TimeDelta
from astropy.units import Quantity
from stingray.loggingconfig import setup_logger

from .io import _can_save_longdouble, _can_serialize_meta
from .utils import (
    sqsum,
    assign_value_if_none,
    make_nd_into_arrays,
    make_1d_arrays_into_nd,
    get_random_state,
    find_nearest,
    rebin_data,
)
from .gti import (
    create_gti_mask,
    check_gtis,
    cross_two_gtis,
    join_gtis,
    gti_border_bins,
    get_btis,
    merge_gtis,
    get_total_gti_length,
    bin_intervals_from_gtis,
    time_intervals_from_gtis,
)
from typing import TYPE_CHECKING, Type, TypeVar, Union

if TYPE_CHECKING:
    from xarray import Dataset
    from pandas import DataFrame
    from astropy.timeseries import TimeSeries
    from astropy.time import TimeDelta
    import numpy.typing as npt

    TTime = Union[Time, TimeDelta, Quantity, npt.ArrayLike]
    Tso = TypeVar("Tso", bound="StingrayObject")


__all__ = [
    "convert_table_attrs_to_lowercase",
    "interpret_times",
    "reduce_precision_if_extended",
    "StingrayObject",
    "StingrayTimeseries",
]

logger = setup_logger()


def convert_table_attrs_to_lowercase(table: Table) -> Table:
    """Convert the column names of an Astropy Table to lowercase."""
    new_table = Table()
    for col in table.colnames:
        new_table[col.lower()] = table[col]
    for key in table.meta.keys():
        new_table.meta[key.lower()] = table.meta[key]

    return new_table


class StingrayObject(object):
    """This base class defines some general-purpose utilities.

    The main purpose is to have a consistent mechanism for:

    + round-tripping to and from Astropy Tables and other dataframes

    + round-tripping to files in different formats

    The idea is that any object inheriting :class:`StingrayObject` should,
    just by defining an attribute called ``main_array_attr``, be able to perform
    the operations above, with no additional effort.

    ``main_array_attr`` is, e.g. ``time`` for :class:`StingrayTimeseries` and
    :class:`Lightcurve`, ``freq`` for :class:`Crossspectrum`, ``energy`` for
    :class:`VarEnergySpectrum`, and so on. It is the array with which all other
    attributes are compared: if they are of the same shape, they get saved as
    columns of the table/dataframe, otherwise as metadata.
    """

    not_array_attr: list = []

    def __init__(cls, *args, **kwargs) -> None:
        if not hasattr(cls, "main_array_attr"):
            raise RuntimeError(
                "A StingrayObject needs to have the main_array_attr attribute specified"
            )

    @property
    def main_array_length(self):
        if getattr(self, self.main_array_attr, None) is None:
            return 0
        return np.shape(np.asanyarray(getattr(self, self.main_array_attr)))[0]

    def data_attributes(self) -> list[str]:
        """Clean up the list of attributes, only giving out those pointing to data.

        List all the attributes that point directly to valid data. This method goes through all the
        attributes of the class, eliminating methods, properties, and attributes that are complicated
        to serialize such as other ``StingrayObject``, or arrays of objects.

        This function does not make difference between array-like data and scalar data.

        Returns
        -------
        data_attributes : list of str
            List of attributes pointing to data that are not methods, properties,
            or other ``StingrayObject`` instances.
        """
        return [
            attr
            for attr in dir(self)
            if (
                not attr.startswith("__")
                and attr not in ["main_array_attr", "not_array_attr"]
                and not isinstance(getattr(self.__class__, attr, None), property)
                and not callable(value := getattr(self, attr))
                and not isinstance(value, StingrayObject)
                and not np.asanyarray(value).dtype == "O"
            )
        ]

    def array_attrs(self) -> list[str]:
        """List the names of the array attributes of the Stingray Object.

        By array attributes, we mean the ones with the same size and shape as
        ``main_array_attr`` (e.g. ``time`` in ``EventList``)

        Returns
        -------
        attributes : list of str
            List of array attributes.
        """

        main_attr = getattr(self, getattr(self, "main_array_attr"))
        if main_attr is None:
            return []

        return [
            attr
            for attr in self.data_attributes()
            if (
                not attr.startswith("_")
                and not attr == self.main_array_attr
                and isinstance(getattr(self, attr), Iterable)
                and attr not in self.not_array_attr
                and not isinstance(getattr(self, attr), str)
                and np.shape(getattr(self, attr))[0] == np.shape(main_attr)[0]
            )
        ]

    def internal_array_attrs(self) -> list[str]:
        """List the names of the internal array attributes of the Stingray Object.

        These are array attributes that can be set by properties, and are generally indicated
        by an underscore followed by the name of the property that links to it (E.g.
        ``_counts`` in ``Lightcurve``).
        By array attributes, we mean the ones with the same size and shape as
        ``main_array_attr`` (e.g. ``time`` in ``EventList``)

        Returns
        -------
        attributes : list of str
            List of internal array attributes.
        """

        main_attr = getattr(self, "main_array_attr")
        main_attr_value = getattr(self, main_attr)
        if main_attr_value is None:
            return []

        all_attrs = []
        for attr in self.data_attributes():
            if (
                not attr == "_" + self.main_array_attr  # e.g. _time in lightcurve
                and attr not in ["_" + a for a in self.not_array_attr]
                and not np.isscalar(value := getattr(self, attr))
                and value is not None
                and not np.size(value) == 0
                and attr.startswith("_")
                and np.shape(value)[0] == np.shape(main_attr_value)[0]
            ):
                all_attrs.append(attr)

        return all_attrs

    def meta_attrs(self) -> list[str]:
        """List the names of the meta attributes of the Stingray Object.

        By array attributes, we mean the ones with a different size and shape
        than ``main_array_attr`` (e.g. ``time`` in ``EventList``)

        Returns
        -------
        attributes : list of str
            List of meta attributes.
        """
        array_attrs = self.array_attrs() + [self.main_array_attr] + self.internal_array_attrs()

        all_meta_attrs = [
            attr
            for attr in self.data_attributes()
            if (attr not in array_attrs and not attr.startswith("_"))
        ]
        if self.not_array_attr is not None and len(self.not_array_attr) >= 1:
            all_meta_attrs += self.not_array_attr
        return all_meta_attrs

    def dict(self) -> dict:
        """Return a dictionary representation of the object."""

        main_attr = self.main_array_attr
        meta_attrs = self.meta_attrs()
        array_attrs = self.array_attrs()
        internal_array_attrs = self.internal_array_attrs()

        results = OrderedDict()
        results[main_attr] = getattr(self, main_attr)

        for attr in internal_array_attrs:
            if isinstance(getattr(self.__class__, attr.lstrip("_"), None), property):
                attr = attr.lstrip("_")
            results[attr] = getattr(self, attr)

        for attr in array_attrs:
            results[attr] = getattr(self, attr)

        for attr in meta_attrs:
            results[attr] = getattr(self, attr)

        return results

    def pretty_print(self, func_to_apply=None, attrs_to_apply=[], attrs_to_discard=[]) -> str:
        """Return a pretty-printed string representation of the object.

        This is useful for debugging, and for interactive use.

        Other parameters
        ----------------
        func_to_apply : function
            A function that modifies the attributes listed in ``attrs_to_apply``.
            It must return the modified attributes and a label to be printed.
            If ``None``, no function is applied.
        attrs_to_apply : list of str
            Attributes to be modified by ``func_to_apply``.
        attrs_to_discard : list of str
            Attributes to be discarded from the output.
        """
        print(self.__class__.__name__)
        print("_" * len(self.__class__.__name__))
        items = self.dict()
        results = ""
        np.set_printoptions(threshold=3, edgeitems=1)
        for attr in items.keys():
            if attr in attrs_to_discard:
                continue
            value = items[attr]
            label = f"{attr:<15}: {items[attr]}"

            if isinstance(value, Iterable) and not isinstance(value, str):
                size = np.shape(value)
                if len(size) == 1:
                    label += f" (size {size[0]})"
                else:
                    label += f" (shape {size})"

            if func_to_apply is not None and attr in attrs_to_apply:
                new_value, new_label = func_to_apply(items[attr])
                label += f"\n{attr + ' (' +new_label + ')':<15}: {new_value}"

            results += label + "\n"
        return results

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return self.pretty_print()

    def __eq__(self, other_ts):
        """Compare two :class:`StingrayObject` instances with ``==``.

        All attributes (internal, array, meta) are compared.
        """

        if not isinstance(other_ts, type(self)):
            raise ValueError(f"{type(self)} can only be compared with a {type(self)} Object")

        self_arr_attrs = self.array_attrs()
        other_arr_attrs = other_ts.array_attrs()

        if not set(self_arr_attrs) == set(other_arr_attrs):
            return False

        self_meta_attrs = self.meta_attrs()
        other_meta_attrs = other_ts.meta_attrs()

        if not set(self_meta_attrs) == set(other_meta_attrs):
            return False

        for attr in self.meta_attrs():
            # They are either both scalar or arrays
            if np.isscalar(getattr(self, attr)) != np.isscalar(getattr(other_ts, attr)):
                return False

            if np.isscalar(getattr(self, attr)):
                if not getattr(self, attr, None) == getattr(other_ts, attr, None):
                    return False
            else:
                if not np.array_equal(
                    getattr(self, attr, None), getattr(other_ts, attr, None), equal_nan=True
                ):
                    return False

        for attr in self.array_attrs():
            if not np.array_equal(getattr(self, attr), getattr(other_ts, attr), equal_nan=True):
                return False

        for attr in self.internal_array_attrs():
            if not np.array_equal(getattr(self, attr), getattr(other_ts, attr), equal_nan=True):
                return False

        return True

    def _default_operated_attrs(self):
        operated_attrs = [attr for attr in self.array_attrs() if not attr.endswith("_err")]
        return operated_attrs

    def _default_error_attrs(self):
        return [attr for attr in self.array_attrs() if attr.endswith("_err")]

    def get_meta_dict(self) -> dict:
        """Give a dictionary with all non-None meta attrs of the object."""
        meta_attrs = self.meta_attrs()
        meta_dict = {}
        for key in meta_attrs:
            val = getattr(self, key)
            if val is not None:
                meta_dict[key] = val
        return meta_dict

    def to_astropy_table(self, no_longdouble=False) -> Table:
        """Create an Astropy Table from a ``StingrayObject``

        Array attributes (e.g. ``time``, ``pi``, ``energy``, etc. for
        ``EventList``) are converted into columns, while meta attributes
        (``mjdref``, ``gti``, etc.) are saved into the ``meta`` dictionary.

        Other Parameters
        ----------------
        no_longdouble : bool
            If True, reduce the precision of longdouble arrays to double precision.
            This needs to be done in some cases, e.g. when the table is to be saved
            in an architecture not supporting extended precision (e.g. ARM), but can
            also be useful when an extended precision is not needed.
        """
        data = {}
        array_attrs = self.array_attrs() + [self.main_array_attr] + self.internal_array_attrs()

        for attr in array_attrs:
            vals = np.asanyarray(getattr(self, attr))
            if no_longdouble:
                vals = reduce_precision_if_extended(vals)
            data[attr] = vals

        ts = Table(data)
        meta_dict = self.get_meta_dict()
        for attr in meta_dict.keys():
            if no_longdouble:
                meta_dict[attr] = reduce_precision_if_extended(meta_dict[attr])
            value = meta_dict[attr]
            rep = repr(value)
            # Work around issue with Numpy 2.0 and Yaml serializer.
            if "np.float" in rep:
                value = float(value)
            elif "np.int" in rep:
                value = int(value)
            meta_dict[attr] = value
        ts.meta.update(meta_dict)

        return ts

    @classmethod
    def from_astropy_table(cls: Type[Tso], ts: Table) -> Tso:
        """Create a Stingray Object object from data in an Astropy Table.

        The table MUST contain at least a column named like the
        ``main_array_attr``.
        The rest of columns will form the array attributes of the
        new object, while the attributes in ds.attrs will
        form the new meta attributes of the object.

        """
        cls = cls()

        if len(ts) == 0:
            # return an empty object
            return cls

        array_attrs = ts.colnames

        # Set the main attribute first
        for attr in array_attrs:
            if attr.lower() == cls.main_array_attr:  # type: ignore
                mainarray = np.array(ts[attr])  # type: ignore
                setattr(cls, cls.main_array_attr, mainarray)  # type: ignore
                break

        for attr in array_attrs:
            if attr.lower() == cls.main_array_attr:  # type: ignore
                continue
            setattr(cls, attr.lower(), np.array(ts[attr]))

        attributes_left_unchanged = []
        for key, val in ts.meta.items():
            if (
                isinstance(getattr(cls.__class__, key.lower(), None), property)
                and getattr(cls.__class__, key.lower(), None).fset is None
            ):
                attributes_left_unchanged.append(key)
                continue

            setattr(cls, key.lower(), val)
        if len(attributes_left_unchanged) > 0:
            # Only warn once, if multiple properties are affected.
            attrs = ",".join(attributes_left_unchanged)
            warnings.warn(
                f"The input table contains protected attribute(s) of StingrayTimeseries: {attrs}. "
                "These values are set internally by the class, and cannot be overwritten. "
                "This issue is common when reading from FITS files using `fmt='fits'`."
                " If this is the case, please consider using `fmt='ogip'` instead."
            )
        return cls

    def to_xarray(self) -> Dataset:
        """Create an ``xarray`` Dataset from a `StingrayObject`.

        Array attributes (e.g. ``time``, ``pi``, ``energy``, etc. for
        ``EventList``) are converted into columns, while meta attributes
        (``mjdref``, ``gti``, etc.) are saved into the ``ds.attrs`` dictionary.
        """
        from xarray import Dataset

        data = {}
        array_attrs = self.array_attrs() + [self.main_array_attr] + self.internal_array_attrs()

        for attr in array_attrs:
            new_data = np.asanyarray(getattr(self, attr))
            ndim = len(np.shape(new_data))
            if ndim > 1:
                new_data = ([attr + f"_dim{i}" for i in range(ndim)], new_data)
            data[attr] = new_data

        ts = Dataset(data)

        ts.attrs.update(self.get_meta_dict())

        return ts

    @classmethod
    def from_xarray(cls: Type[Tso], ts: Dataset) -> Tso:
        """Create a `StingrayObject` from data in an xarray Dataset.

        The dataset MUST contain at least a column named like the
        ``main_array_attr``.
        The rest of columns will form the array attributes of the
        new object, while the attributes in ds.attrs will
        form the new meta attributes of the object.
        """
        cls = cls()

        if len(ts[cls.main_array_attr]) == 0:  # type: ignore
            # return an empty object
            return cls

        # Set the main attribute first
        mainarray = np.array(ts[cls.main_array_attr])  # type: ignore
        setattr(cls, cls.main_array_attr, mainarray)  # type: ignore

        all_array_attrs = []
        for array_attrs in [ts.coords, ts.data_vars]:
            for attr in array_attrs:
                all_array_attrs.append(attr)
                if attr == cls.main_array_attr:  # type: ignore
                    continue
                setattr(cls, attr, np.array(ts[attr]))

        for key, val in ts.attrs.items():
            if key not in all_array_attrs:
                setattr(cls, key, val)

        return cls

    def to_pandas(self) -> DataFrame:
        """Create a pandas ``DataFrame`` from a :class:`StingrayObject`.

        Array attributes (e.g. ``time``, ``pi``, ``energy``, etc. for
        ``EventList``) are converted into columns, while meta attributes
        (``mjdref``, ``gti``, etc.) are saved into the ``ds.attrs`` dictionary.

        Since pandas does not support n-D data, multi-dimensional arrays are
        converted into columns before the conversion, with names ``<colname>_dimN_M_K`` etc.

        See documentation of `make_nd_into_arrays` for details.

        """
        from pandas import DataFrame

        data = {}
        array_attrs = self.array_attrs() + [self.main_array_attr] + self.internal_array_attrs()

        for attr in array_attrs:
            values = np.asanyarray(getattr(self, attr))
            ndim = len(np.shape(values))
            if ndim > 1:
                local_data = make_nd_into_arrays(values, attr)
            else:
                local_data = {attr: values}
            data.update(local_data)

        ts = DataFrame(data)

        ts.attrs.update(self.get_meta_dict())

        return ts

    @classmethod
    def from_pandas(cls: Type[Tso], ts: DataFrame) -> Tso:
        """Create an `StingrayObject` object from data in a pandas DataFrame.

        The dataframe MUST contain at least a column named like the
        ``main_array_attr``.
        The rest of columns will form the array attributes of the
        new object, while the attributes in ds.attrs will
        form the new meta attributes of the object.

        Since pandas does not support n-D data, multi-dimensional arrays can be
        specified as ``<colname>_dimN_M_K`` etc.

        See documentation of `make_1d_arrays_into_nd` for details.

        """
        import re

        cls = cls()

        if len(ts) == 0:
            # return an empty object
            return cls

        array_attrs = ts.columns

        # Set the main attribute first
        mainarray = np.array(ts[cls.main_array_attr])  # type: ignore
        setattr(cls, cls.main_array_attr, mainarray)  # type: ignore

        nd_attrs = []
        for attr in array_attrs:
            if attr == cls.main_array_attr:  # type: ignore
                continue
            if "_dim" in attr:
                nd_attrs.append(re.sub("_dim[0-9].*", "", attr))
            else:
                setattr(cls, attr, np.array(ts[attr]))

        for attr in list(set(nd_attrs)):
            setattr(cls, attr, make_1d_arrays_into_nd(ts, attr))

        for key, val in ts.attrs.items():
            if key not in array_attrs:
                setattr(cls, key, val)

        return cls

    @classmethod
    def read(cls: Type[Tso], filename: str, fmt: str = None) -> Tso:
        r"""Generic reader for :class`StingrayObject`

        Currently supported formats are

        * pickle (not recommended for long-term storage)
        * any other formats compatible with the writers in
          :class:`astropy.table.Table` (ascii.ecsv, hdf5, etc.)

        Files that need the :class:`astropy.table.Table` interface MUST contain
        at least a column named like the ``main_array_attr``.
        The default ascii format is enhanced CSV (ECSV). Data formats
        supporting the serialization of metadata (such as ECSV and HDF5) can
        contain all attributes such as ``mission``, ``gti``, etc with
        no significant loss of information. Other file formats might lose part
        of the metadata, so must be used with care.

        ..note::

            Complex values can be dealt with out-of-the-box in some formats
            like HDF5 or FITS, not in others (e.g. all ASCII formats).
            With these formats, and in any case when fmt is ``None``, complex
            values should be stored as two columns of real numbers, whose names
            are of the format <variablename>.real and <variablename>.imag

        Parameters
        ----------
        filename: str
            Path and file name for the file to be read.

        fmt: str
            Available options are 'pickle', 'hea', and any `Table`-supported
            format such as 'hdf5', 'ascii.ecsv', etc.

        Returns
        -------
        obj: :class:`StingrayObject` object
            The object reconstructed from file
        """

        if fmt is None:
            pass
        elif fmt.lower() == "pickle":
            with open(filename, "rb") as fobj:
                return pickle.load(fobj)
        elif fmt.lower() == "ascii":
            fmt = "ascii.ecsv"

        ts = convert_table_attrs_to_lowercase(Table.read(filename, format=fmt))

        # For specific formats, and in any case when the format is not
        # specified, make sure that complex values are treated correctly.
        if fmt is None or "ascii" in fmt:
            for col in ts.colnames:
                if not ((is_real := col.endswith(".real")) or (is_imag := col.endswith(".imag"))):
                    continue

                new_value = ts[col]

                if is_imag:
                    new_value = new_value * 1j

                # Make sure it's complex, even if we find the real part first
                new_value = new_value + 0.0j

                col_strip = col.replace(".real", "").replace(".imag", "")

                if col_strip not in ts.colnames:
                    # If the column without ".real" or ".imag" doesn't exist,
                    # define it, and make sure it's complex-valued
                    ts[col_strip] = new_value
                else:
                    # If it does exist, sum the new value to it.
                    ts[col_strip] += new_value

                ts.remove_column(col)

        return cls.from_astropy_table(ts)

    def write(self, filename: str, fmt: str = None) -> None:
        """Generic writer for :class`StingrayObject`

        Currently supported formats are

        * pickle (not recommended for long-term storage)
        * any other formats compatible with the writers in
          :class:`astropy.table.Table` (ascii.ecsv, hdf5, etc.)

        ..note::

            Complex values can be dealt with out-of-the-box in some formats
            like HDF5 or FITS, not in others (e.g. all ASCII formats).
            With these formats, and in any case when fmt is ``None``, complex
            values will be stored as two columns of real numbers, whose names
            are of the format <variablename>.real and <variablename>.imag

        Parameters
        ----------
        filename: str
            Name and path of the file to save the object list to.

        fmt: str
            The file format to store the data in.
            Available options are ``pickle``, ``hdf5``, ``ascii``, ``fits``
        """
        if fmt is None:
            pass
        elif fmt.lower() == "pickle":
            with open(filename, "wb") as fobj:
                pickle.dump(self, fobj)
            return
        elif fmt.lower() == "ascii":
            fmt = "ascii.ecsv"

        probe_file = "probe.bu.bu." + filename[-7:]

        CAN_SAVE_LONGD = _can_save_longdouble(probe_file, fmt)
        CAN_SERIALIZE_META = _can_serialize_meta(probe_file, fmt)

        to_be_saved = self

        ts = to_be_saved.to_astropy_table(no_longdouble=not CAN_SAVE_LONGD)

        if fmt is None or "ascii" in fmt:
            for col in ts.colnames:
                if "complex" in str(ts[col][0].dtype):
                    ts[f"{col}.real"] = ts[col].real
                    ts[f"{col}.imag"] = ts[col].imag
                    ts.remove_column(col)

        if CAN_SERIALIZE_META:
            ts.write(filename, format=fmt, overwrite=True, serialize_meta=True)
        else:
            ts.write(filename, format=fmt, overwrite=True)

    def apply_mask(self, mask: npt.ArrayLike, inplace: bool = False, filtered_attrs: list = None):
        """Apply a mask to all array attributes of the object

        Parameters
        ----------
        mask : array of ``bool``
            The mask. Has to be of the same length as ``self.time``

        Other parameters
        ----------------
        inplace : bool
            If True, overwrite the current object. Otherwise, return a new one.
        filtered_attrs : list of str or None
            Array attributes to be filtered. Defaults to all array attributes if ``None``.
            The other array attributes will be set to ``None``. The main array attr is always
            included.

        Returns
        -------
        ts_new : StingrayObject object
            The new object with the mask applied if ``inplace`` is ``False``, otherwise the
            same object.
        """
        all_attrs = self.internal_array_attrs() + self.array_attrs()
        if filtered_attrs is None:
            filtered_attrs = all_attrs

        if inplace:
            new_ts = self
        else:
            new_ts = type(self)()
            for attr in self.meta_attrs():
                setattr(new_ts, attr, copy.deepcopy(getattr(self, attr)))

        # If the main array attr is managed through an internal attr
        # (e.g. lightcurve), set the internal attr instead.
        if hasattr(self, "_" + self.main_array_attr):
            setattr(
                new_ts,
                "_" + self.main_array_attr,
                copy.deepcopy(np.asanyarray(getattr(self, self.main_array_attr))[mask]),
            )
        else:
            setattr(
                new_ts,
                self.main_array_attr,
                copy.deepcopy(np.asanyarray(getattr(self, self.main_array_attr))[mask]),
            )

        for attr in all_attrs:
            if attr not in filtered_attrs:
                # Eliminate all unfiltered attributes
                setattr(new_ts, attr, None)
            else:
                setattr(new_ts, attr, copy.deepcopy(np.asanyarray(getattr(self, attr))[mask]))
        return new_ts

    def _operation_with_other_obj(
        self,
        other,
        operation,
        operated_attrs=None,
        error_attrs=None,
        error_operation=None,
        inplace=False,
    ):
        """
        Helper method to codify an operation of one time series with another (e.g. add, subtract).
        Takes into account the GTIs, and returns a new :class:`StingrayTimeseries` object.

        Parameters
        ----------
        other : :class:`StingrayTimeseries` object
            A second time series object

        operation : function
            An operation between the :class:`StingrayTimeseries` object calling this method, and
            ``other``, operating on all the specified array attributes.

        Other parameters
        ----------------
        operated_attrs : list of str or None
            Array attributes to be operated on. Defaults to all array attributes not ending in
            ``_err``.
            The other array attributes will be discarded from the time series to avoid
            inconsistencies.

        error_attrs : list of str or None
            Array attributes to be operated on with ``error_operation``. Defaults to all array
            attributes ending with ``_err``.

        error_operation : function
            The function used for error propagation. Defaults to the sum of squares.

        Returns
        -------
        ts_new : StingrayTimeseries object
            The new time series calculated in ``operation``
        """

        if operated_attrs is None:
            operated_attrs = self._default_operated_attrs()

        if error_attrs is None:
            error_attrs = self._default_error_attrs()

        if not isinstance(other, type(self)):
            raise TypeError(
                f"{type(self)} objects can only be operated with other {type(self)} objects."
            )

        this_time = getattr(self, self.main_array_attr)
        # ValueError is raised by Numpy while asserting np.equal over arrays
        # with different dimensions.
        try:
            assert np.array_equal(this_time, getattr(other, self.main_array_attr))
        except (ValueError, AssertionError):
            raise ValueError(
                f"The values of {self.main_array_attr} are different in the two {type(self)} "
                "objects."
            )

        if inplace:
            ts_new = self
        else:
            ts_new = type(self)()
        setattr(ts_new, self.main_array_attr, this_time)
        for attr in self.meta_attrs():
            setattr(ts_new, attr, copy.deepcopy(getattr(self, attr)))

        for attr in operated_attrs:
            setattr(
                ts_new,
                attr,
                operation(getattr(self, attr), getattr(other, attr)),
            )

        for attr in error_attrs:
            setattr(
                ts_new,
                attr,
                error_operation(getattr(self, attr), getattr(other, attr)),
            )

        return ts_new

    def add(
        self, other, operated_attrs=None, error_attrs=None, error_operation=sqsum, inplace=False
    ):
        """Add two :class:`StingrayObject` instances.

        Add the array values of two :class:`StingrayObject` instances element by element, assuming
        the main array attributes of the instances match exactly.

        All array attrs ending with ``_err`` are treated as error bars and propagated with the
        sum of squares.

        GTIs are crossed, so that only common intervals are saved.

        Parameters
        ----------
        other : :class:`StingrayTimeseries` object
            A second time series object

        Other parameters
        ----------------
        operated_attrs : list of str or None
            Array attributes to be operated on. Defaults to all array attributes not ending in
            ``_err``.
            The other array attributes will be discarded from the time series to avoid
            inconsistencies.
        error_attrs : list of str or None
            Array attributes to be operated on with ``error_operation``. Defaults to all array
            attributes ending with ``_err``.
        error_operation : function
            Function to be called to propagate the errors
        inplace : bool
            If True, overwrite the current time series. Otherwise, return a new one.
        """
        return self._operation_with_other_obj(
            other,
            np.add,
            operated_attrs=operated_attrs,
            error_attrs=error_attrs,
            error_operation=error_operation,
            inplace=inplace,
        )

    def __add__(self, other):
        """Operation that gets called with the ``+`` operator.

        Add the array values of two :class:`StingrayObject` instances element by element, assuming
        the main array attributes of the instances match exactly.

        All array attrs ending with ``_err`` are treated as error bars and propagated with the
        sum of squares.

        GTIs are crossed, so that only common intervals are saved.
        """

        return self._operation_with_other_obj(
            other,
            np.add,
            error_operation=sqsum,
        )

    def __iadd__(self, other):
        """Operation that gets called with the ``+=`` operator.

        Add the array values of two :class:`StingrayObject` instances element by element, assuming
        the main array attributes of the instances match exactly.

        All array attrs ending with ``_err`` are treated as error bars and propagated with the
        sum of squares.

        GTIs are crossed, so that only common intervals are saved.
        """

        return self._operation_with_other_obj(
            other,
            np.add,
            error_operation=sqsum,
            inplace=True,
        )

    def sub(
        self, other, operated_attrs=None, error_attrs=None, error_operation=sqsum, inplace=False
    ):
        """
        Subtract *all the array attrs* of two :class:`StingrayObject` instances element by element, assuming the main array attributes of the instances match exactly.

        All array attrs ending with ``_err`` are treated as error bars and propagated with the
        sum of squares.

        GTIs are crossed, so that only common intervals are saved.

        Parameters
        ----------
        other : :class:`StingrayTimeseries` object
            A second time series object

        Other parameters
        ----------------
        operated_attrs : list of str or None
            Array attributes to be operated on. Defaults to all array attributes not ending in
            ``_err``.
            The other array attributes will be discarded from the time series to avoid
            inconsistencies.
        error_attrs : list of str or None
            Array attributes to be operated on with ``error_operation``. Defaults to all array
            attributes ending with ``_err``.
        error_operation : function
            Function to be called to propagate the errors
        inplace : bool
            If True, overwrite the current time series. Otherwise, return a new one.
        """
        return self._operation_with_other_obj(
            other,
            np.subtract,
            operated_attrs=operated_attrs,
            error_attrs=error_attrs,
            error_operation=error_operation,
            inplace=inplace,
        )

    def __sub__(self, other):
        """Operation that gets called with the ``-`` operator.

        Subtract *all the array attrs* of two :class:`StingrayObject` instances element by element, assuming the main array attributes of the instances match exactly.

        All array attrs ending with ``_err`` are treated as error bars and propagated with the
        sum of squares.

        GTIs are crossed, so that only common intervals are saved.
        """

        return self._operation_with_other_obj(
            other,
            np.subtract,
            error_operation=sqsum,
        )

    def __isub__(self, other):
        """Operation that gets called with the ``-=`` operator.

        Subtract *all the array attrs* of two :class:`StingrayObject` instances element by element, assuming the main array attributes of the instances match exactly.

        All array attrs ending with ``_err`` are treated as error bars and propagated with the
        sum of squares.

        GTIs are crossed, so that only common intervals are saved.
        """

        return self._operation_with_other_obj(
            other,
            np.subtract,
            error_operation=sqsum,
            inplace=True,
        )

    def __neg__(self):
        """
        Implement the behavior of negation of the array attributes of a :class:`StingrayObject`
        Error attrs are left alone.

        The negation operator ``-`` is supposed to invert the sign of all array attributes of a
        time series object, leaving out the ones ending with ``_err``.

        """

        ts_new = copy.deepcopy(self)
        for attr in self._default_operated_attrs():
            setattr(ts_new, attr, -np.asanyarray(getattr(self, attr)))

        return ts_new

    def __len__(self):
        """
        Return the number of bins of a the main array attributes

        This method overrides the ``len`` function for a :class:`StingrayObject`
        object and returns the length of the array attributes (using the main array attribute
        as probe).
        """
        return np.size(getattr(self, self.main_array_attr))

    def __getitem__(self, index):
        """
        Return an element or a slice of the :class:`StingrayObject`.

        Parameters
        ----------
        index : int or slice instance
            Index value of the time array or a slice object.

        Returns
        -------
        ts_new : :class:`StingrayObject` object
            The new :class:`StingrayObject` object with the set of selected data.
        """

        if isinstance(index, (int, np.integer)):
            start = index
            stop = index + 1
            step = 1
        elif isinstance(index, slice):
            start = assign_value_if_none(index.start, 0)
            stop = assign_value_if_none(index.stop, len(self))
            step = assign_value_if_none(index.step, 1)
        else:
            raise IndexError("The index must be either an integer or a slice " "object !")

        new_ts = type(self)()

        for attr in self.meta_attrs():
            setattr(new_ts, attr, copy.deepcopy(getattr(self, attr)))

        for attr in self.array_attrs() + [self.main_array_attr]:
            setattr(new_ts, attr, getattr(self, attr)[start:stop:step])

        return new_ts


def _ts_sum(ts):
    """Sum the number of values of a time series object.

    If it has a ``counts`` attribute, sum the counts. Otherwise, count the number
    of time samples. Masked values are ignored.
    """
    if hasattr(ts, "counts"):
        return np.sum(ts.counts[ts.mask])
    return np.count_nonzero(ts.mask)


class StingrayTimeseries(StingrayObject):
    """Basic class for time series data.

    This can be events, binned light curves, unevenly sampled light curves, etc. The only
    requirement is that the data (which can be any quantity, related or not to an electromagnetic
    measurement) are associated with a time measurement.
    We make a distinction between the *array* attributes, which have the same length of the
    ``time`` array, and the *meta* attributes, which can be scalars or arrays of different
    size. The array attributes can be multidimensional (e.g. a spectrum for each time bin),
    but their first dimension (``array.shape[0]``) must have same length of the ``time`` array.

    Array attributes are singled out automatically depending on their shape. All filtering
    operations (e.g. ``apply_gtis``, ``rebin``, etc.) are applied to array attributes only.
    For this reason, it is advisable to specify whether a given attribute should *not* be
    considered as an array attribute by adding it to the ``not_array_attr`` list.

    Parameters
    ----------
    time: iterable
        A list or array of time stamps

    Other Parameters
    ----------------
    array_attrs : dict
        Array attributes to be set (e.g. ``{"flux": flux_array, "flux_err": flux_err_array}``).
        In principle, they could be specified as simple keyword arguments. But this way, we
        will run a check on the length of the arrays, and raise an error if they are not of a
        shape compatible with the ``time`` array.

    dt: float
        The time resolution of the time series. Can be a scalar or an array attribute (useful
        for non-evenly sampled data or events from different instruments)

    mjdref : float
        The MJD used as a reference for the time array.

    gtis: ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]``
        Good Time Intervals

    high_precision : bool
        Change the precision of self.time to float128. Useful while dealing with fast pulsars.

    timeref : str
        The time reference, as recorded in the FITS file (e.g. SOLARSYSTEM)

    timesys : str
        The time system, as recorded in the FITS file (e.g. TDB)

    ephem : str
        The JPL ephemeris used to barycenter the data, if any (e.g. DE430)

    skip_checks : bool
        Skip checks on the time array. Useful when the user is reasonably sure that the
        input data are valid.

    **other_kw :
        Used internally. Any other keyword arguments will be set as attributes of the object.

    Attributes
    ----------
    time: numpy.ndarray
        The array of time stamps, in seconds from the reference
        MJD defined in ``mjdref``

    not_array_attr: list
        List of attributes that are never to be considered as array attributes. For example, GTIs
        are not array attributes.

    dt: float
        The time resolution of the measurements. Can be a scalar or an array attribute (useful
        for non-evenly sampled data or events from different instruments). It can also be 0, which
        means that the time series is not evenly sampled and the effects of the time resolution are
        considered negligible for the analysis. This is sometimes the case for events from
        high-energy telescopes.

    mjdref : float
        The MJD used as a reference for the time array.

    gtis: ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]``
        Good Time Intervals

    high_precision : bool
        Change the precision of self.time to float128. Useful while dealing with fast pulsars.

    """

    main_array_attr: str = "time"
    not_array_attr: list = ["gti"]
    _time: TTime = None
    high_precision: bool = False
    mjdref: TTime = 0.0
    dt: float = 0.0

    def __init__(
        self,
        time: TTime = None,
        array_attrs: dict = {},
        mjdref: TTime = 0,
        notes: str = "",
        gti: npt.ArrayLike = None,
        high_precision: bool = False,
        ephem: str = None,
        timeref: str = None,
        timesys: str = None,
        skip_checks: bool = False,
        **other_kw,
    ):
        StingrayObject.__init__(self)

        self.notes = notes
        self.mjdref = mjdref
        self.gti = gti
        self.ephem = ephem
        self.timeref = timeref
        self.timesys = timesys
        self._mask = None
        self.high_precision = high_precision
        self.dt = other_kw.pop("dt", 0)

        self._set_times(time, high_precision=high_precision)
        for kw in other_kw:
            setattr(self, kw, other_kw[kw])
        for kw in array_attrs:
            new_arr = np.asanyarray(array_attrs[kw])
            if self.time.shape[0] != new_arr.shape[0]:
                raise ValueError(f"Lengths of time and {kw} must be equal.")
            setattr(self, kw, new_arr)
        from .utils import is_sorted

        if not skip_checks:
            if self.time is not None and not is_sorted(self.time):
                warnings.warn("The time array is not sorted. Sorting it now.")
                self.sort(inplace=True)

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, value):
        value = self._validate_and_format(value, "time", "time")
        if value is None:
            for attr in self.internal_array_attrs() + self.array_attrs():
                setattr(self, attr, None)
        self._set_times(value, high_precision=self.high_precision)

    @property
    def gti(self):
        if self._gti is None and self._time is not None:
            if isinstance(self.dt, Iterable):
                dt0 = self.dt[0]
                dt1 = self.dt[-1]
            else:
                dt0 = dt1 = self.dt
            self._gti = np.asanyarray([[self._time[0] - dt0 / 2, self._time[-1] + dt1 / 2]])
        return self._gti

    @gti.setter
    def gti(self, value):
        if value is None:
            self._gti = None
            return
        value = np.asanyarray(value)
        self._gti = value
        self._mask = None

    @property
    def mask(self):
        if self._mask is not None:
            return self._mask
        if self._gti is not None:
            self._mask = create_gti_mask(self.time, self._gti, dt=self.dt)
        else:
            self._mask = np.ones_like(self.time, dtype=bool)
        return self._mask

    @property
    def n(self):
        return self.main_array_length

    def _set_times(self, time, high_precision=False):
        if time is None or np.size(time) == 0:
            self._time = None
            return
        time, _ = interpret_times(time, self.mjdref)
        if not high_precision:
            self._time = np.asanyarray(time)
        else:
            self._time = np.asanyarray(time, dtype=np.longdouble)

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return self.pretty_print(
            attrs_to_apply=["gti", "time", "tstart", "tseg", "tstop"],
            func_to_apply=lambda x: (np.asanyarray(x) / 86400 + self.mjdref, "MJD"),
            attrs_to_discard=["_mask", "header"],
        )

    def _validate_and_format(self, value, attr_name, compare_to_attr):
        """Check if the size of a value is compatible with the size of another attribute.

        Different cases are possible:

        - If the value is None, we return None
        - If the value is a scalar, we fail
        - If the value is an array, we check if it has the correct shape by comparing it with
          another attribute. In the special case where the attribute is the same, if it is None
          we assign the new value. Otherwise, the first dimension of the value and the current
          value of the attribute being compared with has to be the same.

        Parameters
        ----------
        value : array-like or None
            The value to check.
        attr_name : str
            The name of the attribute being checked.
        compare_to_attr : str
            The name of the attribute to compare with.

        Returns
        -------
        value : array-like or None
            The value to check wrapped in a class:`np.array`, if it is not None. Otherwise None
        """
        if value is None:
            return None
        value = np.asanyarray(value)
        if len(value.shape) < 1:
            raise ValueError(f"{attr_name} array must be at least 1D")
        # If the attribute we compare it with is the same and it is currently None, we assign it
        # This can happen, e.g. with the time array.
        compare_with = getattr(self, compare_to_attr, None)
        if attr_name == compare_to_attr and compare_with is None:
            return value

        # In the special case where the current value of the attribute being compared
        # is None, this also has to fail.
        if compare_with is None:
            raise ValueError(
                f"Can only assign new {attr_name} if the {compare_to_attr} array is not None"
            )
        if value.shape[0] != compare_with.shape[0]:
            raise ValueError(
                f"Can only assign new {attr_name} of the same shape as the {compare_to_attr} array"
            )
        return value

    @property
    def exposure(self):
        """
        Return the total exposure of the time series, i.e. the sum of the GTIs.

        Returns
        -------
        total_exposure : float
            The total exposure of the time series, in seconds.
        """

        return get_total_gti_length(self.gti)

    def __eq__(self, other_ts):
        return super().__eq__(other_ts)

    def apply_gtis(self, new_gti=None, inplace: bool = True):
        """
        Apply Good Time Intervals (GTIs) to a time series. Filters all the array attributes, only
        keeping the bins that fall into GTIs.

        Parameters
        ----------
        inplace : bool
            If True, overwrite the current time series. Otherwise, return a new one.

        """
        # I import here to avoid the risk of circular imports

        if new_gti is None:
            new_gti = self.gti

        check_gtis(new_gti)

        # This will automatically be recreated from GTIs once I set it to None
        good = create_gti_mask(self.time, new_gti, dt=self.dt)
        newts = self.apply_mask(good, inplace=inplace)
        # Important, otherwise addition/subtraction ops will go into an infinite loop
        if inplace:
            newts.gti = new_gti
        return newts

    def split_by_gti(self, gti=None, min_points=2):
        """
        Split the current :class:`StingrayTimeseries` object into a list of
        :class:`StingrayTimeseries` objects, one for each continuous GTI segment
        as defined in the ``gti`` attribute.

        Parameters
        ----------
        min_points : int, default 1
            The minimum number of data points in each time series. Light
            curves with fewer data points will be ignored.

        Returns
        -------
        list_of_tss : list
            A list of :class:`StingrayTimeseries` objects, one for each GTI segment
        """

        if gti is None:
            gti = self.gti

        list_of_tss = []

        start_bins, stop_bins = gti_border_bins(gti, self.time, self.dt)
        for i in range(len(start_bins)):
            start = start_bins[i]
            stop = stop_bins[i]

            if (stop - start) < min_points:
                continue

            new_gti = np.array([gti[i]])
            mask = create_gti_mask(self.time, new_gti)

            # Note: GTIs are consistent with default in this case!
            new_ts = self.apply_mask(mask)
            new_ts.gti = new_gti

            list_of_tss.append(new_ts)

        return list_of_tss

    def to_astropy_timeseries(self) -> TimeSeries:
        """Save the ``StingrayTimeseries`` to an ``Astropy`` timeseries.

        Array attributes (time, pi, energy, etc.) are converted
        into columns, while meta attributes (mjdref, gti, etc.)
        are saved into the ``meta`` dictionary.

        Returns
        -------
        ts : `astropy.timeseries.TimeSeries`
            A ``TimeSeries`` object with the array attributes as columns,
            and the meta attributes in the `meta` dictionary
        """
        from astropy.timeseries import TimeSeries
        from astropy.time import TimeDelta
        from astropy import units as u

        data = {}
        array_attrs = self.array_attrs()

        for attr in array_attrs:
            if attr == "time":
                continue
            data[attr] = np.asanyarray(getattr(self, attr))

        if data == {}:
            data = None

        if self.time is not None and np.size(self.time) > 0:  # type: ignore
            times = TimeDelta(self.time * u.s)  # type: ignore
            ts = TimeSeries(data=data, time=times)
        else:
            ts = TimeSeries()

        ts.meta.update(self.get_meta_dict())

        return ts

    @classmethod
    def from_astropy_timeseries(cls, ts: TimeSeries) -> StingrayTimeseries:
        """Create a `StingrayTimeseries` from data in an Astropy TimeSeries

        The timeseries has to define at least a column called time,
        the rest of columns will form the array attributes of the
        new time series, while the attributes in table.meta will
        form the new meta attributes of the time series.

        Parameters
        ----------
        ts : `astropy.timeseries.TimeSeries`
            A ``TimeSeries`` object with the array attributes as columns,
            and the meta attributes in the `meta` dictionary

        Returns
        -------
        ts : `StingrayTimeseries`
            Timeseries object
        """

        time = ts["time"]
        mjdref = None
        if "mjdref" in ts.meta:
            mjdref = ts.meta["mjdref"]

        new_cls = cls()
        time, mjdref = interpret_times(time, mjdref)
        new_cls.time = np.asanyarray(time)  # type: ignore

        array_attrs = ts.colnames
        for key, val in ts.meta.items():
            setattr(new_cls, key, val)

        for attr in array_attrs:
            if attr == "time":
                continue
            setattr(new_cls, attr, np.asanyarray(ts[attr]))

        return new_cls

    def change_mjdref(self, new_mjdref: float, inplace=False) -> StingrayTimeseries:
        """Change the MJD reference time (MJDREF) of the time series

        The times of the time series will be shifted in order to be referred to
        this new MJDREF

        Parameters
        ----------
        new_mjdref : float
            New MJDREF

        Other parameters
        ----------------
        inplace : bool
            If True, overwrite the current time series. Otherwise, return a new one.

        Returns
        -------
        new_ts : :class:`StingrayTimeseries` object
            The new time series, shifted by MJDREF
        """
        time_shift = (self.mjdref - new_mjdref) * 86400  # type: ignore

        ts = self.shift(time_shift, inplace=inplace)
        ts.mjdref = new_mjdref  # type: ignore
        return ts

    def shift(self, time_shift: float, inplace=False) -> StingrayTimeseries:
        """Shift the time and the GTIs by the same amount

        Parameters
        ----------
        time_shift: float
            The time interval by which the time series will be shifted (in
            the same units as the time array in :class:`StingrayTimeseries`

        Other parameters
        ----------------
        inplace : bool
            If True, overwrite the current time series. Otherwise, return a new one.

        Returns
        -------
        ts : ``StingrayTimeseries`` object
            The new time series shifted by ``time_shift``

        """
        if inplace:
            ts = self
        else:
            ts = copy.deepcopy(self)
        ts.time = np.asanyarray(ts.time) + time_shift  # type: ignore
        # Pay attention here: if the GTIs are created dynamically while we
        # access the property,
        if ts._gti is not None:
            ts._gti = np.asanyarray(ts._gti) + time_shift  # type: ignore

        return ts

    def _operation_with_other_obj(
        self, other, operation, operated_attrs=None, error_attrs=None, error_operation=None
    ):
        """
        Helper method to codify an operation of one time series with another (e.g. add, subtract).
        Takes into account the GTIs correctly, and returns a new :class:`StingrayTimeseries` object.

        Parameters
        ----------
        other : :class:`StingrayTimeseries` object
            A second time series object

        operation : function
            An operation between the :class:`StingrayTimeseries` object calling this method, and
            ``other``, operating on all the specified array attributes.

        Other parameters
        ----------------
        operated_attrs : list of str or None
            Array attributes to be operated on. Defaults to all array attributes not ending in
            ``_err``.
            The other array attributes will be discarded from the time series to avoid
            inconsistencies.

        error_attrs : list of str or None
            Array attributes to be operated on with ``error_operation``. Defaults to all array
            attributes ending with ``_err``.

        error_operation : function
            The function used for error propagation. Defaults to the sum of squares.

        Returns
        -------
        ts_new : StingrayTimeseries object
            The new time series calculated in ``operation``
        """

        if self.mjdref != other.mjdref:
            warnings.warn("MJDref is different in the two time series")
            other = other.change_mjdref(self.mjdref)

        if not np.array_equal(self.gti, other.gti):
            warnings.warn(
                "The good time intervals in the two time series are different. Data outside the "
                "common GTIs will be discarded."
            )
            common_gti = cross_two_gtis(self.gti, other.gti)
            masked_self = self.apply_gtis(common_gti)
            masked_other = other.apply_gtis(common_gti)
            return masked_self._operation_with_other_obj(
                masked_other,
                operation,
                operated_attrs=operated_attrs,
                error_attrs=error_attrs,
                error_operation=error_operation,
            )

        return super()._operation_with_other_obj(
            other,
            operation,
            operated_attrs=operated_attrs,
            error_attrs=error_attrs,
            error_operation=error_operation,
        )

    def __add__(self, other):
        """
        Add the array values of two time series element by element, assuming they
        have the same time array.

        This magic method adds two :class:`TimeSeries` objects having the same time
        array such that the corresponding array arrays get summed up.

        GTIs are crossed, so that only common intervals are saved.

        Examples
        --------
        >>> time = [5, 10, 15]
        >>> count1 = [300, 100, 400]
        >>> count2 = [600, 1200, 800]
        >>> gti1 = [[0, 25]]
        >>> gti2 = [[0, 25]]
        >>> ts1 = StingrayTimeseries(time, array_attrs=dict(counts=count1), gti=gti1, dt=5)
        >>> ts2 = StingrayTimeseries(time, array_attrs=dict(counts=count2), gti=gti2, dt=5)
        >>> ts = ts1 + ts2
        >>> assert np.allclose(ts.counts, [ 900, 1300, 1200])
        """

        return super().__add__(other)

    def __sub__(self, other):
        """
        Subtract the counts/flux of one time series from the counts/flux of another
        time series element by element, assuming the ``time`` arrays of the time series
        match exactly.

        This magic method adds two :class:`StingrayTimeSeries` objects having the same
        ``time`` array and subtracts the ``counts`` of one :class:`StingrayTimeseries` with
        that of another, while also updating ``countrate``, ``counts_err`` and ``countrate_err``
        correctly.

        GTIs are crossed, so that only common intervals are saved.

        Examples
        --------
        >>> time = [10, 20, 30]
        >>> count1 = [600, 1200, 800]
        >>> count2 = [300, 100, 400]
        >>> gti1 = [[0, 35]]
        >>> gti2 = [[0, 35]]
        >>> ts1 = StingrayTimeseries(time, array_attrs=dict(counts=count1), gti=gti1, dt=10)
        >>> ts2 = StingrayTimeseries(time, array_attrs=dict(counts=count2), gti=gti2, dt=10)
        >>> ts = ts1 - ts2
        >>> assert np.allclose(ts.counts, [ 300, 1100,  400])
        """

        return super().__sub__(other)

    def __getitem__(self, index):
        """
        Return the corresponding count value at the index or a new :class:`StingrayTimeseries`
        object upon slicing.

        This method adds functionality to retrieve the count value at
        a particular index. This also can be used for slicing and generating
        a new :class:`StingrayTimeseries` object. GTIs are recalculated based on the new light
        curve segment

        If the slice object is of kind ``start:stop:step`` and ``dt`` is not 0, GTIs are also
        sliced, by crossing with ``zip(time - self.dt /2, time + self.dt / 2)``

        Parameters
        ----------
        index : int or slice instance
            Index value of the time array or a slice object.

        Examples
        --------
        >>> time = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> count = [11, 22, 33, 44, 55, 66, 77, 88, 99]
        >>> ts = StingrayTimeseries(time, array_attrs=dict(counts=count), dt=1)
        >>> assert np.allclose(ts[2].counts, [33])
        >>> assert np.allclose(ts[:2].counts, [11, 22])
        """

        new_ts = super().__getitem__(index)
        step = 1
        if isinstance(index, slice):
            step = assign_value_if_none(index.step, 1)

        dt = self.dt
        if np.isscalar(dt):
            delta_gti_start = delta_gti_stop = dt * 0.5
        else:
            delta_gti_start = new_ts.dt[0] * 0.5
            delta_gti_stop = new_ts.dt[-1] * 0.5

        new_gti = np.asanyarray(
            [[new_ts.time[0] - delta_gti_start, new_ts.time[-1] + delta_gti_stop]]
        )
        if step > 1 and delta_gti_start > 0:
            new_gt1 = np.array(list(zip(new_ts.time - new_ts.dt / 2, new_ts.time + new_ts.dt / 2)))
            new_gti = cross_two_gtis(new_gti, new_gt1)
        new_gti = cross_two_gtis(self.gti, new_gti)

        new_ts.gti = new_gti
        return new_ts

    def truncate(self, start=0, stop=None, method="index"):
        """
        Truncate a :class:`StingrayTimeseries` object.

        This method takes a ``start`` and a ``stop`` point (either as indices,
        or as times in the same unit as those in the ``time`` attribute, and truncates
        all bins before ``start`` and after ``stop``, then returns a new
        :class:`StingrayTimeseries` object with the truncated time series.

        Parameters
        ----------
        start : int, default 0
            Index (or time stamp) of the starting point of the truncation. If no value is set
            for the start point, then all points from the first element in the ``time`` array
            are taken into account.

        stop : int, default ``None``
            Index (or time stamp) of the ending point (exclusive) of the truncation. If no
            value of stop is set, then points including the last point in
            the counts array are taken in count.

        method : {``index`` | ``time``}, optional, default ``index``
            Type of the start and stop values. If set to ``index`` then
            the values are treated as indices of the counts array, or
            if set to ``time``, the values are treated as actual time values.

        Returns
        -------
        ts_new: :class:`StingrayTimeseries` object
            The :class:`StingrayTimeseries` object with truncated time and arrays.

        Examples
        --------
        >>> time = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> count = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        >>> ts = StingrayTimeseries(time, array_attrs={"counts": count}, dt=1)
        >>> ts_new = ts.truncate(start=2, stop=8)
        >>> assert np.allclose(ts_new.counts, [30, 40, 50, 60, 70, 80])
        >>> assert np.allclose(ts_new.time, [3, 4, 5, 6, 7, 8])
        >>> # Truncation can also be done by time values
        >>> ts_new = ts.truncate(start=6, method='time')
        >>> assert np.allclose(ts_new.time, [6, 7, 8, 9])
        >>> assert np.allclose(ts_new.counts, [60, 70, 80, 90])
        """

        if not isinstance(method, str):
            raise TypeError("The method keyword argument is not a string !")

        if method.lower() not in ["index", "time"]:
            raise ValueError("Unknown method type " + method + ".")

        if method.lower() == "index":
            new_ts = self._truncate_by_index(start, stop)
        else:
            new_ts = self._truncate_by_time(start, stop)
        new_ts.tstart = new_ts.gti[0, 0]
        new_ts.tseg = new_ts.gti[-1, 1] - new_ts.gti[0, 0]
        return new_ts

    def _truncate_by_index(self, start, stop):
        """Private method for truncation using index values."""

        new_ts = self.apply_mask(slice(start, stop))

        dtstart = dtstop = new_ts.dt
        if isinstance(self.dt, Iterable):
            dtstart = self.dt[0]
            dtstop = self.dt[-1]

        gti = cross_two_gtis(
            self.gti,
            np.asanyarray([[new_ts.time[0] - 0.5 * dtstart, new_ts.time[-1] + 0.5 * dtstop]]),
        )

        new_ts.gti = gti

        return new_ts

    def _truncate_by_time(self, start, stop):
        """Helper method for truncation using time values.

        Parameters
        ----------
        start : float
            start time for new time series; all time bins before this time will be discarded

        stop : float
            stop time for new time series; all time bins after this point will be discarded

        Returns
        -------
            new_ts : StingrayTimeseries
                A new :class:`StingrayTimeseries` object with the truncated time bins

        """

        if stop is not None:
            if start > stop:
                raise ValueError("start time must be less than stop time!")

        if not start == 0:
            start = self.time.searchsorted(start)

        if stop is not None:
            stop = self.time.searchsorted(stop)

        return self._truncate_by_index(start, stop)

    def concatenate(self, other, check_gti=True):
        """
        Concatenate two :class:`StingrayTimeseries` objects.

        This method concatenates two or more :class:`StingrayTimeseries` objects along the time
        axis. GTIs are recalculated by merging all the GTIs together. GTIs should not overlap at
        any point.

        Parameters
        ----------
        other : :class:`StingrayTimeseries` object or list of :class:`StingrayTimeseries` objects
            A second time series object, or a list of objects to be concatenated

        Other parameters
        ----------------
        check_gti : bool
            Check if the GTIs are overlapping or not. Default: True
            If this is True and GTIs overlap, an error is raised.
        """

        if check_gti:
            treatment = "append"
        else:
            treatment = "none"
        new_ts = self._join_timeseries(other, strategy=treatment)
        return new_ts

    def _join_timeseries(self, others, strategy="intersection", ignore_meta=[]):
        """Helper method to join two or more :class:`StingrayTimeseries` objects.

        This is a helper method that can be called by other user-facing methods, such as
        :class:`StingrayTimeseries().join()`.

        Standard attributes such as ``pi`` and ``energy`` remain ``None`` if they are ``None``
        in both. Otherwise, ``np.nan`` is used as a default value for the missing values.
        Arbitrary array attributes are created and joined using the same convention.

        Multiple checks are done on the joined time series. If the time array of the series
        being joined is empty, it is ignored (and a copy of the original time series is returned
        instead). If the time resolution is different, the final time series will associate
        different time resolutions to different time bins.
        If the MJDREF is different (including being 0), the time reference will be changed to
        the one of the first time series. An empty time series will be ignored.

        Parameters
        ----------
        other : :class:`StingrayTimeseries` or class:`list` of :class:`StingrayTimeseries`
            The other :class:`StingrayTimeseries` object which is supposed to be joined with.
            If ``other`` is a list, it is assumed to be a list of :class:`StingrayTimeseries`
            and they are all joined, one by one.

        Other parameters
        ----------------
        strategy : {"intersection", "union", "append", "infer", "none"}
            Method to use to merge the GTIs. If "intersection", the GTIs are merged
            using the intersection of the GTIs. If "union", the GTIs are merged
            using the union of the GTIs. If "none", a single GTI with the minimum and
            the maximum time stamps of all GTIs is returned. If "infer", the strategy
            is decided based on the GTIs. If there are no overlaps, "union" is used,
            otherwise "intersection" is used. If "append", the GTIs are simply appended
            but they must be mutually exclusive.

        Returns
        -------
        `ts_new` : :class:`StingrayTimeseries` object
            The resulting :class:`StingrayTimeseries` object.
        """

        new_ts = type(self)()

        if not (
            isinstance(others, Iterable)
            and not isinstance(others, str)
            and not isinstance(others, StingrayObject)
        ):
            others = [others]
        else:
            others = others

        # First of all, check if there are empty objects
        for obj in others:
            if not isinstance(obj, type(self)):
                raise TypeError(
                    f"{type(self)} objects can only be merged with other {type(self)} objects."
                )
            if getattr(obj, "time", None) is None or np.size(obj.time) == 0:
                warnings.warn("One of the time series you are joining is empty.")
                others.remove(obj)

        if len(others) == 0:
            return copy.deepcopy(self)

        for i, other in enumerate(others):
            # Tolerance for MJDREF:1 microsecond
            if not np.isclose(self.mjdref, other.mjdref, atol=1e-6 / 86400):
                warnings.warn("Attribute mjdref is different in the time series being merged.")
                others[i] = other.change_mjdref(self.mjdref)

        all_objs = [self] + others

        # Check if none of the GTIs was already initialized.
        all_gti = [obj._gti for obj in all_objs if obj._gti is not None]

        if len(all_gti) == 0 or strategy == "none":
            new_gti = None
        else:
            # For this, initialize the GTIs
            new_gti = merge_gtis([obj.gti for obj in all_objs], strategy=strategy)

        all_time_arrays = [obj.time for obj in all_objs if obj.time is not None]

        new_ts.time = np.concatenate(all_time_arrays)
        order = np.argsort(new_ts.time)
        new_ts.time = new_ts.time[order]

        new_ts.gti = new_gti

        dts = list(set([getattr(obj, "dt", None) for obj in all_objs]))
        if len(dts) != 1:
            warnings.warn("The time resolution is different. Transforming in array")

            new_dt = np.concatenate([np.zeros_like(obj.time) + obj.dt for obj in all_objs])
            new_ts.dt = new_dt[order]
        else:
            new_ts.dt = dts[0]

        def _get_set_from_many_lists(lists):
            """Make a single set out of many lists."""
            all_vals = []
            for ls in lists:
                all_vals += ls
            return set(all_vals)

        def _get_all_array_attrs(objs):
            """Get all array attributes from the time series being merged. Do not include time."""
            return _get_set_from_many_lists(
                [obj.array_attrs() + obj.internal_array_attrs() for obj in objs]
            )

        for attr in _get_all_array_attrs(all_objs):
            # if it's here, it means that it's an array attr in at least one object.
            # So, everywhere it's None, it needs to be set to 0s of the same length as time
            new_attr_values = []
            for obj in all_objs:
                if getattr(obj, attr, None) is None:
                    warnings.warn(
                        f"The {attr} array is empty in one of the time series being merged. "
                        "Setting it to NaN for the affected events"
                    )
                    new_attr_values.append(np.zeros_like(obj.time) + np.nan)
                else:
                    new_attr_values.append(getattr(obj, attr))

            new_attr = np.concatenate(new_attr_values)[order]
            setattr(new_ts, attr, new_attr)

        all_meta_attrs = _get_set_from_many_lists([obj.meta_attrs() for obj in all_objs])
        # The attributes being treated separately are removed from the standard treatment
        # When energy, pi etc. are None, they might appear in the meta_attrs, so we
        # also add them to the list of attributes to be removed if present.
        to_remove = ["gti", "dt"] + new_ts.array_attrs()
        for attr in to_remove:
            if attr in all_meta_attrs:
                all_meta_attrs.remove(attr)

        for attr in ignore_meta:
            logger.info(f"The {attr} attribute will be removed from the output ")
            if attr in all_meta_attrs:
                all_meta_attrs.remove(attr)

        def _safe_concatenate(a, b):
            if isinstance(a, str) and isinstance(b, str):
                return a + "," + b
            else:
                if isinstance(a, tuple):
                    return a + (b,)
                return (a, b)

        for attr in all_meta_attrs:
            self_attr = getattr(self, attr, None)
            new_val = self_attr
            for other in others:
                other_attr = getattr(other, attr, None)
                if self_attr != other_attr:
                    warnings.warn(
                        "Attribute " + attr + " is different in the time series being merged."
                    )
                    new_val = _safe_concatenate(new_val, other_attr)
            setattr(new_ts, attr, new_val)

        new_ts.mjdref = self.mjdref

        return new_ts

    def join(self, *args, **kwargs):
        """
        Join other :class:`StingrayTimeseries` objects with the current one.

        If both are empty, an empty :class:`StingrayTimeseries` is returned.

        Standard attributes such as ``pi`` and ``energy`` remain ``None`` if they are ``None``
        in both. Otherwise, ``np.nan`` is used as a default value for the missing values.
        Arbitrary array attributes are created and joined using the same convention.

        Multiple checks are done on the joined time series. If the time array of the series
        being joined is empty, it is ignored. If the time resolution is different, the final
        time series will have the rougher time resolution. If the MJDREF is different, the time
        reference will be changed to the one of the first time series. An empty time series will
        be ignored.

        Note: ``join`` is not equivalent to ``concatenate``. ``concatenate`` is used to join
        multiple **non-overlapping** time series along the time axis, while ``join`` is more
        general, and can be used to join multiple time series with different strategies (see
        parameter ``strategy`` below).

        Parameters
        ----------
        other : :class:`StingrayTimeseries` or class:`list` of :class:`StingrayTimeseries`
            The other :class:`StingrayTimeseries` object which is supposed to be joined with.
            If ``other`` is a list, it is assumed to be a list of :class:`StingrayTimeseries`
            and they are all joined, one by one.

        Other parameters
        ----------------
        strategy : {"intersection", "union", "append", "infer", "none"}
            Method to use to merge the GTIs. If "intersection", the GTIs are merged
            using the intersection of the GTIs. If "union", the GTIs are merged
            using the union of the GTIs. If "none", a single GTI with the minimum and
            the maximum time stamps of all GTIs is returned. If "infer", the strategy
            is decided based on the GTIs. If there are no overlaps, "union" is used,
            otherwise "intersection" is used. If "append", the GTIs are simply appended
            but they must be mutually exclusive.

        Returns
        -------
        `ts_new` : :class:`StingrayTimeseries` object
            The resulting :class:`StingrayTimeseries` object.
        """
        return self._join_timeseries(*args, **kwargs)

    def rebin(self, dt_new=None, f=None, method="sum"):
        """
        Rebin the time series to a new time resolution. While the new
        resolution need not be an integer multiple of the previous time
        resolution, be aware that if it is not, the last bin will be cut
        off by the fraction left over by the integer division.

        Parameters
        ----------
        dt_new: float
            The new time resolution of the time series. Must be larger than
            the time resolution of the old time series!

        method: {``sum`` | ``mean`` | ``average``}, optional, default ``sum``
            This keyword argument sets whether the counts in the new bins
            should be summed or averaged.

        Other Parameters
        ----------------
        f: float
            the rebin factor. If specified, it substitutes ``dt_new`` with
            ``f*self.dt``

        Returns
        -------
        ts_new: :class:`StingrayTimeseries` object
            The :class:`StingrayTimeseries` object with the new, binned time series.
        """

        if f is None and dt_new is None:
            raise ValueError("You need to specify at least one between f and " "dt_new")
        elif f is not None:
            dt_new = f * self.dt

        if np.any(dt_new < np.asanyarray(self.dt)):
            raise ValueError("The new time resolution must be larger than the old one!")

        gti_new = []

        new_ts = type(self)()

        for attr in self.array_attrs() + self.internal_array_attrs():
            bin_time, bin_counts, bin_err = [], [], []
            if attr.endswith("_err"):
                continue
            e_temp = None
            for g in self.gti:
                if g[1] - g[0] < dt_new:
                    continue
                else:
                    # find start and end of GTI segment in data
                    start_ind = self.time.searchsorted(g[0])
                    end_ind = self.time.searchsorted(g[1])

                    t_temp = self.time[start_ind:end_ind]
                    c_temp = getattr(self, attr)[start_ind:end_ind]

                    if hasattr(self, attr + "_err"):
                        e_temp = getattr(self, attr + "_err")[start_ind:end_ind]

                    bin_t, bin_c, bin_e, _ = rebin_data(
                        t_temp, c_temp, dt_new, yerr=e_temp, method=method, dx=self.dt
                    )

                    bin_time.extend(bin_t)
                    bin_counts.extend(bin_c)
                    bin_err.extend(bin_e)
                    gti_new.append(g)
            if new_ts.time is None:
                new_ts.time = np.array(bin_time)
            setattr(new_ts, attr, bin_counts)
            if e_temp is not None:
                setattr(new_ts, attr + "_err", bin_err)

        if len(gti_new) == 0:
            raise ValueError("No valid GTIs after rebin.")
        new_ts.gti = np.asanyarray(gti_new)

        for attr in self.meta_attrs():
            if attr == "dt":
                continue
            setattr(new_ts, attr, copy.deepcopy(getattr(self, attr)))
        new_ts.dt = dt_new
        return new_ts

    def sort(self, reverse=False, inplace=False):
        """
        Sort a ``StingrayTimeseries`` object by time.

        A ``StingrayTimeseries`` can be sorted in either increasing or decreasing order
        using this method. The time array gets sorted and the counts array is
        changed accordingly.

        Parameters
        ----------
        reverse : boolean, default False
            If True then the object is sorted in reverse order.
        inplace : bool
            If True, overwrite the current time series. Otherwise, return a new one.

        Examples
        --------
        >>> time = [2, 1, 3]
        >>> count = [200, 100, 300]
        >>> ts = StingrayTimeseries(time, array_attrs={"counts": count}, dt=1, skip_checks=True)
        >>> ts_new = ts.sort()
        >>> ts_new.time
        array([1, 2, 3])
        >>> assert np.allclose(ts_new.counts, [100, 200, 300])

        Returns
        -------
        ts_new: :class:`StingrayTimeseries` object
            The :class:`StingrayTimeseries` object with sorted time and counts
            arrays.
        """

        mask = np.argsort(self.time)
        if reverse:
            mask = mask[::-1]
        return self.apply_mask(mask, inplace=inplace)

    def fill_bad_time_intervals(
        self,
        max_length=None,
        attrs_to_randomize=None,
        buffer_size=None,
        even_sampling=None,
        seed=None,
    ):
        """Fill short bad time intervals with random data.

        .. warning::
            This method is only appropriate for *very short* bad time intervals. The simulated data
            are basically white noise, so they are able to alter the statistical properties of
            variable data. For very short gaps in the data, the effect of these small
            injections of white noise should be negligible. How short depends on the single case,
            the user is urged not to use the method as a black box and make simulations to measure
            its effect. If you have long bad time intervals, you should use more advanced
            techniques, not currently available in Stingray for this use case, such as Gaussian
            Processes. In particular, please verify that the values of ``max_length`` and
            ``buffer_size`` are adequate to your case.

        To fill the gaps in all but the time points (i.e., flux measures, energies), we take the
        ``buffer_size`` (by default, the largest value between 100 and the estimated samples in
        a ``max_length``-long gap) valid data points closest to the gap and repeat them randomly
        with the same empirical statistical distribution. So, if the `my_fancy_attr` attribute, in
        the 100 points of the buffer, has 30 times 10, 10 times 9, and 60 times 11, there will be
        *on average* 30% of 10, 60% of 11, and 10% of 9 in the simulated data.

        Times are treated differently depending on the fact that the time series is evenly
        sampled or not. If it is not, the times are simulated from a uniform distribution with the
        same count rate found in the buffer. Otherwise, times just follow the same grid used
        inside GTIs. Using the evenly sampled or not is decided based on the ``even_sampling``
        parameter. If left to ``None``, the time series is considered evenly sampled if
        ``self.dt`` is greater than zero and the median separation between subsequent times is
        within 1% of the time resolution.

        Other Parameters
        ----------------
        max_length : float
            Maximum length of a bad time interval to be filled. If None, the criterion is bad
            time intervals shorter than 1/100th of the longest good time interval.
        attrs_to_randomize : list of str, default None
            List of array_attrs to randomize. ``If None``, all array_attrs are randomized.
            It should not include ``time`` and ``_mask``, which are treated separately.
        buffer_size : int, default 100
            Number of good data points to use to calculate the means and variance the random data
            on each side of the bad time interval
        even_sampling : bool, default None
            Force the treatment of the data as evenly sampled or not. If None, the data are
            considered evenly sampled if ``self.dt`` is larger than zero and the median
            separation between subsequent times is within 1% of ``self.dt``.
        seed : int, default None
            Random seed to use for the simulation. If None, a random seed is generated.

        """

        rs = get_random_state(seed)

        if attrs_to_randomize is None:
            attrs_to_randomize = self.array_attrs() + self.internal_array_attrs()
            for attr in ["time", "_mask"]:
                if attr in attrs_to_randomize:
                    attrs_to_randomize.remove(attr)

        attrs_to_leave_alone = [
            a
            for a in self.array_attrs() + self.internal_array_attrs()
            if a not in attrs_to_randomize
        ]

        if max_length is None:
            max_length = np.max(self.gti[:, 1] - self.gti[:, 0]) / 100

        btis = get_btis(self.gti, self.time[0], self.time[-1])
        if len(btis) == 0:
            logger.info("No bad time intervals to fill")
            return copy.deepcopy(self)
        filtered_times = self.time[self.mask]

        new_times = [filtered_times.copy()]
        new_attrs = {}
        mean_data_separation = np.median(np.diff(filtered_times))
        if even_sampling is None:
            # The time series is considered evenly sampled if the median separation between
            # subsequent times is within 1% of the time resolution
            even_sampling = False
            if self.dt > 0 and np.isclose(mean_data_separation, self.dt, rtol=0.01):
                even_sampling = True
            logger.info(f"Data are {'not' if not even_sampling else ''} evenly sampled")

        if even_sampling:
            est_samples_in_gap = int(max_length / self.dt)
        else:
            est_samples_in_gap = int(max_length / mean_data_separation)

        if buffer_size is None:
            buffer_size = max(100, est_samples_in_gap)

        added_gtis = []

        total_filled_time = 0
        for bti in btis:
            length = bti[1] - bti[0]
            if length > max_length:
                continue
            logger.info(f"Filling bad time interval {bti} ({length:.4f} s)")
            epsilon = 1e-5 * length
            added_gtis.append([bti[0] - epsilon, bti[1] + epsilon])
            filt_low_t, filt_low_idx = find_nearest(filtered_times, bti[0])
            filt_hig_t, filt_hig_idx = find_nearest(filtered_times, bti[1], side="right")
            if even_sampling:
                local_new_times = np.arange(bti[0] + self.dt / 2, bti[1], self.dt)
                nevents = local_new_times.size
            else:
                low_time_arr = filtered_times[max(filt_low_idx - buffer_size, 0) : filt_low_idx]
                low_time_arr = low_time_arr[low_time_arr > bti[0] - buffer_size]
                high_time_arr = filtered_times[filt_hig_idx : buffer_size + filt_hig_idx]
                high_time_arr = high_time_arr[high_time_arr < bti[1] + buffer_size]

                if len(low_time_arr) > 0 and (filt_low_t - low_time_arr[0]) > 0:
                    ctrate_low = np.count_nonzero(low_time_arr) / (filt_low_t - low_time_arr[0])
                else:
                    ctrate_low = np.nan
                if len(high_time_arr) > 0 and (high_time_arr[-1] - filt_hig_t) > 0:
                    ctrate_high = np.count_nonzero(high_time_arr) / (high_time_arr[-1] - filt_hig_t)
                else:
                    ctrate_high = np.nan

                if not np.isfinite(ctrate_low) and not np.isfinite(ctrate_high):
                    warnings.warn(
                        f"No valid data around to simulate the time series in interval "
                        f"{bti[0]:g}-{bti[1]:g}. Skipping. Please check that the buffer size is "
                        f"adequate."
                    )
                    continue
                ctrate = np.nanmean([ctrate_low, ctrate_high])
                nevents = rs.poisson(ctrate * (bti[1] - bti[0]))
                local_new_times = rs.uniform(bti[0], bti[1], nevents)
            new_times.append(local_new_times)

            for attr in attrs_to_randomize:
                low_arr = getattr(self, attr)[max(buffer_size - filt_low_idx, 0) : filt_low_idx]
                high_arr = getattr(self, attr)[filt_hig_idx : buffer_size + filt_hig_idx]
                if attr not in new_attrs:
                    new_attrs[attr] = [getattr(self, attr)[self.mask]]
                new_attrs[attr].append(rs.choice(np.concatenate([low_arr, high_arr]), nevents))
            for attr in attrs_to_leave_alone:
                if attr not in new_attrs:
                    new_attrs[attr] = [getattr(self, attr)[self.mask]]
                if attr == "_mask":
                    new_attrs[attr].append(np.ones(nevents, dtype=bool))
                else:
                    new_attrs[attr].append(np.zeros(nevents) + np.nan)
            total_filled_time += length

        logger.info(f"A total of {total_filled_time} s of data were simulated")

        new_gtis = join_gtis(self.gti, added_gtis)
        new_times = np.concatenate(new_times)
        order = np.argsort(new_times)
        new_obj = type(self)()
        new_obj.time = new_times[order]

        for attr in self.meta_attrs():
            setattr(new_obj, attr, getattr(self, attr))

        for attr, values in new_attrs.items():
            setattr(new_obj, attr, np.concatenate(values)[order])
        new_obj.gti = new_gtis
        return new_obj

    def plot(
        self,
        attr,
        witherrors=False,
        labels=None,
        ax=None,
        title=None,
        marker="-",
        save=False,
        filename=None,
        plot_btis=True,
        axis_limits=None,
    ):
        """
        Plot the time series using ``matplotlib``.

        Plot the time series object on a graph ``self.time`` on x-axis and
        ``self.counts`` on y-axis with ``self.counts_err`` optionally
        as error bars.

        Parameters
        ----------
        attr: str
            Attribute to plot.

        Other parameters
        ----------------
        witherrors: boolean, default False
            Whether to plot the StingrayTimeseries with errorbars or not
        labels : iterable, default ``None``
            A list or tuple with ``xlabel`` and ``ylabel`` as strings. E.g.
            if the attribute is ``'counts'``, the list of labels
            could be ``['Time (s)', 'Counts (s^-1)']``
        ax : ``matplotlib.pyplot.axis`` object
            Axis to be used for plotting. Defaults to creating a new one.
        axis_limits : list, tuple, string, default ``None``
            Parameter to set axis properties of the ``matplotlib`` figure. For example
            it can be a list like ``[xmin, xmax, ymin, ymax]`` or any other
            acceptable argument for the``matplotlib.pyplot.axis()`` method.
        title : str, default ``None``
            The title of the plot.
        marker : str, default '-'
            Line style and color of the plot. Line styles and colors are
            combined in a single format string, as in ``'bo'`` for blue
            circles. See ``matplotlib.pyplot.plot`` for more options.
        save : boolean, optional, default ``False``
            If ``True``, save the figure with specified filename.
        filename : str
            File name of the image to save. Depends on the boolean ``save``.
        plot_btis : bool
            Plot the bad time intervals as red areas on the plot
        """
        import matplotlib.pyplot as plt

        if ax is None:
            plt.figure(attr)
            ax = plt.gca()

        valid_labels = (isinstance(labels, Iterable) and not isinstance(labels, str)) and len(
            labels
        ) == 2
        if labels is not None and not valid_labels:
            warnings.warn("``labels`` must be an iterable with two labels for x and y axes.")

        if labels is None or not valid_labels:
            labels = ["Time (s)"] + [attr]

        xlabel = labels[0]
        ylabel = labels[1]
        # Default values for labels

        ax.plot(self.time, getattr(self, attr), marker, ds="steps-mid", label=attr, zorder=10)

        if witherrors and (
            attr + "_err" in self.array_attrs()
            or "_" + attr + "_err" in self.internal_array_attrs()
        ):
            ax.errorbar(
                self.time,
                getattr(self, attr),
                yerr=getattr(self, attr + "_err"),
                fmt="o",
                zorder=10,
            )

        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

        if axis_limits is not None:
            ax.set_xlim(axis_limits[0], axis_limits[1])
            ax.set_ylim(axis_limits[2], axis_limits[3])
        if title is not None:
            ax.set_title(title)

        if save:
            if filename is None:
                ax.figure.savefig("out.png")
            else:
                ax.figure.savefig(filename)

        if plot_btis and self.gti is not None and len(self.gti) > 1:
            tstart = min(self.time[0] - self.dt / 2, self.gti[0, 0])
            tend = max(self.time[-1] + self.dt / 2, self.gti[-1, 1])
            btis = get_btis(self.gti, tstart, tend)
            for bti in btis:
                plt.axvspan(
                    bti[0],
                    bti[1],
                    alpha=0.5,
                    facecolor="r",
                    zorder=1,
                    edgecolor="none",
                )
        return ax

    def estimate_segment_size(self, min_counts=None, min_samples=None, even_sampling=None):
        """Estimate a reasonable segment length for segment-by-segment analysis.

        The user has to specify a criterion based on a minimum number of counts (if
        the time series has a ``counts`` attribute) or a minimum number of time samples.
        At least one between ``min_counts`` and ``min_samples`` must be specified.
        In the special case of a time series with ``dt=0`` (event list-like, where each time
        stamp correspond to a single count), the two definitions are equivalent.

        Other Parameters
        ----------------
        min_counts : int
            Minimum number of counts for each chunk. Optional (but needs ``min_samples``
            if left unspecified). Only makes sense if the series has a ``counts`` attribute and
            it is evenly sampled.
        min_samples : int
            Minimum number of time bins. Optional (but needs ``min_counts`` if left unspecified).
        even_sampling : bool
            Force the treatment of the data as evenly sampled or not. If None, the data are
            considered evenly sampled if ``self.dt`` is larger than zero and the median
            separation between subsequent times is within 1% of ``self.dt``.

        Returns
        -------
        segment_size : float
            The length of the light curve chunks that satisfies the conditions

        Examples
        --------
        >>> import numpy as np
        >>> time = np.arange(150)
        >>> counts = np.zeros_like(time) + 3
        >>> ts = StingrayTimeseries(time, counts=counts, dt=1)
        >>> assert np.isclose(ts.estimate_segment_size(min_counts=10, min_samples=3), 4.0)
        >>> assert np.isclose(ts.estimate_segment_size(min_counts=10, min_samples=5), 5.0)
        >>> counts[2:4] = 1
        >>> ts = StingrayTimeseries(time, counts=counts, dt=1)
        >>> assert np.isclose(ts.estimate_segment_size(min_counts=3, min_samples=1), 3.0)
        >>> # A slightly more complex example
        >>> dt=0.2
        >>> time = np.arange(0, 1000, dt)
        >>> counts = np.random.poisson(100, size=len(time))
        >>> ts = StingrayTimeseries(time, counts=counts, dt=dt)
        >>> assert np.isclose(ts.estimate_segment_size(100, 2), 0.4)
        >>> min_total_bins = 40
        >>> assert np.isclose(ts.estimate_segment_size(100, 40), 8.0)
        """
        if min_counts is None and min_samples is None:
            raise ValueError("You have to specify at least one of min_counts or min_samples")

        mean_data_separation = np.median(np.diff(self.time))

        if even_sampling is None:
            # The time series is considered evenly sampled if the median separation between
            # subsequent times is within 1% of the time resolution
            even_sampling = False
            if (
                self.dt is not None
                and self.dt > 0
                and np.isclose(mean_data_separation, self.dt, rtol=0.01)
            ):
                even_sampling = True
            logger.info(f"Data are {'not' if not even_sampling else ''} evenly sampled")

        if min_counts is None:
            if even_sampling and hasattr(self, "counts"):
                min_counts = 0
            else:
                min_counts = min_samples

        mean_ctrate = _ts_sum(self) / self.exposure

        rough_estimate = np.ceil(min_counts / mean_ctrate)

        # If data are evenly sampled, even sampling make the segment an integer multiple of dt.
        # Otherwise, just use steps of 1 second.
        if even_sampling:
            step = self.dt
        else:
            step = 1.0

        rough_estimate = np.ceil(min_counts / mean_ctrate / step) * step

        segment_size = np.max([rough_estimate, min_samples * step])

        keep_searching = True

        while keep_searching:
            start_times, stop_times, results = self.analyze_segments(_ts_sum, segment_size)
            mincounts = np.min(results)
            if mincounts >= min_counts:
                keep_searching = False
            else:
                segment_size += step

        return segment_size

    def analyze_segments(self, func, segment_size, fraction_step=1, **kwargs):
        """Analyze segments of the light curve with any function.

        Intervals with less than one data point are skipped.

        Parameters
        ----------
        func : function
            Function accepting a :class:`StingrayTimeseries` object as single argument, plus
            possible additional keyword arguments, and returning a number or a
            tuple - e.g., ``(result, error)`` where both ``result`` and ``error`` are
            numbers.
        segment_size : float
            Length in seconds of the light curve segments. If None, the full GTIs are considered
            instead as segments.

        Other parameters
        ----------------
        fraction_step : float
            If the step is not a full ``segment_size`` but less (e.g. a moving window),
            this indicates the ratio between step step and ``segment_size`` (e.g.
            0.5 means that the window shifts of half ``segment_size``)
        kwargs : keyword arguments
            These additional keyword arguments, if present, they will be passed
            to ``func``

        Returns
        -------
        start_times : array
            Lower time boundaries of all time segments.
        stop_times : array
            upper time boundaries of all segments.
        result : list of N elements
            The result of ``func`` for each segment of the light curve. If the function
            returns multiple outputs, they are returned as a list of arrays.
            If a given interval has not enough data for a calculation, ``None`` is returned.

        Examples
        --------
        >>> import numpy as np
        >>> time = np.arange(0, 10, 0.1)
        >>> counts = np.zeros_like(time) + 10
        >>> ts = StingrayTimeseries(time, counts=counts, dt=0.1)
        >>> # Define a function that calculates the mean
        >>> mean_func = lambda ts: np.mean(ts.counts)
        >>> # Calculate the mean in segments of 5 seconds
        >>> start, stop, res = ts.analyze_segments(mean_func, 5)
        >>> len(res) == 2
        True
        >>> np.allclose(res, 10)
        True
        """

        if segment_size is None:
            start_times = self.gti[:, 0]
            stop_times = self.gti[:, 1]
            start = np.searchsorted(self.time, start_times)
            stop = np.searchsorted(self.time, stop_times)
        elif self.dt > 0:
            start, stop = bin_intervals_from_gtis(
                self.gti, segment_size, self.time, fraction_step=fraction_step, dt=self.dt
            )
            start_times = self.time[start] - 0.5 * self.dt
            # Remember that stop is one element above the last element, because
            # it's defined to be used in intervals start:stop
            stop_times = self.time[stop - 1] + self.dt * 1.5
        else:
            start_times, stop_times = time_intervals_from_gtis(
                self.gti, segment_size, fraction_step=fraction_step
            )
            start = np.searchsorted(self.time, start_times)
            stop = np.searchsorted(self.time, stop_times)

        results = []

        n_outs = 1
        for i, (st, sp, tst, tsp) in enumerate(zip(start, stop, start_times, stop_times)):
            if sp - st <= 1:
                warnings.warn(
                    f"Segment {i} ({tst}--{tsp}) has one data point or less. Skipping it "
                )

                continue
            lc_filt = self[st:sp]
            lc_filt.gti = np.asanyarray([[tst, tsp]])

            res = func(lc_filt, **kwargs)
            results.append(res)
            if isinstance(res, Iterable) and not isinstance(res, str):
                n_outs = len(res)

        # If the function returns multiple outputs, we need to separate them

        if n_outs > 1:
            outs = [[] for _ in range(n_outs)]
            for res in results:
                for i in range(n_outs):
                    outs[i].append(res[i])
            results = outs

        # Try to transform into a (possibly multi-dimensional) numpy array
        try:
            results = np.array(results)
        except ValueError:  # pragma: no cover
            pass

        return start_times, stop_times, results

    def analyze_by_gti(self, func, fraction_step=1, **kwargs):
        """Analyze the light curve with any function, on a GTI-by-GTI base.

        Parameters
        ----------
        func : function
            Function accepting a :class:`StingrayTimeseries` object as single argument, plus
            possible additional keyword arguments, and returning a number or a
            tuple - e.g., ``(result, error)`` where both ``result`` and ``error`` are
            numbers.

        Other parameters
        ----------------
        fraction_step : float
            By default, segments do not overlap (``fraction_step`` = 1). If ``fraction_step`` < 1,
            then the start points of consecutive segments are ``fraction_step * segment_size``
            apart, and consecutive segments overlap. For example, for ``fraction_step`` = 0.5,
            the window shifts one half of ``segment_size``)
        kwargs : keyword arguments
            These additional keyword arguments, if present, they will be passed
            to ``func``

        Returns
        -------
        start_times : array
            Lower time boundaries of all time segments.
        stop_times : array
            upper time boundaries of all segments.
        result : array of N elements
            The result of ``func`` for each segment of the light curve
        """
        return self.analyze_segments(func, segment_size=None, fraction_step=fraction_step, **kwargs)

    def apply_gti_lists(self, new_gti_lists):
        """Split the event list into different files, each with a different GTI.

        Parameters
        ----------
        new_gti_lists : list of lists
            A list of lists of GTIs. Each sublist should contain a list of GTIs
            for a new file.

        Returns
        -------
        output_files : list of str
            A list of the output file names.

        """

        if len(new_gti_lists[0]) == len(self.gti) and np.all(
            np.abs(np.asanyarray(new_gti_lists[0]).flatten() - self.gti.flatten()) < 1e-3
        ):
            ev = self[:]
            yield ev

        else:
            for gti in new_gti_lists:
                if len(gti) == 0:
                    continue
                gti = np.asarray(gti)
                lower_edge = np.searchsorted(self.time, gti[0, 0])
                upper_edge = np.searchsorted(self.time, gti[-1, 1])
                if upper_edge == self.time.size:
                    upper_edge -= 1
                if self.time[upper_edge] > gti[-1, 1]:
                    upper_edge -= 1
                ev = self[lower_edge : upper_edge + 1]

                if hasattr(ev, "gti"):
                    ev.gti = gti

                yield ev

    def filter_at_time_intervals(self, time_intervals, check_gtis=True):
        """Filter the event list at the given time intervals.

        Parameters
        ----------
        time_intervals : 2-d float array
            List of time intervals of the form ``[[time0_0, time0_1], [time1_0, time1_1], ...]``

        Returns
        -------
        output_files : list of str
            A list of the output file names.
        """
        if len(np.shape(time_intervals)) == 1:
            time_intervals = [time_intervals]
        if check_gtis:
            new_gti = [cross_two_gtis(self.gti, [t_int]) for t_int in time_intervals]
        else:
            new_gti = [np.asarray([t_int]) for t_int in time_intervals]
        return self.apply_gti_lists(new_gti)


def interpret_times(time: TTime, mjdref: float = 0) -> tuple[npt.ArrayLike, float]:
    """Understand the format of input times, and return seconds from MJDREF

    Parameters
    ----------
    time : class:`astropy.Time`, class:`time.Time`, class:`astropy.TimeDelta`, class:`astropy.Quantity`, class:`np.array`
        Input times.

    Other Parameters
    ----------------
    mjdref : float
        Input MJD reference of the times. Optional.

    Returns
    -------
    time_s : class:`np.array`
        Times, in seconds from MJDREF
    mjdref : float
        MJDREF. If the input time is a `time.Time` object and the input mjdref
        is 0, it will be defined as the MJD of the input time.

    Examples
    --------
    >>> import astropy.units as u
    >>> newt, mjdref = interpret_times(None)
    >>> assert newt is None
    >>> time = Time(57483, format='mjd')
    >>> newt, mjdref = interpret_times(time)
    >>> assert newt == 0
    >>> assert mjdref == 57483
    >>> time = Time([57483], format='mjd')
    >>> newt, mjdref = interpret_times(time)
    >>> assert np.allclose(newt, 0)
    >>> assert mjdref == 57483
    >>> time = TimeDelta([3, 4, 5] * u.s)
    >>> newt, mjdref = interpret_times(time)
    >>> assert np.allclose(newt, [3, 4, 5])
    >>> time = np.array([3, 4, 5])
    >>> newt, mjdref = interpret_times(time, mjdref=45000)
    >>> assert np.allclose(newt, [3, 4, 5])
    >>> assert mjdref == 45000
    >>> time = np.array([3, 4, 5] * u.s)
    >>> newt, mjdref = interpret_times(time, mjdref=45000)
    >>> assert np.allclose(newt, [3, 4, 5])
    >>> assert mjdref == 45000
    >>> newt, mjdref = interpret_times(1, mjdref=45000)
    >>> assert newt == 1
    >>> newt, mjdref = interpret_times(list, mjdref=45000)
    Traceback (most recent call last):
    ...
    ValueError: Unknown time format: ...
    >>> newt, mjdref = interpret_times("guadfkljfd", mjdref=45000)
    Traceback (most recent call last):
    ...
    ValueError: Unknown time format: ...
    """
    if time is None:
        return None, mjdref

    if isinstance(time, TimeDelta):
        out_times = time.to("s").value
        return out_times, mjdref

    if isinstance(time, Time):
        mjds = time.mjd
        if mjdref == 0:
            if np.all(mjds > 10000):
                if isinstance(mjds, Iterable):
                    mjdref = mjds[0]
                else:
                    mjdref = mjds

        out_times = (mjds - mjdref) * 86400
        return out_times, mjdref

    if isinstance(time, Quantity):
        out_times = time.to("s").value
        return out_times, mjdref

    if isinstance(time, (tuple, list, np.ndarray)):
        return time, mjdref

    if not isinstance(time, Iterable):
        try:
            float(time)
            return time, mjdref
        except (ValueError, TypeError):
            pass

    raise ValueError(f"Unknown time format: {type(time)}")


def reduce_precision_if_extended(
    x, probe_types=["float128", "float96", "float80", "longdouble"], destination=float
):
    """Reduce a number to a standard float if extended precision.

    Ignore all non-float types.

    Parameters
    ----------
    x : float
        The number to be reduced

    Returns
    -------
    x_red : same type of input
        The input, only reduce to ``float`` precision if ``np.float128``

    Examples
    --------
    >>> x = 1.0
    >>> val = reduce_precision_if_extended(x, probe_types=["float64"])
    >>> assert val is x
    >>> x = "1wrt"
    >>> assert reduce_precision_if_extended(x, probe_types=["float64"]) is x
    >>> x = np.asanyarray(1.0).astype(int)
    >>> val = reduce_precision_if_extended(x, probe_types=["float64"])
    >>> assert val is x
    >>> x = np.asanyarray([1.0, 2]).astype(int)
    >>> val = reduce_precision_if_extended(x, probe_types=["float64"])
    >>> assert val is x
    >>> x = np.asanyarray([1.0]).astype(int)
    >>> val = reduce_precision_if_extended(x, probe_types=["float64"])
    >>> assert val is x
    >>> x = np.asanyarray(1.0).astype(np.float64)
    >>> reduce_precision_if_extended(x, probe_types=["float64"], destination=np.float32) is x
    False
    >>> x = np.asanyarray([1.0]).astype(np.float64)
    >>> reduce_precision_if_extended(x, probe_types=["float64"], destination=np.float32) is x
    False
    """

    def obj2sctype(x):
        """Convert an object to a numpy scalar type."""
        if hasattr(np, "obj2sctype"):
            return np.obj2sctype(x)

        if isinstance(x, str):
            return "str"

        if isinstance(x, Iterable) and np.size(x) > 1:
            return obj2sctype(x[0])

        if "numpy" not in str(type(x)):
            return "None"

        return x.dtype.type
        # return np.dtype(x).type

    if any([t in str(obj2sctype(x)) for t in probe_types]):
        x_ret = x.astype(destination)
        return x_ret
    return x
