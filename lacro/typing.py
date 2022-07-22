# -*- coding: utf-8 -*-
from typing import KeysView, Sequence, TypeVar, Union, ValuesView

T_co = TypeVar('T_co', covariant=True)

SizedIterable = Union[Sequence[T_co], ValuesView[T_co], KeysView[T_co]]
