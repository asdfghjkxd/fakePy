import functools
from typing import *
from copy import deepcopy
from operator import mul, add


class Array:
    def __init__(self, shape: Optional[Union[List[int], Tuple[int]]] = None,
                 init: Optional[List[List]] = None):
        self.mat = self._zero_matrix(shape) if init is None else init

    def __call__(self):
        return self.mat

    def __repr__(self):
        return str(self.mat)

    def __str__(self):
        return str(self.mat)

    #TODO: MATRIX ADDITION, SUBTRACTION, MULTIPLICATION
    def __add__(self, other):
        if self.assert_dims(other):
            pass
        else:
            raise ValueError('Both Arrays are not of the same dimension spec')

    def __sub__(self, other):
        if self.assert_dims(other):
            pass
        else:
            raise ValueError('Both Arrays are not of the same dimension spec')

    def __mul__(self, other):
        """M x A  (*)  A x N == M x N"""
        self_shape = self.shape()
        other_shape = other.shape()
        if self_shape[-1] != other_shape[0]:
            raise ValueError('Both Arrays of not of compatible dimension spec')
        else:
            pass

    def __neg__(self):
        """Using inner func for reshape"""

        def go_inner(ls: List):
            if type(ls) == list:
                if len(ls) > 0:
                    if any(map(lambda x: type(x) == int, ls)):
                        ls[:] = [-v for v in ls]
                    else:
                        go_inner(ls[0])
                        go_inner(ls[1:])
            else:
                return

        go_inner(self.mat)
        return Array(init=self.mat)

    def assert_dims(self, other):
        return functools.reduce(mul, self.shape()) == functools.reduce(mul, other.shape())

    def _zero_matrix(self, dims: Union[List[int], Tuple[int]]):
        """Returns an empty matrix"""

        if len(dims) == 1:
            arr_shape = dims[0]
            return type(dims)([0 for _ in range(arr_shape)])
        else:
            return [self._zero_matrix(dims[1:]) for _ in range(dims[0])]

    def _modify(self, idx: Union[List[int], Tuple[int]], value: Optional[Any] = None):
        """Inner function to search and modify the idx"""

        # assert dims are the same
        assert len(shape(self.mat)) != len(idx), 'Invalid Dimensions'

        try:
            if len(shape(self.mat)) == 1:
                self.mat[idx[0]] = value
            else:
                current = self.mat[idx[0]]
                print(current)
                idx = idx[1:]

                for i in idx[:-1]:
                    current = current[i]

                current[idx[-1]] = 0 if value is None else value
        except (IndexError, KeyError):
            raise ValueError(f'{idx} is invalid')

    def shape(self):
        """Returns the shape of the array"""

        def inner(ls: List[List]):
            if type(ls) != list and type(ls) != tuple:
                return
            else:
                mat_shape.append(len(ls))
                inner(ls[0])

        mat_shape = []

        inner(self.mat)

        if mat_shape is []:
            return 'Invalid Matrix'
        else:
            return mat_shape

    def insert(self, idx: Union[List[int], Tuple[int]], value: Optional[Any] = None):
        self._modify(idx, value)

    def delete(self, idx: Union[List[int], Tuple[int]]):
        self._modify(idx)

    def reshape(self, dims: Union[List[int], Tuple[int]], inplace: bool = False):
        if functools.reduce(mul, dims) != functools.reduce(mul, self.shape()):
            raise ValueError(f'Invalid dimension spec: {dims}')
        else:
            def go_inner(ls: List):
                if type(ls) == list:
                    if len(ls) > 0:
                        if any(map(lambda x: type(x) == int, ls)):
                            ls[:] = [flattened.pop(0) for _ in range(len(ls))]
                        else:
                            go_inner(ls[0])
                            go_inner(ls[1:])
                else:
                    return

            flattened = self.flatten(inplace=False)
            new = self._zero_matrix(dims)
            go_inner(new)

            if inplace:
                self.mat = new
            else:
                return new

    def expand_dims(self, inplace: bool = False):
        """Increases the dimensionality of the array by 1, adding one onto the outermost dimension"""
        return self.reshape(dims=[1] + self.shape(), inplace=inplace)

    def compress_along_axis(self, axis: int):
        """Sums up all elements along a specified axis, lists are concatenated and values are added up"""

        if axis < 0 or axis > len(self.shape()) - 1:
            raise ValueError(f'Cannot squeeze to axis {axis}')
        elif len(self.shape()) == 1:
            return [sum(self.mat)]
        else:
            def go_layer(ls: List):
                nonlocal descended_layer

                if type(ls) == list:
                    if len(ls) > 0:
                        if descended_layer == axis:
                            if type(functools.reduce(add, ls)) == int:
                                ls[:] = [functools.reduce(add, ls)]
                                return descended_layer
                            else:
                                ls[:] = functools.reduce(add, ls)
                        else:
                            descended_layer += 1
                            traversed = go_layer(ls[0])
                            if traversed is not None:
                                descended_layer -= traversed - 1
                            go_layer(ls[1:])
                else:
                    return

            descended_layer = 0
            go_layer(self.mat)

    def squeeze(self, axis: Optional[int] = None, inplace: bool = False):
        """Destroys all redundant dimensions in the array (all 1s)"""

        shape = self.shape()

        if axis is None:
            shape = list(filter(lambda x: x > 1, shape))
        else:
            shape = shape[:axis] + shape[axis + 1:]
            if functools.reduce(mul, shape) != functools.reduce(mul, self.shape()):
                raise ValueError('Invalid Squeeze Axis: Squeezing along this axis creates an array with the wrong '
                                 'flattened shape')

        if inplace:
            self.reshape(dims=shape, inplace=True)
        else:
            return self.reshape(dims=shape)

    def flatten(self, inplace: bool = False):
        def collapse(arr):
            return functools.reduce(lambda x, y: x + y, arr)

        flattened_shape = functools.reduce(lambda x, y: x * y, self.shape())
        copied = deepcopy(self.mat)

        while len(copied) != flattened_shape:
            copied = collapse(copied)

        if inplace:
            self.mat = copied
        else:
            return copied

    def transpose(self):
        """Swap the first 2 dimensions"""

        self.mat = list(map(list, zip(*self.mat)))


if __name__ == '__main__':
    mat3 = Array(init=[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    print('Shape:', mat3.shape())
    mat3.reshape(dims=[1, 3, 1, 3], inplace=True)
    print(mat3)
    mat3.squeeze(axis=0, inplace=True)
    print(mat3, '\tShape:', mat3.shape())
    mat3.expand_dims(inplace=True)
    print(mat3, '\tShape:', mat3.shape())
    mat3 = -mat3
    print(mat3)
    mat3 = -mat3
    print(mat3)
    mat3.flatten(inplace=True)
    print(mat3)
    mat3.reshape(dims=[1, 3, 3], inplace=True)
    print(mat3)
    mat3.compress_along_axis(axis=0)
    print(mat3)
    mat3.reshape(dims=[3, 3, 1], inplace=True)
    print(mat3)
    mat3.transpose()
    print(mat3)
    mat3.transpose()
    print(mat3)
