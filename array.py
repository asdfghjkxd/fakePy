import functools
from typing import *
from copy import deepcopy
from operator import mul, add, sub


class Array:
    """A crappy copy of numpy's implementation of ndarray"""

    def __init__(self, shape: Optional[Union[List[int], Tuple[int]]] = None,
                 init: Optional[Union[List[List], List[int]]] = None):

        if all(map(lambda x: x is None, (shape, init))):
            raise ValueError('Either shape or initialisation matrix must be defined')
        self.mat = self._zero_matrix(shape) if init is None else init
        flat = self.flatten()
        if not all(type(x) == int or float for x in flat):
            self.mat = None
            raise ValueError('Array must contain only ints and floats')

    def __call__(self):
        return self.mat

    def __repr__(self):
        return str(self.mat)

    def __str__(self):
        return str(self.mat)

    def __add__(self, other):
        if self.assert_dims(other):
            self_flat = self.flatten(inplace=False)
            other_flat = other.flatten(inplace=False)
            combined = [functools.reduce(add, x) for x in (zip(self_flat, other_flat))]
            new_arr = Array(init=combined)
            new_arr.reshape(dims=self.shape(), inplace=True)
            return new_arr
        else:
            raise ValueError('Both Arrays are not of the same dimension spec')

    def __sub__(self, other):
        if self.assert_dims(other):
            self_flat = self.flatten(inplace=False)
            other_flat = other.flatten(inplace=False)
            combined = [functools.reduce(sub, x) for x in (zip(self_flat, other_flat))]
            new_arr = Array(init=combined)
            new_arr.reshape(dims=self.shape(), inplace=True)
            return new_arr
        else:
            raise ValueError('Both Arrays are not of the same dimension spec')

    def __mul__(self, other):
        """M x A  (*)  A x N == M x N"""

        def go_factor(ls: List, operator: Callable):
            if type(ls) == list:
                if len(ls) > 0:
                    if any(map(lambda x: type(x) == int, ls)):
                        ls[:] = [operator(v) for v in ls]
                    else:
                        go_factor(ls[0], operator)
                        go_factor(ls[1:], operator)
            else:
                return

        if isinstance(other, int):
            go_factor(self.mat, operator=lambda x: other * x)
            return Array(init=self.mat)
        elif isinstance(other, Array):
            self_shape = self.shape()
            other_shape = other.shape()
            if self_shape[-1] != other_shape[0]:
                raise ValueError('Both Arrays of not of compatible dimension spec')
            elif len(self_shape) == len(other_shape) == 2:
                # TODO ONLY WORKS FOR 2D ARRS FOR NOW IDK LOL
                self_mat = self()
                other_mat = other()
                zeros = self._zero_matrix(dims=[self_shape[0], other_shape[-1]])

                # taken from https://www.programiz.com/python-programming/examples/multiply-matrix cuz my math fail
                for i in range(len(self_mat)):
                    for j in range(len(other_mat[0])):
                        for k in range(len(other_mat)):
                            zeros[i][j] += self_mat[i][k] * other_mat[k][j]

                reshaped = Array(init=zeros)
                return reshaped
            else:
                raise ValueError('Array multiplication can only be done on 2D arrays')
        else:
            raise TypeError(f'Cannot multiply array by type<{type(other)}>')

    def __neg__(self):
        """Using inner func for negation"""

        return self.__mul__(-1)

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

    @classmethod
    def arange(cls, start: Union[int, float] = 0, end: Union[float, int] = 0, step: Optional[Union[int, float]] = None,
               shape: Optional[Union[List[int], Tuple[int]]] = None):
        """Similar to numpy's arange, generating a range of numbers according to step and then reshaping if necessary"""

        if step is None:
            step = type(end)(1)

        all_elements = []

        while start < end:
            all_elements.append(start)
            start += step

        new_arange = Array(init=all_elements)

        if shape is not None:
            new_arange.reshape(dims=shape, inplace=True)

        return new_arange

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
        """Inserts the item into the array"""
        self._modify(idx, value)

    def delete(self, idx: Union[List[int], Tuple[int]]):
        """Purge an item from the array and replace it by 0"""
        self._modify(idx)

    def reshape(self, dims: Union[List[int], Tuple[int]], inplace: bool = False):
        """Important Array reshaping function"""

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

    def sum(self, axis: int):
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
                                #TODO
                                # ls[:] = functools.reduce(add, ls)
                                # print('THING', ls)

                                return descended_layer
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
        """Flattens the array into 1D"""

        def collapse(arr):
            """Destroys the outermost dimension"""

            return functools.reduce(add, arr)

        flattened_shape = functools.reduce(mul, self.shape())
        copied = deepcopy(self.mat)

        while len(copied) != flattened_shape or type(copied[0]) == list:
            copied = collapse(copied)

        if inplace:
            self.mat = copied
        else:
            return copied

    def transpose(self):
        """Swap the first 2 dimensions"""

        # only works on 2 dims ;-;
        if len(self.shape()) == 2:
            self.mat = list(map(list, zip(*self.mat)))
        elif len(self.shape()) > 2:
            print('Warning: Tranposing >2D Matrix, result is the permutation of the first 2 axes')
            self.mat = list(map(list, zip(*self.mat)))

    def diagonal(self):
        """
        Returns all diagonals of the array

        Idea taken from:
        https://stackoverflow.com/questions/6313308/get-all-the-diagonals-in-a-matrix-list-of-lists-in-python
        """

        curr_shape = self.shape()
        mat = self.mat
        while curr_shape[0] == 1:
            mat = self.reshape(dims=curr_shape[1:])
            curr_shape = curr_shape[1:]

        max_col = len(mat[0])
        max_row = len(mat)
        cols = [[] for _ in range(max_col)]
        rows = [[] for _ in range(max_row)]
        forward_diag = [[] for _ in range(max_row + max_col - 1)]
        backward_diag = [[] for _ in range(len(forward_diag))]
        min_backwards = -max_row + 1

        for col in range(max_col):
            for row in range(max_row):
                cols[col].append(mat[row][col])
                rows[row].append(mat[row][col])
                forward_diag[row + col].append(mat[row][col])
                backward_diag[col - row - min_backwards].append(mat[row][col])

        diags = {
            'rows': rows,
            'cols': cols,
            'fdiags': forward_diag,
            'bdiags': backward_diag
        }

        return diags

    def clear(self, inplace: bool = False):
        """Cleans out the matrix and fill with 0s"""

        if inplace:
            self.mat = self.__mul__(other=0)
        else:
            return self.__mul__(other=0)


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
    mat3.sum(axis=1)
    print(mat3)
    mat3.reshape(dims=[3, 3, 1], inplace=True)
    print(mat3)
    mat3.transpose()
    print(mat3)
    mat3.transpose()
    print(mat3)

    mat4 = Array(init=[[1], [2], [3], [4]])
    print(mat4.shape())
    mat4.reshape(dims=[2, 2], inplace=True)
    print(mat4.flatten(inplace=True))
    print(mat4)

    mat0 = Array(init=[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    mat1 = Array(init=[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    mat2 = mat0 + mat1
    print(mat2)

    matminus = mat0 - mat1
    print(matminus)

    test1 = Array(init=[[2], [1], [1]])
    test2 = Array(init=[[10, 20, 30]])
    print(test1.shape())
    print(test2.shape())
    print(test1 * test2, '\t\tShape:', (test1 * test2).shape())

    for item, value in mat0.diagonal().items():
        print(item, value)

    test2.clear(inplace=True)
    print(test2)

    #ew = Array() is an error lol

    newnew = Array.arange(end=10, shape=[2, 5])
    print(newnew)

