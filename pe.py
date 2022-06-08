##################
### Question 3 ###
##################

## Task A ###
def row_sum(matrix):
    return list(map(lambda row: sum(row), matrix))


def col_sum(matrix):
    # print(matrix)
    if matrix == [] or matrix[0] == []:
        return []

    return [sum(map(lambda x: x[0], matrix))] + col_sum(list(map(lambda x: x[1:], matrix)))


#
# no error checking needed
#
def get_shape(lst):
    shape = [len(lst)]
    tmp = lst[0]

    while isinstance(tmp, list):
        shape.append(len(tmp))
        tmp = tmp[0]

    return shape


def get_value(lst, idx):
    if len(idx) == 1:
        return lst[idx[0]]
    else:
        return get_value(lst[idx[0]], idx[1:])


def set_value(lst, idx, val):  # same as get_value.....
    if len(idx) == 1:
        lst[idx[0]] = val
        return
    else:
        return set_value(lst[idx[0]], idx[1:], val)


def create_arr(shape):
    if len(shape) == 1:
        return [0] * shape[0]
    else:
        result = [[]] * shape[0]
        for i in range(shape[0]):
            # print(result[i])
            result[i] = create_arr(shape[1:])

        return result


# get_shape(create_lst([3,4]) gives [3,4] with all values are 0

def next_idx(idx, shape):
    if len(shape) == 0:
        return None  # reach the max already
    else:
        if idx[-1] == shape[-1] - 1:
            temp = next_idx(idx[:-1], shape[:-1])
            if temp is None:  # need this to avoid error...boundary condition can check.
                return None
            else:
                return temp + [0]
        else:
            idx[-1] += 1
            return idx


def test(shape):
    idx = [0] * len(shape)
    while idx is not None:
        idx = next_idx(idx, shape)
        print(idx)

    return


# sum along - tts version

def sum_along(axis, lst):  # axis starts with zero...need to check row and column
    # setting up
    shape = get_shape(lst)
    s = shape.pop(axis)  # remind them

    if len(shape) == 0:  # this is a 1D problem
        return sum(lst)

    result = create_arr(shape)
    rIdx = [0] * len(shape)

    while rIdx is not None:
        val = 0

        for i in range(s):
            idx = rIdx.copy()
            idx.insert(axis, i)
            val += get_value(lst, idx)

        set_value(result, rIdx, val)
        rIdx = next_idx(rIdx, shape)

    return result


##################
### Question 4 ###
##################
class Matrix(object):

    ## Task A ###
    def __init__(self, nrows, ncols):
        self.dict = {}
        self.nrows = nrows
        self.ncols = ncols

    def get(self, idx):
        return self.dict.get(idx, 0)

    def insert(self, idx, val):
        self.dict[idx] = val

    def delete(self, idx):
        self.dict.pop(idx)

    def dict2list(self):
        res = [[0] * self.ncols for _ in range(self.nrows)]
        for idx in self.dict:
            res[idx[0]][idx[1]] = self.dict[idx]
        return res

    ## Task B ###
    def transpose(self):
        output = Matrix(self.ncols, self.nrows)
        for (idx, value) in self.dict.items():
            output.insert((idx[1], idx[0]), value)

        return output

    ## Task C ###

    def multiply(self, m2):
        output = Matrix(self.nrows, m2.ncols)
        for (i, k) in self.dict:
            for j in range(m2.ncols):
                if (k, j) in m2.dict:
                    output.insert((i, j), output.get((i, j)) + self.get((i, k)) * m2.get((k, j)))

        return output

    ## For debug ###
    def __str__(self):
        return f'{self.nrows} rows, {self.ncols} cols, {self.dict}'
