def matrix_dot_vector(a:list[list[int|float]],b:list[int|float])-> list[int|float]:
    result = []
    for i in range(len(a)):
      if len(a[i]) != len(b):
        return -1
      elem = 0
      for j in range(len(a[i])):
        elem += a[i][j] * b[j]
      result.append(elem)
    return result
