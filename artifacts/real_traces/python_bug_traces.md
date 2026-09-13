# Executed Python trace examples

## negative_max

Return the largest number in a non-empty list.

```python
def find_max(a):
    m = 0
    for x in a:
        if x > m:
            m = x
    return m

assert find_max([-5, -2, -8, -11, -3]) == -2
```

Expected: `-2`; actual: `0`; events: 15.

```text
#0 call L1 `def find_max(a):` [a=[-5, -2, -8, -11, -3]]
#2 line L3 `for x in a:` [a=[-5, -2, -8, -11, -3], m=0]
#3 line L4 `if x > m:` [a=[-5, -2, -8, -11, -3], m=0, x=-5]
#5 line L4 `if x > m:` [a=[-5, -2, -8, -11, -3], m=0, x=-2]
#7 line L4 `if x > m:` [a=[-5, -2, -8, -11, -3], m=0, x=-8]
#9 line L4 `if x > m:` [a=[-5, -2, -8, -11, -3], m=0, x=-11]
#11 line L4 `if x > m:` [a=[-5, -2, -8, -11, -3], m=0, x=-3]
#14 return L6 `return m` [a=[-5, -2, -8, -11, -3], m=0, x=-3] -> 0
OUTPUT 0
```

## factorial_endpoint

Return n factorial for a non-negative integer n.

```python
def factorial(n):
    result = 1
    for value in range(1, n):
        result *= value
    return result

assert factorial(6) == 720
```

Expected: `720`; actual: `120`; events: 15.

```text
#0 call L1 `def factorial(n):` [n=6]
#2 line L3 `for value in range(1, n):` [n=6, result=1]
#3 line L4 `result *= value` [n=6, result=1, value=1]
#5 line L4 `result *= value` [n=6, result=1, value=2]
#6 line L3 `for value in range(1, n):` [n=6, result=2, value=2]
#7 line L4 `result *= value` [n=6, result=2, value=3]
#8 line L3 `for value in range(1, n):` [n=6, result=6, value=3]
#9 line L4 `result *= value` [n=6, result=6, value=4]
#10 line L3 `for value in range(1, n):` [n=6, result=24, value=4]
#11 line L4 `result *= value` [n=6, result=24, value=5]
#12 line L3 `for value in range(1, n):` [n=6, result=120, value=5]
#14 return L5 `return result` [n=6, result=120, value=5] -> 120
OUTPUT 120
```

## first_index

Return the index of the first target value, or -1 when absent.

```python
def first_index(values, target):
    for index, value in enumerate(values):
        if value == target:
            return value
    return -1

assert first_index([8, 4, 9, 4, 3], 4) == 1
```

Expected: `1`; actual: `4`; events: 7.

```text
#0 call L1 `def first_index(values, target):` [target=4, values=[8, 4, 9, 4, 3]]
#2 line L3 `if value == target:` [index=0, target=4, value=8, values=[8, 4, 9, 4, 3]]
#4 line L3 `if value == target:` [index=1, target=4, value=4, values=[8, 4, 9, 4, 3]]
#6 return L4 `return value` [index=1, target=4, value=4, values=[8, 4, 9, 4, 3]] -> 4
OUTPUT 4
```

## count_even

Count how many integers in the list are even.

```python
def count_even(values):
    count = 0
    for value in values:
        if value % 2 == 1:
            count += 1
    return count

assert count_even([2, 7, 4, 9, 6, 11, 8]) == 4
```

Expected: `4`; actual: `3`; events: 22.

```text
#0 call L1 `def count_even(values):` [values=[2, 7, 4, 9, 6, 11, 8]]
#2 line L3 `for value in values:` [count=0, values=[2, 7, 4, 9, 6, 11, 8]]
#3 line L4 `if value % 2 == 1:` [count=0, value=2, values=[2, 7, 4, 9, 6, 11, 8]]
#5 line L4 `if value % 2 == 1:` [count=0, value=7, values=[2, 7, 4, 9, 6, 11, 8]]
#7 line L3 `for value in values:` [count=1, value=7, values=[2, 7, 4, 9, 6, 11, 8]]
#8 line L4 `if value % 2 == 1:` [count=1, value=4, values=[2, 7, 4, 9, 6, 11, 8]]
#10 line L4 `if value % 2 == 1:` [count=1, value=9, values=[2, 7, 4, 9, 6, 11, 8]]
#12 line L3 `for value in values:` [count=2, value=9, values=[2, 7, 4, 9, 6, 11, 8]]
#13 line L4 `if value % 2 == 1:` [count=2, value=6, values=[2, 7, 4, 9, 6, 11, 8]]
#15 line L4 `if value % 2 == 1:` [count=2, value=11, values=[2, 7, 4, 9, 6, 11, 8]]
#17 line L3 `for value in values:` [count=3, value=11, values=[2, 7, 4, 9, 6, 11, 8]]
#18 line L4 `if value % 2 == 1:` [count=3, value=8, values=[2, 7, 4, 9, 6, 11, 8]]
#21 return L6 `return count` [count=3, value=8, values=[2, 7, 4, 9, 6, 11, 8]] -> 3
OUTPUT 3
```

## clamp_upper

Clamp value so it lies in the inclusive interval [low, high].

```python
def clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return value
    return value

assert clamp(19, 0, 10) == 10
```

Expected: `10`; actual: `19`; events: 5.

```text
#0 call L1 `def clamp(value, low, high):` [high=10, low=0, value=19]
#4 return L5 `return value` [high=10, low=0, value=19] -> 19
OUTPUT 19
```

## average_denominator

Return the arithmetic mean of a non-empty list of numbers.

```python
def average(values):
    total = 0
    for value in values:
        total += value
    return total / (len(values) - 1)

assert average([2, 4, 6, 8, 10]) == 6.0
```

Expected: `6.0`; actual: `7.5`; events: 15.

```text
#0 call L1 `def average(values):` [values=[2, 4, 6, 8, 10]]
#2 line L3 `for value in values:` [total=0, values=[2, 4, 6, 8, 10]]
#3 line L4 `total += value` [total=0, value=2, values=[2, 4, 6, 8, 10]]
#4 line L3 `for value in values:` [total=2, value=2, values=[2, 4, 6, 8, 10]]
#5 line L4 `total += value` [total=2, value=4, values=[2, 4, 6, 8, 10]]
#6 line L3 `for value in values:` [total=6, value=4, values=[2, 4, 6, 8, 10]]
#7 line L4 `total += value` [total=6, value=6, values=[2, 4, 6, 8, 10]]
#8 line L3 `for value in values:` [total=12, value=6, values=[2, 4, 6, 8, 10]]
#9 line L4 `total += value` [total=12, value=8, values=[2, 4, 6, 8, 10]]
#10 line L3 `for value in values:` [total=20, value=8, values=[2, 4, 6, 8, 10]]
#11 line L4 `total += value` [total=20, value=10, values=[2, 4, 6, 8, 10]]
#12 line L3 `for value in values:` [total=30, value=10, values=[2, 4, 6, 8, 10]]
#14 return L5 `return total / (len(values) - 1)` [total=30, value=10, values=[2, 4, 6, 8, 10]] -> 7.5
OUTPUT 7.5
```
