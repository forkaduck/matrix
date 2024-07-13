#!/bin/python

from colorama import Back
import math

def index_print(array: list, highlights: list):
    colors = [Back.GREEN, Back.RED]
    colors_used = 0

    print("[", end = '')
    for i in range(0, len(array)):
        if i in highlights:
            print(colors[colors_used % len(colors)] + str(array[i]) + Back.RESET, end = '')
            colors_used += 1
        else:
            print(array[i], end = '')

        if i != len(array) - 1:
            print(', ', end = '')

    print("]")


data = []

for ix in range(0,10):
    data.append(5)

print(data)
print("------------------")


for k in range(0, int(math.sqrt(len(data))) + 1, 1):
    for ix in range(0, len(data), 2 << k):
        io = ix + (1 << k)

        index_print(data, [ix, io])

        if io < len(data):
            data[ix] += data[io]
            data[io] = 1


    print("------------------")

print(data)
