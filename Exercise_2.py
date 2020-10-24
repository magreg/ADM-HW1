# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 10:10:32 2020

@author: marco
"""

""" 
------------------------- EXERCISE 2 -----------------------------------------
"""

# Import packages
import math
import os
import random
import re
import sys


''' Birthday Cake Candles '''

def birthdayCakeCandles(candles):
    candles.sort()
    output = candles.count(candles[len(candles)-1])
    return output

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()


'''
-----------------------------------------------------------------
'''

''' Kangaroo (Number Line Jumps) '''

def kangaroo(x1, v1, x2, v2):

    if x2 > x1 and v2 > v1:
        return "NO"
    else:
        if v2-v1 == 0:
            return 'NO'
        else:
            output = (x1-x2) % (v2-v1)   
            if output == 0:
                return 'YES'
            else:
                return 'NO'


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    x1V1X2V2 = input().split()

    x1 = int(x1V1X2V2[0])

    v1 = int(x1V1X2V2[1])

    x2 = int(x1V1X2V2[2])

    v2 = int(x1V1X2V2[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()

'''
-----------------------------------------------------------------
'''

''' Strange Advertising (Viral Advertising) '''
# For this problem I have read some post in the discussion webpage of the Hackerrank
def viralAdvertising(n):
    shared = 5
    cumulative = 0
    for i in range(1, n + 1):
        liked = shared // 2
        shared = liked * 3
        cumulative += liked
    return cumulative



if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()
    
'''
-----------------------------------------------------------------
'''

''' Recursive digit sum '''
# For this problem I have read some post in the discussion webpage of the Hackerrank

def superDigit(n, k):
    digits = map(int, list(n))
    return get_digit(str(sum(digits) * k))
def get_digit(p):
    if len(p) == 1:
        return int(p)
    else:
        digits = map(int, list(p))
        return get_digit(str(sum(digits)))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nk = input().split()

    n = nk[0]

    k = int(nk[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()
    
'''
-----------------------------------------------------------------
'''

''' Inssertion Sort part 1 '''
# For this problem I have read some post in the discussion webpage of the Hackerrank

def insertionSort1(n, arr):
    n-=1
    currentvalue = arr[n]
    while n>0:
        if currentvalue > arr[n-1]:
            arr[n] = currentvalue
            print(*arr)
            break
        else:
            arr[n] =arr[n-1]
            print(*arr)
            
        n-=1
    else:
        arr[0] = currentvalue
        print(*arr)


if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)

'''
-----------------------------------------------------------------
'''

''' Insertion Sort part 2 '''
# For this problem I have read some post in the discussion webpage of the Hackerrank

def insertionSort2(n, arr):
    for n in range(1,n):
        currentvalue = arr[n]
        while n>0:
            if currentvalue > arr[n-1]:
                arr[n] = currentvalue
                print(*arr)
                break
            else:
                arr[n] =arr[n-1]
            n-=1
        else:
            arr[0] = currentvalue
            print(*arr)
        
if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)