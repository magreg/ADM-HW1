# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 09:36:49 2020

@author: marco
"""

""" 
------------------------- EXERCISE 1 -----------------------------------------
"""

''' INTRODUCIOTN '''

# Import packages
import math
import os
import random
import re
import sys

''' Ex 1 : Say "Hello, World!" with Python''' 

print("Hello, World!")

''' Ex 2 : Python if-else'''

n = int(input().strip())

if (n % 2 != 0):
    print('Weird')
elif ((n % 2 == 0) and ((n >= 2) and (n <= 5))):
    print('Not Weird')
elif ((n % 2 == 0) and ((n >= 6) and (n <= 20))):
    print('Weird')
elif ((n % 2 == 0) and (n > 20)):
    print('Not Weird')

''' Ex 3 : Arithmetic Operators'''

if __name__ == '__main__':
    a = int(input())
    b = int(input())

print(a+b)
print(a-b)
print(a*b)

''' Ex 4 : Python Division'''

if __name__ == '__main__':
    a = int(input())
    b = int(input())

print(a // b)
print(a / b)

''' Ex 5 : Loops'''

if __name__ == '__main__':
    n = int(input())

p = 0
i = 0
while i < n:
        p = i ** 2
        i = i + 1
        print(p)
        
''' Ex 6 : Write a Function'''

def is_leap(year):
    if year % 400 == 0:
        return True
    if year % 100 == 0:
        return False
    if year % 4 == 0:
        return True
    else:
        return False

''' Ex 7 : Print a Function'''

if __name__ == '__main__':
    n = int(input())

for i in range(1, n+1):
    print(i, end='')
    

'''
-----------------------------------------------------------------
'''


''' DATA TYPES'''

''' Ex 1 : Lists''' 
if __name__ == '__main__':
    N = int(input())

Result = [];
for i in range(0,N):
    num = input().split()
    if num[0] == "print":
        print(Result)
    elif num[0] == "insert":
        Result.insert(int(num[1]),int(num[2]))
    elif num[0] == "remove":
        Result.remove(int(num[1]))
    elif num[0] == "pop":
        Result.pop()
    elif num[0] == "append":
        Result.append(int(num[1]))
    elif num[0] == "sort":
        Result.sort()
    else:
        Result.reverse()

''' Ex 2 : List Comprehensions'''

if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())

listaX = range(x+1)
listaY = range(y+1)
listaZ = range(z+1)

coordinates = [[i, j, k] for i in listaX for j in listaY for k in listaZ if i + j + k != n]

print(coordinates)

''' Ex 3 : Find the Runner-Up Score!'''

if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())

    arr_set = set(arr)
    arr_list = list(arr_set)
    arr_list.sort(reverse = True)
    print(arr_list[1])

''' Ex 4 : Nested Lists'''
# For this problem I have read some post in the discussion webpage of the Hackerrank

List = []

if __name__ == '__main__':
    for _ in range(int(input())):
        name = input()
        score = float(input())
        List.append([name, score])

second_highest = sorted(set([score for name, score in List]))[1]
print('\n'.join(sorted([name for name, score in List if score == second_highest])))

''' Ex 5 : Finding the percentage'''

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores        
    query_name = input()

student = list(student_marks[query_name])
avg = sum(student) / len(student)
print('%.2f' % avg)

''' Ex 6 : Tuples'''

if __name__ == '__main__':
    n = int(raw_input())
    integer_list = map(int, raw_input().split())

t = tuple(integer_list)
print(hash(t))


'''
-----------------------------------------------------------------
'''


''' STRINGS '''

''' Ex 1 : sWAP cASE''' 

def swap_case(s):
    result = s.swapcase()
    return result

''' Ex 2 : String Split and Join'''

def split_and_join(line):
    return "-".join(line.split())

''' Ex 3 : What's your name?'''

def print_full_name(a, b):
        print(f"Hello {a} {b}! You just delved into python." )

''' Ex 4 : Mutations'''

def mutate_string(string, position, character):
    string = string[:position] + character + string[(position+1):]
    return string

''' Ex 5 : Find a string'''

def count_substring(string, sub_string):
    count = 0
    for j in range(len(string) - len(sub_string) + 1):
        if string[j:j+len(sub_string)] == sub_string:
           count += 1
    return count

''' Ex 6 : String Validators'''

if __name__ == '__main__':
    s = input()

print(any(i.isalnum() for i in s))
print(any(i.isalpha() for i in s))
print(any(i.isdigit() for i in s))
print(any(i.islower() for i in s))
print(any(i.isupper() for i in s))

''' Ex 7 : Text Alignment'''

#Replace all ______ with rjust, ljust or center. 

thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

''' Ex 8 : Text Wrap'''

def wrap(string, max_width):
    result = textwrap.fill(string, width = max_width)
    return result

''' Ex 9 : Designer Door Mat'''
# For this problem I have read some post in the discussion webpage of the Hackerrank

N, M = map(int, input().split())

point = '.|.'

for i in range(1, N, 2): 
    points = point * i
    print(points.center(M, '-'))

print('WELCOME'.center(M, '-'))

for i in range(N - 2, -1, -2): 
    points = point * i
    print(points.center(M, '-'))

''' Ex 10 : String Formatting'''

def print_formatted(number):
    w = len(str(bin(number)).replace('0b',''))
    for i in range(1,number+1):   
        d = str(i).rjust(w,' ')
        b = bin(i)[2:].rjust(w,' ')  
        o = oct(i)[2:].rjust(w, ' ')
        h = hex(i)[2:].rjust(w, ' ').upper()
        print(d, o, h, b)

''' Ex 11 : Alphabet Rangoli'''
# For this problem I have read some post in the discussion webpage of the Hackerrank

def print_rangoli(size):
    
    width  = size*4-3
    string = ''

    for i in range(1,size+1):
        for j in range(0,i):
            string += chr(96+size-j)
            if len(string) < width :
                string += '-'
        for k in range(i-1,0,-1):    
            string += chr(97+size-k)
            if len(string) < width :
                string += '-'
        print(string.center(width,'-'))
        string = ''

    for i in range(size-1,0,-1):
        string = ''
        for j in range(0,i):
            string += chr(96+size-j)
            if len(string) < width :
                string += '-'
        for k in range(i-1,0,-1):
            string += chr(97+size-k)
            if len(string) < width :
                string += '-'
        print(string.center(width,'-'))
        
''' Ex 12 : Capitalize'''

def solve(s):
    for i in s.split():
        s = s.replace(i,i.capitalize())
    return s

''' Ex 13 : The Minion game'''
# For this problem I have read some post in the discussion webpage of the Hackerrank

def minion_game(string):
    
    consonant = 0
    vowel = 0
    vowel_list = ['a','e','i','o','u','A','E','I','O','U']
    n = len(string)
 
    for i, y in enumerate(string):
        if y in vowel_list:
            vowel += n-i
        else:
            consonant += n-i
 
    if vowel == consonant:
        print ("Draw")
    elif vowel > consonant:
        print ("Kevin {}".format(vowel))
    else:
        print ("Stuart {}".format(consonant))

''' Ex 14 : Merge the Tools'''
# For this problem I have read some post in the discussion webpage of the Hackerrank

def merge_the_tools(string, k):

    subsegments_numbers = int(len(string) / k)

    for i in range(subsegments_numbers):
        
        y = string[i * k : (i + 1) * k]
        u = ""
        for c in y:
            if c not in u:
                u += c
        print(u)


'''
-----------------------------------------------------------------
'''


''' SETS '''

''' Ex 1 : Introduction to Sets''' 

def average(array):
    sum_a = sum(set(array))
    len_a = len(set(array))
    avg = sum_a/len_a
    return avg

''' Ex 2 : No idea!'''

n = input().split()
m = map(int, n)
num = map(int, input().split())
A = set(map(int, input().split()))
B = set(map(int, input().split()))

happy = 0
for i in num:
    if i in A:
        happy += 1
    elif i in B:
        happy -= 1

print (happy)

''' Ex 3 : Symmetric Difference'''

M = set(input())
M_set = set(map(int, input().split()))
N = set(input())
N_set = set(map(int, input().split()))

union = M_set.union(N_set)
intersection = M_set.intersection(N_set)
symmetric_difference = union - intersection
for i in sorted(list(symmetric_difference)):
    print(i)

''' Ex 4 : Set.add()'''

stamps = set()
n = int(input())

for _ in range(n):
    stamps.add(input())

print (len(stamps))

''' Ex 5 : Set.discard(), .remove() & .pop()'''

n = int(input())
s = set(map(int, input().split()))

m = int(input())

for _ in range(m):
    a = input().split(" ")
    if (a[0] == "pop"):
        s.pop()
    elif (a[0] == 'remove'):
        s.remove(int(a[1]))
    elif (a[0] == 'discard'):
        s.discard(int(a[1]))
print (sum(s))

''' Ex 6 : Set.union() Operation'''

e_num = int(input())
eng_set = set(map(int, input().split()))

f_num = int(input())
fren_set = set(map(int, input().split()))

print(len(eng_set.union(fren_set)))

''' Ex 7 : Set.intersection() Operation '''

eng_num = int(input())
eng_set = set(map(int, input().split()))

fren_num = int(input())
fren_set = set(map(int, input().split()))

print(len(eng_set.intersection(fren_set)))

''' Ex 8 : Set.difference() Operation'''

eng_num = int(input())
eng_set = set(map(int, input().split()))

fren_num = int(input())
fren_set = set(map(int, input().split()))

result = eng_set.union(fren_set) - fren_set

print(len(result))

''' EX 9 : Set.symmetric_difference() Operation'''

eng_num = int(input())
eng_set = set(map(int, input().split()))

fren_num = int(input())
fren_set = set(map(int, input().split()))

result = eng_set.symmetric_difference(fren_set)
print(len(result))

''' EX 10 : Set Mutations'''

s=set(map(int, input().split()))
N=int(input())

for i in range(N):
    (p, q)=input().split()
    s2=set(map(int, input().split()))
    if p=='intersection_update':
        s.intersection_update(s2)
    elif p=='update':
        s.update(s2)
    elif p=='symmetric_difference_update':
        s.symmetric_difference_update(s2)
    elif p=='difference_update':
        s.difference_update(s2)
print (sum(s))

''' EX 11 : The Capitain's Room'''
# For this problem I have read some post in the discussion webpage of the Hackerrank

K = int(input())
room_List = map(int,input().split())
room = sorted(room_List) 

for i in range(1,len(room)):
    if i != len(room)-1:
        if room[i] != room[i-1] and room[i] != room[i+1]:
            print(room[i])
            break
    else:
        print(room[i])

''' EX 12 : Check Subset'''

for _ in range(int(input())):
    a = int(input())
    A = set(input().split())
    b = int(input())
    B = set(input().split())
    
    if A.union(B) == B:
        print("True")
    else:
        print("False")
        
''' EX 13 : Check Strict Superst'''
A  = set(input().split())
n = int(input())

for i in range(n):
    s = set(input().split())
    if (s & A != s) or (s == A):
        result = False
        break
    else:
        result = True
print(result)


'''
-----------------------------------------------------------------
'''

''' COLLECTIONS '''

# Import packages
import math
import os
import random
import re
import sys

from collections import Counter
from collections import defaultdict
from collections import namedtuple
from collections import OrderedDict
from collections import deque

''' Ex 1 : collections.Counter()''' 

X = int(input())
sizes = Counter(map(int, input().strip().split()))
N = int(input())

earn = 0
for i in range(N):
    size, price = map(int, input().split())
    if sizes[size]:
        earn += price
        sizes[size] -= 1
print(earn)

''' Ex 2 : DefaultDictTutorial'''

d = defaultdict(list)

m, n = map(int, input().split())

for a in range(1, m + 1):
    d[input().strip()].append(str(a))

for a in range(n):
    b = input().strip()
    if not d[b]:
        print (-1)
    else:
        print (' '.join(d[b]))

''' Ex 3 : Collections.namedtuple()'''

n = int(input())
school = ','.join(input().split())
Student = namedtuple('Student',school)

sum = 0
for i in range(n):
    row = input().split()
    student = Student(*row)
    sum += int(student.MARKS)

print(sum/n)

''' Ex 4 : Collections.OrderDict()'''

N = int(input())
d = OrderedDict()

for i in range(N):
    obj = input().split()
    Price = int(obj[-1])
    Name = " ".join(obj[:-1])
    if(d.get(Name)):
        d[Name] += Price
    else:
        d[Name] = Price
for i in d.keys():
    print(i, d[i])

''' Ex 5 : Word Order'''
# For this problem I have read some post in the discussion webpage of the Hackerrank

N = int(input())
Obj = OrderedDict()

for i in range(N):
    word = input()
    if word in Obj:
        Obj[word] +=1
    else:
        Obj[word] = 1

print(len(Obj));

for k,v in Obj.items():
    print(v,end = " ")

''' Ex 6 : Collections.deque()'''

Obj = deque()

for i in range(int(input())):
    s = input().split()
    if s[0] == 'append':
        Obj.append(s[1])
    elif s[0] == 'appendleft':
        Obj.appendleft(s[1])
    elif s[0] == 'pop':
        Obj.pop()
    else:
        Obj.popleft()

print (" ".join(Obj)) 

''' Ex 7 : Company Logo'''
# For this problem I have read some post in the discussion webpage of the Hackerrank

s = sorted(input().strip())
s_counter = Counter(s).most_common()
s_counter = sorted(s_counter, key=lambda x: (x[1] * -1, x[0]))
for i in range(0, 3):
    print(s_counter[i][0], s_counter[i][1])

''' Ex 8 : Piling Up!'''

# I did not resolve this exercise because I did not understand how to solve it 
# even after reading some post in the discussion webpage.



'''
-----------------------------------------------------------------
'''

''' DATE AND TIME '''

# Import packages
import calendar
import math
import os
import random
import re
import sys
from datetime import datetime

''' Ex 1 : Calendar Module''' 

m, d, y = map(int, input().strip().split())

day = calendar.weekday(y, m, d)

Output = calendar.day_name[day]

print(Output.upper())

''' Ex 2 : Time Delta'''

def time_delta(t1, t2):
    t1 = datetime.strptime(t1, '%a %d %b %Y %H:%M:%S %z')
    t2 = datetime.strptime(t2, '%a %d %b %Y %H:%M:%S %z')
    result = str(int(abs((t1-t2).total_seconds())))
    return result

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())
    
    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)

        fptr.write(delta + '\n')

    fptr.close()


'''
-----------------------------------------------------------------
'''

''' EXCEPTIONS '''


''' Ex 1 : Exceptions''' 
# For this problem I have read some post in the discussion webpage of the Hackerrank

for i in range(int(input())):
    try:
        a,b = map(int,input().split())
        integer_division = a // b
        print(integer_division)
    except ZeroDivisionError as e:
        print("Error Code:", e)
    except ValueError as e:
        print("Error Code:", e)


'''
-----------------------------------------------------------------
'''

''' BUIL-INS '''

# Import packages
import math
import os
import random
import re
import sys

''' Ex 1 : Zipped!''' 
# For this problem I have read some post in the discussion webpage of the Hackerrank
n, m = map(int, input().split())

grade = []
for _ in range(m):
    grade += [input().strip().split()]

for s in zip(*grade):
    print (format(sum(map(float, s)) / m, '.1f'))

''' Ex 2 : Athlete Sort'''
# For this problem I have read some post in the discussion webpage of the Hackerrank

n, m = map(int, input().split())
rows = [input() for _ in range(n)]
k = int(input())

for row in sorted(rows, key=lambda row: int(row.split()[k])):
    print(row)
''' Ex 3 : ginortS'''
# For this problem I have read some post in the discussion webpage of the Hackerrank

print(*sorted(input(), key=lambda c: (c.isdigit() - c.islower(), c in '02468', c)), sep='')


'''
-----------------------------------------------------------------
'''

''' PYTHON FUNCTIONS '''

''' Ex 1 : Map and Lambda Expression ''' 

cube = lambda x: x ** 3 

def fibonacci(n):
    if n<2:
        return range(n)
    a = [0,1]
    for i in range(0,n-2):
        Sum =a[i+1]+a[i]
        a.append(Sum)
    return(a)

'''
-----------------------------------------------------------------
'''

''' REGEX AND PARSING CHALLENGES '''

# Import Packages
import re
import email.utils
from html.parser import HTMLParser

''' Ex 1 : Detect FLoating Point Number''' 

for _ in range(int(input())):
    print(bool(re.match(r'^[-+]?[0-9]*\.[0-9]+$', input())))

''' Ex 2 : re.split()'''

regex_pattern = r"[.,]+"

''' Ex 3 : group(), groups() & groupdict()'''

string = re.search(r'([a-zA-Z0-9])\1', input().strip())
print(string.group(1) if string else -1)

''' Ex 4 : re.findall() & re.finditer()'''

v = 'aeiou'
c = 'qwrtypsdfghjklzxcvbnm'

output = re.findall(r'(?<=[' + c + '])([' + v + ']{2,})(?=[' + c + '])', input(), flags=re.I)
result = '\n'.join(output or ['-1'])
print(result)

''' Ex 5 : re.start() & re.end()'''

String, Sub_String = input(), input()
mat = re.finditer(r'(?=(' + Sub_String + '))', String)

any_match = False
for i in mat:
    any_match = True
    print((i.start(1), i.end(1) - 1))

if any_match == False:
    print((-1, -1))

''' Ex 6 : Regex Substitution'''

for i in range(int(input())):
    string = ''
    string = re.sub(r'(?<= )&&(?= )','and',input())
    result = re.sub(r'(?<= )\|\|(?= )','or',string)
    print(result)

''' Ex 7 : Validating Roman Numerals'''
# For this problem I have read some post in the discussion webpage of the Hackerrank

regex_pattern = r"M{0,3}(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[VX]|V?I{0,3})$"

''' Ex 8 : Validating phone numbers'''
# For this problem I have read some post in the discussion webpage of the Hackerrank

n = int(input())
pattern = r"^[789]\d{9}$"    

for i in range(n):
    num = input()
    if(len(num)==10 and num.isdigit()):
        output = re.findall(pattern,num)
        if(len(output)==1):
            print("YES")
        else:
            print("NO")
    else:
        print("NO")

''' Ex 9 : Validating and Parsing Email Addresses'''
# For this problem I have read some post in the discussion webpage of the Hackerrank

n = int(input())

pattern = r'^[a-z][\w\-\.]+@[a-z]+\.[a-z]{1,3}$'
for i in range(0, n):
    addr = email.utils.parseaddr(input())
    if re.search(pattern, addr[1]):
        print(email.utils.formataddr(addr)) 

''' Ex 10 : HTML Parser - part 1'''
# For this problem I have read some post in the discussion webpage of the Hackerrank

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print ('Start :', tag)
        for i in attrs:
            print ('->', i[0], '>', i[1])

    def handle_endtag(self, tag):
        print ('End   :', tag)

    def handle_startendtag(self, tag, attrs):
        print ('Empty :', tag)
        for i in attrs:
            print ('->', i[0], '>', i[1])

parser = MyHTMLParser()
for _ in range(int(input())):
    parser.feed(input())

''' Ex 11 : HTML Parser - part 2'''
# For this problem I have read some post in the discussion webpage of the Hackerrank

class MyHTMLParser(HTMLParser):
    def handle_comment(self, comment):
        if '\n' in comment:
            print('>>> Multi-line Comment')
        else:
            print('>>> Single-line Comment')
        print(comment)

    def handle_data(self, data):
        if data == '\n': return
        print('>>> Data')
        print(data)
  
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()

''' Ex 12 : Hex Color Code'''
# I did not resolve this exercise because I did not understand how to solve it 
# even after reading some post in the discussion webpage.

''' Ex 13 : Detect HTML Tags, Attributes and Attribute Values'''
# I did not resolve this exercise because I did not understand how to solve it 
# even after reading some post in the discussion webpage.

''' Ex 14 : Validating UID'''
# I did not resolve this exercise because I did not understand how to solve it 
# even after reading some post in the discussion webpage.

''' Ex 15 : Validating Credit Card Numbers'''
# I did not resolve this exercise because I did not understand how to solve it 
# even after reading some post in the discussion webpage.

''' Ex 16 : 'Validating Postal Codes'''
# I did not resolve this exercise because I did not understand how to solve it 
# even after reading some post in the discussion webpage.

''' Ex 17 : Matrix Scripts'''
# I did not resolve this exercise because I did not understand how to solve it 
# even after reading some post in the discussion webpage.


'''
-----------------------------------------------------------------
'''

''' XML'''

''' Ex 1 : Find the Score''' 

def get_attr_number(node):
    score = len(node.attrib) + sum(get_attr_number(child) for child in node)
    return score

''' Ex 2 : Find the Maximum Depth'''

maxdepth = 0
def depth(elem, level):
    global maxdepth
    if (level == maxdepth):
        maxdepth += 1

    for child in elem:
        depth(child, level + 1)


'''
-----------------------------------------------------------------
'''

''' CLOSURES AND DECORATIONS '''

''' Ex 1 : Standardize Mobile Number using Decorators''' 
# For this problem I have read some post in the discussion webpage of the Hackerrank

def wrapper(f):
    def fun(l):
        f(['+91 ' + i[-10:-5] + ' ' + i[-5:] for i in l])
    return fun

''' Ex 2 : Name Directory'''
# For this problem I have read some post in the discussion webpage of the Hackerrank

def person_lister(f):
    def inner(people):
        output = map(f, sorted(people, key=lambda x: int(x[2])))
        return output
    return inner

'''
-----------------------------------------------------------------
'''

''' NUMPY '''

# Import packages
import numpy

''' Ex 1 : Arrays''' 

def arrays(arr):
    a = numpy.array(arr, float)
    result = a[::-1]
    return result

''' Ex 2 : Shape and Reshape'''

array = numpy.array(input().split(), int)
arr = numpy.array(array)
new_arr = numpy.reshape(arr,(3,3))
print(new_arr)

''' Ex 3 : Transpose and Flatten'''

n, m = map(int, input().split())
array = numpy.array([input().strip().split() for _ in range(n)], int)
transpose = array.transpose()
flatten = array.flatten()
print (transpose)
print (flatten)

''' Ex 4 : Concatenate'''

n, m, p = map(int, input().split())
array_a = numpy.array([input().split() for _ in range(n)],int)
array_b = numpy.array([input().split() for _ in range(m)],int)
result = numpy.concatenate((array_a, array_b), axis = 0)
print(result)

''' Ex 5 : Zeros and Ones'''

shape= tuple(map(int,input().split()))
z = numpy.zeros(shape,int)
o = numpy.ones(shape,int)
print(z, o, sep='\n')

''' Ex 6 : Eye and Identity'''

numpy.set_printoptions(sign=' ')
n, m = map(int, input().split())
matrix = numpy.eye(n, m)
print(matrix)

''' Ex 7 : Array Mathematics'''

a = []
b = []

n, m = map(int, input().split())
for _ in range(n):
    a.append(input().split())
for _ in range(n):
    b.append(input().split())

A = numpy.array(a, int)
B = numpy.array(b, int)

add = numpy.add(A,B)
sub = numpy.subtract(A,B)
mul = numpy.multiply(A,B)
div = numpy.floor_divide(A,B)
mod = numpy.mod(A,B)
po = numpy.power(A,B)

print(add)
print(sub)
print(mul)
print(div)
print(mod)
print(po)

''' Ex 8 : Floor, Ceil and Rint'''

numpy.set_printoptions(sign=' ')
arr = input().strip().split(' ')
A = numpy.array(arr, float)

print(numpy.floor(A))
print(numpy.ceil(A))
print(numpy.rint(A))

''' Ex 9 : Sum and Prod'''

n, m = map(int, input().split())
array = numpy.array([input().strip().split() for _ in range(n)], int)

somma = numpy.sum(array, axis=0)
result = numpy.prod(somma)
print(result)

''' Ex 10 : Min and Max'''

n, m = map(int, input().split())
array = numpy.array([input().strip().split() for _ in range(n)], int)

mini = numpy.min(array, axis=1)
result = numpy.max(mini)
print(result)

''' Ex 11 : Mean, Var, and Std'''

numpy.set_printoptions(legacy='1.13')
n, m = map(int, input().split())
array = numpy.array([input().strip().split() for _ in range(n)], int)

Media = numpy.mean(array, axis = 1)
Varianza = numpy.var(array, axis = 0)
Dev_standard = numpy.std(array, axis = None)

print(Media)
print(Varianza)
print(Dev_standard)

''' Ex 12 : Dot and Cross'''

n = int(input())
A = numpy.array([input().split() for i in range(n)], int)
B = numpy.array([input().split() for i in range(n)], int)
result = numpy.dot(A,B)
print(result)

''' Ex 13 : Inner and Outer'''

A = numpy.array(input().split() , int)
B = numpy.array(input().split() , int)

I = numpy.inner(A, B)
O = numpy.outer(A, B)
print(I)
print(O)

''' Ex 14 : Polynomials'''

m = numpy.array(input().split(), float)
n = float(input())

result = numpy.polyval(m, n)
print(result)

''' Ex 15 : Linear Algebra'''

numpy.set_printoptions(legacy='1.13')

n = int(input())
A = numpy.array([input().split() for i in range(n)], float)

det_A = numpy.linalg.det(A)
print(det_A)


'''
-----------------------------------------------------------------
'''