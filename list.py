#!/usr/bin/env python
# -*- coding: UTF-8 -*
import os
import numpy as np

a=os.listdir('data/demo')
a.sort()
print(a)
size=len(a)
image_name = [1]*size
file=open('images.txt','w+')
file.close()
n=0
for i in range(size):
    file=open('images.txt','a+')
    file.write(a[n]+'\n')
    file.close()
    n=n+1
