# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 11:04:07 2018

@author: Vette
"""
import os, shutil


orig_data_set_dir = r'C:\Users\Vette\Desktop\Regis\#MSDS686 Deep Learning\kaggle_cats_dogs\train'

base_dir = r'C:\Users\Vette\Desktop\Regis\#MSDS686 Deep Learning\cats_dogs'
#os.mkdir(base_dir)

# setup dirs
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir= os.path.join(base_dir, 'test')
os.mkdir(test_dir)
test2_dir= os.path.join(base_dir, 'test2')
os.mkdir(test2_dir)

train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

test2_cats_dir = os.path.join(test2_dir, 'cats')
os.mkdir(test2_cats_dir)

test2_dogs_dir = os.path.join(test2_dir, 'dogs')
os.mkdir(test2_dogs_dir)
i=0
# now do files
i=i+1
print('working:', i)
fnames = ['cat.{}.jpg'.format(i) for i in range(0, 8000)]
for fname in fnames:
    src = os.path.join(orig_data_set_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copy(src, dst)

i=i+1
print('working:', i)    
fnames = ['cat.{}.jpg'.format(i) for i in range(8000, 9500)]
for fname in fnames:
    src = os.path.join(orig_data_set_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copy(src, dst)

i=i+1
print('working:', i)
fnames = ['cat.{}.jpg'.format(i) for i in range(9500, 11000)]
for fname in fnames:
    src = os.path.join(orig_data_set_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copy(src, dst)

i=i+1
print('working:', i)
fnames = ['cat.{}.jpg'.format(i) for i in range(11000, 12500)]
for fname in fnames:
    src = os.path.join(orig_data_set_dir, fname)
    dst = os.path.join(test2_cats_dir, fname)
    shutil.copy(src, dst)

i=i+1
print('working:', i)
fnames = ['dog.{}.jpg'.format(i) for i in range(0, 8000)]
for fname in fnames:
    src = os.path.join(orig_data_set_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copy(src, dst)   

i=i+1
print('working:', i)
fnames = ['dog.{}.jpg'.format(i) for i in range(8000, 9500)]
for fname in fnames:
    src = os.path.join(orig_data_set_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copy(src, dst)

i=i+1
print('working:', i)
fnames = ['dog.{}.jpg'.format(i) for i in range(9500, 11000)]
for fname in fnames:
    src = os.path.join(orig_data_set_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copy(src, dst)    

i=i+1
print('working:', i)
fnames = ['dog.{}.jpg'.format(i) for i in range(11000, 12500)]
for fname in fnames:
    src = os.path.join(orig_data_set_dir, fname)
    dst = os.path.join(test2_dogs_dir, fname)
    shutil.copy(src, dst)        
    
    
    
    