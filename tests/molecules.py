#!/usr/bin/python
import ctypes
ctypes.CDLL("libmpi.so.0", ctypes.RTLD_GLOBAL)
import sys
import pypdb
import numpy as np

pypdb.init(sys.argv)

dt = 0.001
D = 1.0;

m = pypdb.Molecules()
m.add_particle(0.5,0.5,0.5)
m.add_particle(0.5,0.5,0.5)
m.add_particle(0.5,0.5,0.5)
m.add_particle(0.5,0.5,0.5)
m.add_particle(0.5,0.5,0.5)
m.diffuse(dt,D)
for i in zip(m.get_x(),m.get_y(),m.get_z()):
    print i
m.remove_particle(1)
print '\n'
for i in zip(m.get_x(),m.get_y(),m.get_z()):
    print i

#print m.get_x()[0],',',m.get_y()[0],',',m.get_z()[0]