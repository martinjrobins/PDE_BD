#!/usr/bin/python
import ctypes
ctypes.CDLL("libmpi.so.0", ctypes.RTLD_GLOBAL)
import sys
import pypdb
import numpy as np

pypdb.init(sys.argv)
dt = 0.001
dx = 0.1

p = pypdb.Pde(dt,dx)
p.add_particle(0.5,0.5,0.5)
for t in np.arange(0,0.1,dt):
    pypdb.write_grid("test%05d.pvtu" % (t/dt),p.get_grid())
    p.integrate(dt)