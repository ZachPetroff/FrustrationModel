#!/usr/bin/python3
import sys
import os
import numpy
sys.argv.append('install')
if 'install' in sys.argv and not '--install-lib' in sys.argv:
    LOCAL_INSTALL = True
    sys.argv.append('--install-lib')
    sys.argv.append('.')
    if os.path.isdir('build'):
        DO_NOT_DELETE_BUILD = True
    else:
        DO_NOT_DELETE_BUILD = False
    if os.path.isfile('../align.so'):
        os.remove('../align.so')
else:
    LOCAL_INSTALL = False

from distutils.core import setup, Extension

extinctionbursts = Extension('extinctionbursts', sources=['agents.cpp', 'bodies.cpp', 'environments.cpp', 'simulate.cpp', 'pymodule.cpp'], language="c++", include_dirs = [numpy.get_include()])

setup(name = 'extinctionbursts', version='1.0', description='Aligns two graphs.', ext_modules = [extinctionbursts])

if LOCAL_INSTALL:
    import os, shutil
    os.remove('../extinctionbursts-1.0.egg-info')
    if not DO_NOT_DELETE_BUILD:
        shutil.rmtree('build')
    for filename in os.listdir('..'):
        if filename.startswith('extinctionbursts.') and filename.endswith('.so'):
            os.rename('../'+filename, '../extinctionbursts.so')
            break
