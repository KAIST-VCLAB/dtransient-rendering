#!/bin/bash

# 1. Get Path-space Differentiable Renderer (PSDR) [Zhang et al. 2020]
wget https://shuangz.com/projects/psdr-sg20/psdr-sg20_code.zip
unzip psdr-sg20_code.zip
rm psdr-sg20_code.zip

# 2. Rename pypsdr to pydtrr
mv code/pypsdr code/pydtrr

# 3. Copy ours
cp -rf ./* code 2>/dev/null
mv code/code code/psdr-sg20_code
cd code

# 4. Install dependencies
mkdir -p dependencies
cd dependencies

git clone https://github.com/embree/embree.git
git clone https://github.com/pybind/pybind11.git

# Compile Embree
cd embree
mkdir -p build
cd build
cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DEMBREE_ISPC_SUPPORT=OFF ..
make install -j

# Compile Pybind11
cd ../../pybind11
mkdir -p build
cd build
cmake ..
make check -j
make install -j

cd ../../..
echo "================================================================"
echo "Installed at: $(pwd)"
echo "Go over the step 4 at the installation procedure using Docker."
echo "================================================================"

echo '''export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib''' >> ~/.bashrc
exec bash -l
