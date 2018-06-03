#!/usr/bin/env bash

mkdir include
mkdir lib

git clone https://github.com/pierfied/Chronos.git
cd Chronos
cmake .
make
mv libchronos.so ../lib
mv src/*.h ../include
cd ..
rm -rf Chronos

mkdir likelihood
cp -r src CMakeLists.txt likelihood
cd likelihood
ln -s ../include include
ln -s ../lib lib
cmake .
make
mv liblikelihood.so ../lib
mv src/*.h ../include
cd ..
rm -rf likelihood