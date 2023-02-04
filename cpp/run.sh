mkdir -p build
cd build
rm -rf ./*
cmake ..
make -j4
./naive_ptq_bin