# Compiles Cpp files.
cd ../src/cpp
g++ -O3 -fopenmp main.cpp -o solver.out
mv solver.out ../../bin