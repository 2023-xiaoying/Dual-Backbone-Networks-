#/bin/bash

TF_CFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
# specify your cuda root directory here
CUDA_ROOT=/usr/local/cuda-11.7

$CUDA_ROOT/bin/nvcc ./tf_sampling_g.cu -o ./tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 -shared ./tf_sampling.cpp ./tf_sampling_g.cu.o -o ./tf_sampling_so.so -I $CUDA_ROOT/include -L $CUDA_ROOT/lib64/ -fPIC -I /home/hello/anaconda3/envs/RandLA-Net-master/lib/python3.6/site-packages/tensorflow_core/include -D_GLIBCXX_USE_CXX11_ABI=0 -L /home/hello/anaconda3/envs/RandLA-Net-master/lib/python3.6/site-packages/tensorflow_core -l:libtensorflow_framework.so.1 -fuse-ld=gold -O2

cd ../
