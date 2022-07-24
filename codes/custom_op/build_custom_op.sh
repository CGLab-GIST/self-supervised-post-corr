TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

mkdir -p /usr/local/lib/python3.6/dist-packages/tensorflow/include/third_party/gpus/cuda/include \
  && cp -r /usr/local/cuda/targets/x86_64-linux/include/* \
  /usr/local/lib/python3.6/dist-packages/tensorflow/include/third_party/gpus/cuda/include


# [post_correction] file
nvcc -std=c++11 -c -o post_correction.cu.o post_correction.cu.cc \
  -I /usr/local ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr
g++ -std=c++11 -shared -o post_correction_lib.so post_correction.cc \
  -L /usr/local/cuda/lib64/ post_correction.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]}
cp post_correction_lib.so ./../models/post_correction_lib.so
rm post_correction.cu.o
rm post_correction_lib.so