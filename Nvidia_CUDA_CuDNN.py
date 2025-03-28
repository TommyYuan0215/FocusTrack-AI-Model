import tensorflow as tf

print("TensorFlow CUDA Version:", tf.sysconfig.get_build_info()["cuda_version"])
print("TensorFlow cuDNN Version:", tf.sysconfig.get_build_info()["cudnn_version"])
print("CUDA Available:", tf.test.is_built_with_cuda())

print("GPU Devices:", tf.config.list_physical_devices('GPU'))