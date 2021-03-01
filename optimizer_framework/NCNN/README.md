ncnn framework

[ncnn](https://github.com/Tencent/ncnn) designed by Tencent. This framework optimized models Neural Network for mobile-platform.

# how to use ?
## build
  > ncnn support for multi-platform like android, ios, linux, raspberry pi, jetson nano, arm,...
and below guideline for linux platform. you can find something else in [original guideline](https://github.com/Tencent/ncnn#howto)
  - Step 1: clone repository
  
      `git clone https://github.com/Tencent/ncnn` && `cd ncnn`
  - Step 2: init submodule
  
      `git submodule update --init`
  - Step 3: build
    
      `mkdir build` && `cd build`
      
      `cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=ON -DNCNN_SYSTEM_GLSLANG=ON -DNCNN_BUILD_EXAMPLES=ON ..`
      
      `make -j$(nproc)` || `make -j4` - depend on number of thread
      
      `make install`
   - Step 4: get `include header file` & lib `libncnn.a` in folder `install`
## convert model to ncnn format (file.bin & file.param)
### pytorch
### tensorflow
### mxnet
