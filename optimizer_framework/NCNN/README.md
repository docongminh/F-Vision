ncnn framework

  > [ncnn](https://github.com/Tencent/ncnn) designed by Tencent. This framework optimized models Neural Network for mobile-platform.
  > ncnn support for multi-platform like android, ios, linux, raspberry pi, jetson nano, arm,...
and below guideline for linux platform. you can find something else in [original guideline](https://github.com/Tencent/ncnn#howto)

# how to use ?
## build
### re-build framework
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
### use framework built for specify palatform
    > lib built by Tencent team public in [release list](https://github.com/Tencent/ncnn/releases)
    > Download lib for your platform same re-build process
## convert model to ncnn format (file.bin & file.param)
### pytorch
  > Overview flow as [ncnn wiki](https://github.com/Tencent/ncnn/wiki/use-ncnn-with-pytorch-or-onnx): `pytorch model` -> `onnx` -> `ncnn`
  > Add on more tools support convert model [PytorchConverter](https://github.com/starimeL/PytorchConverter), [brocolli](https://github.com/inisis/brocolli)
  - Step 1: Convert Pytorch to Onnx
    ```python
        import torch
        import torch.onnx
        
        # Model loaded weight(.pth, .pkl or pretrain in torchvision)
        model = Model()
        
        # input model for forward(0 method
        input = torch.rand(1, 3, 112, 112)
        
        # export model
        output = torch.onnx._export(model, input, "onnx_converted.onnx", export_params=True)
  - Step 2: Fix redundant operators
    > In ncnn wiki mentioned `model may contains many redundant operators such as Shape, Gather, Unsqueeze that us not supported in ncnn
    ```
      Shape not supported yet!
      Gather not supported yet!
        # axis=0
      Unsqueeze not supported yet!
        # axes 7
      Unsqueeze not supported yet!
        # axes 7
    ```
    To solve it. Use handy [tool](https://github.com/daquexian/onnx-simplifier) developed by danquexian
    `python3 -m onnxsim onnx_converted.onnx onnx_removed.onnx`
  - Step 3: Convert onnx to ncnn
  > [Use tools convert by ncnn](https://github.com/docongminh/F-Vision/tree/master/optimizer_framework/NCNN#use-framework-built-for-specify-patlform) to convert onnx model to ncnn model.
    `./onnx2ncnn onnx_removed.onnx ncnn_model.param ncnn_model.bin`
  
### tensorflow
   > Tools support [ckpt2ncnn](https://github.com/hanzy88/ckpt2ncnn) or [tensorflow2ncnn](https://github.com/hanzy88/tensorflow2ncnn)
### mxnet
  > ncnn supported convert mxnet to ncnn. Use ncnn tools
  `./mxnet2ncnn model-symbol.json model-0000.params ncnn_model.bin ncnn_model.param`
