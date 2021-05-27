# source lib
source="https://github.com/Tencent/ncnn/releases/download/20210124/"
# download ncnn lib for ubuntu 20.04
lib_file="ncnn-20210124-ubuntu-2004-shared.zip"
lib_dir="ncnn"
build_dir="build"
source_path="$source$lib_file"
filename="$( cut -d '.' -f 1 <<< "$lib_file")"; echo "lib file: $filename"
#download lib file
wget $source_path
# unzip ncnn lib
mkdir ncnn
unzip $lib_file
# copy header file and lib ncnn
cp -r $filename/* ncnn
# combine inference
mkdir build
cd build
cmake ..
make -j$(nproc)
