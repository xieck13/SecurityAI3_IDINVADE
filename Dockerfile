# Base Images
## 从天池基础镜像构建
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/python:3

 
## 把当前文件夹里的文件构建到镜像的根目录下
ADD ./reference_model /reference_model/
ADD ./distance_module /distance_module/
ADD ./reserved /
## 在构建镜像时安装依赖包
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pybind11
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple sklearn
ADD ./stopwords /stopwords
ADD main.py /

## 指定默认工作目录为根目录（需要把run.sh和生成的结果文件都放在该文件夹下，提交后才能运行）

WORKDIR /

## 镜像启动后统一执行 sh run.sh
CMD ["sh", "run.sh"]
