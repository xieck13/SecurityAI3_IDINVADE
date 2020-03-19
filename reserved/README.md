## 天池挑战者第三期 - Docker镜像模版

### 文件说明
./
├── README.md                                     | 这篇说明文档
├── benchmark_texts.txt                           | 示例输入文本，用来读取后生成提交结果
#### 和对抗样本生成逻辑有关的文件
├── example_module                                | 示例Python模块，可以替换为你自己的模块
├── requirements.txt                              | 在此填写需要的Python依赖包
├── main.py                                       | 镜像中执行的Python脚本
#### 可用于检查生成结果的格式，下文会提到
├── sanity_check.py                               | 用于检查生成的提交结果文件的脚本
#### Docker镜像所需文件，若无必要请勿修改
├── Dockerfile                                    | Docker配置文件
└── run.sh                                        | 执行Python脚本的shell脚本

### 怎么使用（此处假定已经熟悉了天池平台上的Docker练习场）

- (1) 在当前目录构建镜像
```bash
docker build -t registry.cn-<region>.aliyuncs.com/<username>/<repository>:<version> .
```
<region>: 你的阿里云容器服务区域，比如shanghai或shenzhen
<username>: 你的阿里云容器服务用户名，比如xiaoming
<repository>: 你的阿里云容器仓库名，比如tiaozhanzhe3
<version>: 你的容器镜像版本，比如0.1

- (2) 测试你的镜像
在你的某个可共享目录下，放置一个虚拟的评测文件，比如Mac上的/Users/<username>/ 目录可以链接给Docker来读取：
```bash
mkdir /Users/harry/tianchi_docs
mv benchmark_texts.txt /Users/harry/tianchi_docs/
```

准备好虚拟评测文件后，运行(1)中构建好的镜像，注意共享路径要链接到镜像中的/tcdata：
```bash
docker run \
-v <local/path/to/shared/directory>:/tcdata \
registry.cn-<region>.aliyuncs.com/<username>/<repository>:<version> \
sh -c "sh run.sh; head adversarial.txt; python sanity_check.py"
```
<local/path/to/shared/direcotry> 在上文的例子中，应当对应 /Users/harry/tianchi_docs

如果运行正常，你的镜像会读取虚拟评测文件，生成一个json格式的文本，并打印到你的屏幕。

- (3) 推送镜像到你的容器服务
```bash
docker push registry.cn-<region>.aliyuncs.com/<username>/<repository>:<version>
```

- (4) 在天池页面提交

如果对源码没有改动，或直接使用registry.cn-shanghai.aliyuncs.com/pepsimist/default:0.3，在比赛中应当能够顺利评测。

- (5) 加入你自己的文本变异逻辑，并重复步骤(1)-(4)

按照这份模版 原封不动地打包一个docker镜像，即可成功提交并跑通，也就是有了“交白卷”的能力。卷子上的内容靠大家各展所长，加油！
