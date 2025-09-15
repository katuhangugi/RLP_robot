# 如何启动

## 检查文件
一个应该有6个文件:
```
    0001-baselines.patch
    0002-rpl.patch
    0003-.patch
    Dockerfile
    Makefile
    setup.md
```

## setup

```shell
#0
docker ps -a
docker start rpl-test
doocker exec =it -it rpl-test bash
#以交互式方式进入终端
docker start -i rpl-test

# 1. 编译镜像
# 如果遇到git下载失败的问题，把dockerfile中的`https://ghp.ci/https://github.com/`替换成其他的git镜像
make build-image

# 2. 启动容器
make start-container

# 后面都是在容器内的操作

# 3. 下载代码
# 遇到git下载失败问题同上
cd code
make clone-code

# 4. 代码打补丁
make patch-code

# 5. 安装依赖
make deps

# 6. 执行训练测试
make test-train
```
