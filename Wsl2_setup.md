# 如何启动
## 检查文件
一个应该有7个文件:
```
    environment.yml
    apply_patch_0001.py
    apply_patch_0002.py
    apply_patch_0003.py
    Dockerfile
    Makefile
    setup.md
```
## setup
```shell
    # 创建环境
    conda env create -f environment1.yml
    conda env create -f environment1.yml
    conda activate RPL
    # 安装包
    cd residual-policy-learning/rpl_environments && pip install -e .
    cd ../..
    cd residual-policy-learning/src/baselines && pip install -e .
    cd ../../..
    cd residual-policy-learning/homestri-ur5e-rl && pip install -e .
    cd ../..
    # 打补丁
    python apply_patch_0001.py
    python apply_patch_0002.py
    python apply_patch_0003.py
    # 测试
    cd residual-policy-learning/tensorflow/experiment && python3 train_staged.py --env FetchPickAndPlace-v1 --n_epochs 100 --num_cpu 1 --config_path=configs/pickandplace.json --logdir logs/seed0/FetchPickAndPlace-v1 --seed 0 --random_eps=0.3 --controller_prop=0.8
```

# 修改窗口尺寸：
```    
找到：
    \home\joe\anaconda3\envs\py10-1\lib\python3.10\site-packages\glfw\__init__.py

    修改函数：
 
    def create_window(width, height, title, monitor, share):
    """
    Creates a window and its associated context.

    Wrapper for:
        GLFWwindow* glfwCreateWindow(int width, int height, const char* title, GLFWmonitor* monitor, GLFWwindow* share);
    """
    #修改窗口尺寸
    width=1600
    height=1400
    return _glfw.glfwCreateWindow(width, height, _to_char_p(title),
                                  monitor, share)
```