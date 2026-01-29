# 环境配置
```bash
conda create -n pinn python=3.10
conda activate pinn
pip install torch --index-url https://download.pytorch.org/whl/cu118 #请按需下载对应cuda版本
pip install -r requirements.txt
```
# 使用方法
运行[main.py](src/main.py)，其中变量`noise`表示数据中加入噪声的比例；`epochs_adam`表示训练轮数；`model_id`表示使用的模型，`1`为Baseline，`2`为resnet；`experiment_name`会被在加输出的图表的名字后。