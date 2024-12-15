---
title: lerobot 前期准备~
date: 2024-12-16 21:00:00
categories: 
- robot
- AI
toc: true
tags:
- lerobot
---

## 安装mini conda
- 计划使用macos macmini 现在也比较便宜
```shell
mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

- linux 版本
```shell 
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

> 安装之后记得关闭 然后再打开终端
```shell 
source ~/miniconda3/bin/activate
# 然后初始化conda
conda init --all
```


## 创造conda虚拟环境
```shell
conda create -n lerobot python=3.8
conda activate lerobot
```

## 拉取仓库
```shell
# 拉取克隆
git clone https://github.com/huggingface/lerobot.git ~/lerobot
```

## 安装依赖项
```shell 
cd ~/lerobot && pip install -e ".[feetech]"
```

> 如果是linux (mac不需要)
```shell 
conda install -y -c conda-forge ffmpeg
pip uninstall -y opencv-python
conda install -y -c conda-forge "opencv>=4.10.0"
```

## 配置电机
这一步是为了寻找usb接口
```shell
python lerobot/scripts/find_motors_bus_port.py
```
> 插入电机之后 运行脚本 然后断开 按一次enter就好 会显示出你的接口号 一定要记住了 记不住也没关系 再运行一次就好 : )

## 配置电机的ID
```shell
python lerobot/scripts/configure_motor.py \
  --port 你的端口文件上面那个案例显示出来的 \
  --brand feetech \
  --model sts3215 \
  --baudrate 1000000 \
  --ID 1
```
这样依次 一直到6 另外6个电机也这样重复操作
领导电机的齿轮 在编号之后需要拆除 齿轮 详细请看视频

## 组装
此处大力出奇迹

- 主臂 leader arm
```shell 
python lerobot/scripts/control_robot.py calibrate \
    --robot-path lerobot/configs/robot/so100.yaml \
    --robot-overrides '~cameras' --arms main_leader
```

- 从臂 follow arm 
```shell  
python lerobot/scripts/control_robot.py calibrate \
    --robot-path lerobot/configs/robot/so100.yaml \
    --robot-overrides '~cameras' --arms main_follower
```
此处做个区分

## 简单操作
```shell
# 不会连接到摄像头
python lerobot/scripts/control_robot.py teleoperate \
    --robot-path lerobot/configs/robot/so100.yaml \
    --robot-overrides '~cameras' \
    --display-cameras 0
```

## 买个摄像头 然后连接 有iphone最好
