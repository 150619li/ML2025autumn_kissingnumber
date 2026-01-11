# 使用强化学习和蒙特卡洛树搜索求解接吻数问题

本项目维 2025 秋季学期机器学习大作业 Baseline2（MCTS+RL）

接吻数游戏在 ```Game.py``` 中实现。```Coach.py``` 包含核心训练循环，```MCTS.py``` 执行蒙特卡洛树搜索。自我对弈的参数可以在 ```main.py``` 中指定，包括维度、搜索空间、UCB公式的超参数等。神经网络架构在 ```NeuralNet.py``` 中实现。

## 安装

安装 conda 环境：

```
conda create --name kissingnumber python=3.10
conda activate kissingnumber
```

然后，安装我们的项目：

```
git clone https://github.com/150619li/ML2025autumn_kissingnumber
cd ML2025autumn_kissingnumber
pip install -r requirements.txt
```

开始搜索：
```bash
python main.py
```
## 实验

我们展示了不同模型在 $\mathrm{dim}=1,2,3,4,5,6,7,8,9$ 维接吻数问题上获得的最佳结果。详见我们的报告。

![image](assets/image.png)

## 致谢

本项目基于 [开源代码](https://github.com/YK-YoungK/ML_proj_KissingNumber) 开发，进行了代码效率提升以及更高维的搜索实验。