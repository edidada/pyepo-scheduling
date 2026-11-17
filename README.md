# pyepo-scheduling

一个**调度问题**的具体应用案例：用 **Predict-then-Optimize（先预测、后优化）** 方法做
**云作业到时间槽的调度**，并对比「两阶段 MSE」与「决策导向 SPO+」两种训练方式。

方法论对应 [PyEPO](https://github.com/PyEPO/PyEPO) 的 SPO+ 损失（Elmachtoub & Grigas,
*Smart Predict-then-Optimize*, 2022）。本项目把其中的优化 oracle 换成 Hungarian
（指派问题）求解器，因此**不依赖 Gurobi / PyEPO 运行时**，可以直接在 GitHub Actions 上
安装依赖并跑通；如需接入 PyEPO，只需把 `scheduling/optimizer.py` 的 oracle 换成对应的
`pyepo` optModel 即可。

## 具体场景：一天的批处理作业调度

云厂商每天要把一批 `n` 个作业排进 `n` 个时间槽：

- 每个时间槽 `t` 有一个**已知**电价 `q[t]`（峰时更贵）；
- 每个作业 `j` 的运行时长 `p[j]` **事先未知**，但与可观测特征 `X[j]`（作业类别、
  输入规模、内存需求……）相关；
- 把作业 `j` 放进槽 `t` 还会产生一个**已知**的 setup / 数据搬运成本 `A[j, t]`。

目标是把作业分配到各不相同的槽（一个排列 `sigma`），使总成本最小：

```
C(sigma) = sum_j ( q[sigma(j)] * p[j] + A[j, sigma(j)] )
cost[j, t] = q[t] * p[j] + A[j, t]
```

这是一个线性指派问题（用 `scipy.optimize.linear_sum_assignment` 求解），也是一个货真价实的
调度决策。流程是：**从特征预测运行时长 → 构造成本矩阵 → 求解调度 → 用真实运行时长评估
决策后悔值（regret）**。

关键点：真实运行时长是特征的**非线性**函数，而预测器是线性的（模型错配）。这正是
「只顾预测误差」与「顾及决策质量」会产生分歧的场景。

## 两种训练策略

| 策略 | 做法 |
| --- | --- |
| 两阶段（two-stage / MSE） | 用岭回归拟合运行时长，最小化预测误差，完全不管下游决策 |
| 决策导向（SPO+） | 直接对 SPO+ 损失做梯度下降，梯度为 `2*(omega*(c) - omega*(2*chat-c))`，经链式法则回传到预测器参数 |

SPO+ 损失是决策后悔值的凸上界；训练中通过 Hungarian oracle 两次求解（真实成本 `c` 与
扰动成本 `2*chat - c`）得到次梯度。评估指标为测试集上的**归一化后悔值**（越低越好）。

## 目录结构

```
scheduling/
  data.py       合成数据：非线性真实运行时长、电价、setup 成本
  optimizer.py  Hungarian 指派 oracle（omega*(c)）
  loss.py       SPO+ 损失与次梯度
  train.py      两阶段岭回归 / SPO+ 全批量梯度下降
  evaluate.py   归一化 regret 评估
main.py         跑完整案例并打印对比表
tests/          pytest 单测
.github/workflows/ci.yml   GitHub Actions：装依赖、编译、测试、运行
```

## 运行

```
pip install -r requirements.txt
python -m compileall -q scheduling tests main.py
python -m pytest -q
python main.py
```

## CI / 本地不编译

本项目**不在本地编译或运行**，验证全部交给 GitHub Actions：每次 `push` / `pull_request`
（也可 `workflow_dispatch` 手动触发）都会安装依赖、`compileall` 字节编译、跑 pytest，
再执行 `main.py`。查看结果：

```
gh run list
gh run watch <run-id> --exit-status
gh run view <run-id> --log
```
