---
title: Flow Matching 初探
description: Flow Matching 入门
lastUpdated: true
---

根据我们在前两篇文章的经验，确定一个 Diffusion Model 主要包括三个要素：

1. 前向过程，即 ```x_t = add_noise(x_0, noise, t)```
2. 反向过程，即 ```x_{t-1} = step(model_output, t, x)```
3. 训练目标，即 ```loss_fn(model_output, gt_t)```，需要确定 ```gt_t```

在这三要素确定后我们便很容易地能够实现 Diffusion Model 的训练及采样过程：

``` python showLineNumbers
# 训练过程
for x_0, c in batch:
    # 时间步及噪声采样
    t = torch.randint(0, num_train_timesteps, (batch_size,))
    noise = torch.randn_like(x_0)

    # 前向过程
    x_t = add_noise(x_0, noise, t)
    
    # 模型预测
    model_output = model(x_t, t, c)

    # 确定 gt_t
    gt_t = ...
    
    # 计算 loss
    loss = loss_fn(model_output, gt_t)
    
    # 模型更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 采样过程
x = torch.randn(x_size)  # 初始噪声

for t in range(num_sample_timesteps):
    # 模型预测
    model_output = model(x, t, c)
    
    # 反向过程
    x = step(model_output, t, x)
```