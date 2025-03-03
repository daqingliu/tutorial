---
title: Flow Matching 初探
description: Flow Matching 入门
lastUpdated: true
---

## 回顾：Diffusion 三要素

根据我们在前两篇文章的经验，确定一个 Diffusion Model 主要包括三个要素：

1. 前向过程，即 ```x_t = add_noise(x_0, noise, t)```
2. 反向过程，即 ```x_{t-1} = step(model_output, t, x)```
3. 训练目标，即 ```loss_fn(model_output, gt_t)```，需要确定 ```gt_t```

在这三要素确定后我们便很容易地能够实现 Diffusion Model 的训练及采样过程：

``` python
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

## Flow Matching 极简理解

> 注：本文中的 Flow Matching 特指论文中 Optimal Transport 的情况（即 Rectified Follow）

针对 Diffusion 三要素，Flow Matching 实际上是在 DDPM 的基础上借助做了大幅的简化，从而得到了更好的性质与表现。

### 前向过程

首先是针对前向过程，其本身是从 $p_{data}$ 到 $p_{prior}$ 的概率的变化轨迹，在 FM 中被定义为了 'Flow'，这个概念其实很容易理解，就像是概率分布在流动变化一样。它将整个过程定义在了 $[0,1]$ 的区间中，即 $x_1 \sim p_{data}$，$x_0 \sim p_{prior} = \mathcal{N}(0, I)$，与 DDPM 差不多是反过来了。

前向过程中，FM 的目标是将其定义为最简单的形式：从 $p_{data}$ 到 $p_{prior}$ 是走了一条笔直且匀速的直线：

$$
x_t = t*x_1 + (1-t)*x_0
$$

``` python
def add_noise(x_1, noise, t):
    x_0 = noise
    x_t = t * x_1 + (1 - t) * x_0
    return x_t
```

### 反向过程

既然前向过程是一条笔直匀速的直线，那理想的反向过程应当也是一条笔直匀速的直线，我们仅需要确定每一个 step 的速度即可，我们将其定义为 $v_t$，在每个 $dt$ 时间步内：

$$
x_{t+dt} = x_t + v_t * dt
$$

这里 $v_t$ 我们定义为模型的预测值，有：

``` python
def step(model_output, t, x):
    v_t = model_output
    dt = 1 / num_sample_steps
    x_new = x + v_t * dt
    return x_new
```

### 训练目标

我们的训练目标仍然是为了让模型预测值与真实值逼近，因此仍采用 MSE Loss。而真实值就是理想反向过程的速度，对 $t$ 时刻求导易得：

$$
v_t = \frac{dx_t}{dt} = x_1 - x_0
$$

```python
def get_gt(x_1, noise, t):
    x_0 = noise
    return x_1 - x_0

loss_fn = nn.MSELoss()
```

注：这里 gt 实际上已经与 x_0, x_1, t 无关了

## Flow Matching 的代码实现

### MNIST数据集与UNet网络

这里我们使用 MNIST 数据集，并直接复用上一篇文章中网络结构：


```python
import torch
import torchvision
from torch import nn
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from diffusers import UNet2DModel


# 加载数据集
dataset = load_dataset("mnist")

# 图像预处理
image_preprocess = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1)
])

def transform(examples):
    image = [image_preprocess(image) for image in examples["image"]]
    # 标签转换为long tensor
    label = [torch.tensor(l, dtype=torch.long) for l in examples["label"]]
    return {"image": image, "label": label}

# 应用预处理
train_dataset = dataset['train'].with_transform(transform)
```

    Parameter 'transform'=<function transform at 0x7f6b33254860> of the transform datasets.arrow_dataset.Dataset.set_format couldn't be hashed properly, a random hash was used instead. Make sure your transforms and parameters are serializable with pickle or dill for the dataset fingerprinting and caching to work. If you reuse this transform, the caching mechanism will consider it to be different from the previous calls and recompute everything. This warning is only showed once. Subsequent hashing failures won't be showed.



```python
class ClassConditionedUnet(nn.Module):
    def __init__(self, num_classes=10, class_emb_size=4):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, class_emb_size)

        # self.model 是一个无条件的 UNet，使用额外的输入通道以接受条件信息（类别嵌入）
        self.model = UNet2DModel(
            sample_size=28,  # 图像分辨率
            in_channels=1 + class_emb_size,  # 额外的输入通道用于类别条件
            out_channels=1,  # 输出通道数，灰度图
            layers_per_block=2,
            block_out_channels=(32, 64, 64),
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
        )

    def forward(self, x, t, class_labels):
        bs, ch, w, h = x.shape

        class_cond = self.class_emb(class_labels)
        class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)

        net_input = torch.cat((x, class_cond), 1)

        return self.model(net_input, t).sample  # (bs, 1, 28, 28)
```

### FlowMatchScheduler

这里我们 follow diffusers 的实践，将与 diffusion 框架相关的代码都放在 Scheduler 类中。

> A scheduler takes a model’s output (the sample which the diffusion process is iterating on) and a timestep to return a denoised sample.
> A scheduler defines how to iteratively add noise to an image or how to update a sample based on a model’s output:
> - during *training*, a scheduler adds noise (there are different algorithms for how to add noise) to a sample to train a diffusion model
> - during *inference*, a scheduler defines how to update a sample based on a pretrained model’s output

为了和 diffusers 一致，部分输入输出的参数命名进行了微调。


```python
class FlowMatchScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
    ):
        self.num_train_timesteps = num_train_timesteps
        
    def add_noise(
        self,
        sample: torch.FloatTensor,
        timestep: int,
        noise: torch.FloatTensor,
    ) -> torch.FloatTensor:
        x_0 = noise
        t = timestep / self.num_train_timesteps
        x_1 = sample

        while len(t.shape) < len(x_1.shape):
            t = t.unsqueeze(-1)
            
        x_t = t * x_1 + (1 - t) * x_0
        return x_t
    
    def step(
        self,
        model_output: torch.FloatTensor,
        sample: torch.FloatTensor,
        num_sample_timesteps: int = 50,
    ) -> torch.FloatTensor:
        v_t = model_output
        dt = 1.0 / num_sample_timesteps
        x = sample
        x_new = x + v_t * dt
        return x_new
```

### 模型训练

整体大的结构和上篇文章基本无异


```python
batch_size = 256
n_epochs = 40
num_train_timesteps = 1000
num_sample_timesteps = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

noise_scheduler = FlowMatchScheduler(num_train_timesteps=num_train_timesteps)
model = ClassConditionedUnet(num_classes=10+1, class_emb_size=8).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Keeping a record of the losses for later viewing
losses = []

# The training loop
for epoch in range(n_epochs):
    for batch in tqdm(train_dataloader):
        # Get some data and prepare the corrupted version
        image = batch['image'].to(device)
        label = batch['label'].to(device)
        
        # 用于 CFG 训练
        mask = torch.rand_like(label, dtype=torch.float) < 0.2
        label = label+1
        label[mask] = 0
        
        noise = torch.randn_like(image)
        timestep = torch.randint(0, num_train_timesteps, (image.shape[0],)).long().to(device)
        noisy_image = noise_scheduler.add_noise(image, timestep, noise)

        # Get the model prediction
        pred = model(noisy_image, timestep, label)
        
        # Calculate the loss
        loss = loss_fn(pred, image-noise)

        # Backprop and update the params:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store the loss for later
        losses.append(loss.item())
        
    if epoch in [0, 5, 9, 19 ,29, 39]:
        # Prepare random x to start from, plus some desired labels y
        x = torch.randn(44, 1, 28, 28).to(device)
        y = torch.tensor([range(11)]*4).flatten().to(device)
        
        # 设定指导因子，通常设为4.0左右
        guidance_scale = 4.0

        # 创建空标签（根据模型训练时的设定调整）
        # 假设训练时用0表示空条件，若不同需替换为对应值
        y_empty = torch.zeros_like(y)  # 保持与y相同的形状和类型

        # Sampling loop
        for t in range(num_sample_timesteps):
            # 扩展输入为两倍批次（条件 + 非条件）
            x_in = torch.cat([x] * 2)
            # 拼接条件与空标签
            y_in = torch.cat([y, y_empty])
            
            # 获取模型预测（同时计算条件和非条件）
            with torch.no_grad():
                residual = model(x_in, t, y_in)
            
            # 拆分为条件预测和非条件预测
            cond_residual, uncond_residual = residual.chunk(2)
            
            # 应用CFG公式：混合预测结果
            guided_residual = uncond_residual + guidance_scale * (cond_residual - uncond_residual)
            
            # 更新样本
            x = noise_scheduler.step(guided_residual, x, num_sample_timesteps)
        
        # Show the results
        img = (-x.detach().cpu().clip(-1, 1) + 1) / 2.0
        fig, ax = plt.subplots(1, 1, figsize=(11, 4))
        ax.imshow(torchvision.utils.make_grid(img, nrow=11)[0], cmap="Greys")
        ax.axis('off')
        plt.show()

    # Print out the average of the last 100 loss values to get an idea of progress:
    avg_loss = sum(losses[-100:]) / 100
    print(f"Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}")
    
fig, ax = plt.subplots(1, 1, figsize=(11, 4))
plt.plot(losses)
plt.show()  

```

      0%|          | 0/234 [00:00<?, ?it/s]

    100%|██████████| 234/234 [00:18<00:00, 12.61it/s]



    
![](https://image-1304830922.cos.ap-shanghai.myqcloud.com/20250303232302980.png)
    


    Finished epoch 0. Average of the last 100 loss values: 0.231267


    100%|██████████| 234/234 [00:18<00:00, 12.56it/s]


    Finished epoch 1. Average of the last 100 loss values: 0.200926


    100%|██████████| 234/234 [00:18<00:00, 12.64it/s]


    Finished epoch 2. Average of the last 100 loss values: 0.188842


    100%|██████████| 234/234 [00:18<00:00, 12.65it/s]


    Finished epoch 3. Average of the last 100 loss values: 0.184601


    100%|██████████| 234/234 [00:18<00:00, 12.66it/s]


    Finished epoch 4. Average of the last 100 loss values: 0.180163


    100%|██████████| 234/234 [00:18<00:00, 12.66it/s]



    
![](https://image-1304830922.cos.ap-shanghai.myqcloud.com/20250303232320579.png)
    


    Finished epoch 5. Average of the last 100 loss values: 0.177932


    100%|██████████| 234/234 [00:18<00:00, 12.55it/s]


    Finished epoch 6. Average of the last 100 loss values: 0.175908


    100%|██████████| 234/234 [00:18<00:00, 12.64it/s]


    Finished epoch 7. Average of the last 100 loss values: 0.174406


    100%|██████████| 234/234 [00:18<00:00, 12.64it/s]


    Finished epoch 8. Average of the last 100 loss values: 0.171955


    100%|██████████| 234/234 [00:18<00:00, 12.64it/s]



    
![](https://image-1304830922.cos.ap-shanghai.myqcloud.com/20250303232334717.png)
    


    Finished epoch 9. Average of the last 100 loss values: 0.171124


    100%|██████████| 234/234 [00:18<00:00, 12.63it/s]


    Finished epoch 10. Average of the last 100 loss values: 0.170358


    100%|██████████| 234/234 [00:18<00:00, 12.64it/s]


    Finished epoch 11. Average of the last 100 loss values: 0.168870


    100%|██████████| 234/234 [00:18<00:00, 12.56it/s]


    Finished epoch 12. Average of the last 100 loss values: 0.169351


    100%|██████████| 234/234 [00:18<00:00, 12.64it/s]


    Finished epoch 13. Average of the last 100 loss values: 0.168222


    100%|██████████| 234/234 [00:18<00:00, 12.65it/s]


    Finished epoch 14. Average of the last 100 loss values: 0.168372


    100%|██████████| 234/234 [00:18<00:00, 12.63it/s]


    Finished epoch 15. Average of the last 100 loss values: 0.167132


    100%|██████████| 234/234 [00:18<00:00, 12.65it/s]


    Finished epoch 16. Average of the last 100 loss values: 0.167101


    100%|██████████| 234/234 [00:18<00:00, 12.65it/s]


    Finished epoch 17. Average of the last 100 loss values: 0.166592


    100%|██████████| 234/234 [00:18<00:00, 12.57it/s]


    Finished epoch 18. Average of the last 100 loss values: 0.165176


    100%|██████████| 234/234 [00:18<00:00, 12.64it/s]



    
![](https://image-1304830922.cos.ap-shanghai.myqcloud.com/20250303232358256.png)
    


    Finished epoch 19. Average of the last 100 loss values: 0.165992


    100%|██████████| 234/234 [00:18<00:00, 12.64it/s]


    Finished epoch 20. Average of the last 100 loss values: 0.165668


    100%|██████████| 234/234 [00:18<00:00, 12.64it/s]


    Finished epoch 21. Average of the last 100 loss values: 0.164040


    100%|██████████| 234/234 [00:18<00:00, 12.64it/s]


    Finished epoch 22. Average of the last 100 loss values: 0.165244


    100%|██████████| 234/234 [00:18<00:00, 12.64it/s]


    Finished epoch 23. Average of the last 100 loss values: 0.164972


    100%|██████████| 234/234 [00:18<00:00, 12.56it/s]


    Finished epoch 24. Average of the last 100 loss values: 0.163400


    100%|██████████| 234/234 [00:18<00:00, 12.65it/s]


    Finished epoch 25. Average of the last 100 loss values: 0.163531


    100%|██████████| 234/234 [00:18<00:00, 12.65it/s]


    Finished epoch 26. Average of the last 100 loss values: 0.164280


    100%|██████████| 234/234 [00:18<00:00, 12.64it/s]


    Finished epoch 27. Average of the last 100 loss values: 0.163436


    100%|██████████| 234/234 [00:18<00:00, 12.65it/s]


    Finished epoch 28. Average of the last 100 loss values: 0.163196


    100%|██████████| 234/234 [00:18<00:00, 12.58it/s]



    
![](https://image-1304830922.cos.ap-shanghai.myqcloud.com/20250303232411244.png)
    


    Finished epoch 29. Average of the last 100 loss values: 0.161450


    100%|██████████| 234/234 [00:18<00:00, 12.65it/s]


    Finished epoch 30. Average of the last 100 loss values: 0.162166


    100%|██████████| 234/234 [00:18<00:00, 12.64it/s]


    Finished epoch 31. Average of the last 100 loss values: 0.162726


    100%|██████████| 234/234 [00:18<00:00, 12.64it/s]


    Finished epoch 32. Average of the last 100 loss values: 0.161698


    100%|██████████| 234/234 [00:18<00:00, 12.64it/s]


    Finished epoch 33. Average of the last 100 loss values: 0.162693


    100%|██████████| 234/234 [00:18<00:00, 12.64it/s]


    Finished epoch 34. Average of the last 100 loss values: 0.160825


    100%|██████████| 234/234 [00:18<00:00, 12.54it/s]


    Finished epoch 35. Average of the last 100 loss values: 0.161347


    100%|██████████| 234/234 [00:18<00:00, 12.65it/s]


    Finished epoch 36. Average of the last 100 loss values: 0.161966


    100%|██████████| 234/234 [00:18<00:00, 12.65it/s]


    Finished epoch 37. Average of the last 100 loss values: 0.162540


    100%|██████████| 234/234 [00:18<00:00, 12.65it/s]


    Finished epoch 38. Average of the last 100 loss values: 0.160515


    100%|██████████| 234/234 [00:18<00:00, 12.65it/s]



    
![](https://image-1304830922.cos.ap-shanghai.myqcloud.com/20250303232432539.png)
    


    Finished epoch 39. Average of the last 100 loss values: 0.161052



![](https://image-1304830922.cos.ap-shanghai.myqcloud.com/20250303232445954.png)
    


## 参考

1. [Flow Matching | Explanation + PyTorch Implementation - Outlier](https://www.youtube.com/watch?v=7cMzfkWFWhI)
2. [扩散模型流匹配（Flow Matching）真实面目揭秘 - 童发发](https://www.bilibili.com/video/BV1E7ykYqEHd)
3. [TongTong313/rectified-flow](https://github.com/TongTong313/rectified-flow/tree/main)
4. [diffusers/schedulers/scheduling_flow_match_euler_discrete.py](https://github.com/huggingface/diffusers/blob/v0.32.2/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py)
5. [schedulers - diffusers](https://huggingface.co/docs/diffusers/api/schedulers/overview)
