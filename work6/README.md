# 计算机图形学实验报告
## 实验六：可微光栅化与网格优化

| 项目 | 内容 |
|------|------|
| 姓名 | 王宇畅 |
| 学号 | 202311030025 |
| 授课教师 | 张鸿文 |
| 助教 | 张怡冉 |
| 日期 | 2026年5月26日 |

---

## 一、项目架构

采用 Jupyter Notebook 与 PyTorch3D 结合的方式，实现基于可微渲染的3D网格优化：

```
CG-Lab/
├── work6/
   ├── main.ipynb              # 基础实验：剪影监督形状优化
   ├── main1.ipynb             # 选做实验：RGB监督纹理联合优化
   ├── cow.obj                 # 目标奶牛模型
   ├── cow_shape_only.obj      # 基础实验输出（形状）
   ├── cow_textured_final.obj  # 选做实验输出（彩色）
   ├── .gitignore
   └── README.md
```

**代码组织结构：**

```python
# 基础实验核心类
class ShapeOnlyOptimizer:
    - __init__()      # 初始化球体网格、软光栅化器
    - compute_loss()  # 剪影损失 + 正则化
    - optimize()      # 优化循环
    - visualize()     # 可视化剪影对比

# 选做实验核心类
class ShapeAndTextureOptimizer:
    - __init__()           # 初始化网格、顶点颜色参数
    - compute_loss()       # 剪影损失 + RGB损失 + 正则化 + 颜色平滑
    - optimize()           # 联合优化循环
    - visualize()          # 可视化剪影和RGB对比
```

---

## 二、核心代码逻辑

### 2.1 软光栅化配置（解决梯度消失）

```python
# 软光栅化渲染器设置
raster_settings = RasterizationSettings(
    image_size=256,
    blur_radius=np.log(1./1e-4 - 1.)*1e-4,  # 边缘模糊半径
    faces_per_pixel=50                       # 每像素多面提供梯度
)
shader = SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-4))
```

### 2.2 正则化损失（防止网格崩塌）

```python
# 三种正则化损失
loss_laplacian = mesh_laplacian_smoothing(mesh)  # 拉普拉斯平滑
loss_edge = mesh_edge_loss(mesh)                  # 边长一致性
loss_normal = mesh_normal_consistency(mesh)       # 法线一致性

# 总损失函数
total_loss = loss_silhouette + \
             1.0 * loss_laplacian + \
             0.1 * loss_edge + \
             0.01 * loss_normal
```

### 2.3 优化循环（基础实验）

```python
for epoch in range(epochs):
    # 变形网格
    deformed_mesh = src_mesh.offset_verts(deform_verts)
    
    # 渲染剪影
    pred_silhouette = shader(rasterizer(deformed_mesh))
    
    # 计算损失并反向传播
    loss = compute_loss(pred_silhouette, target_silhouette, deformed_mesh)
    loss.backward()
    optimizer.step()
```

### 2.4 联合纹理优化（选做实验）

```python
# 可优化参数：顶点位置 + 顶点颜色
self.deform_verts = torch.zeros_like(verts, requires_grad=True)
self.vertex_colors = torch.nn.Parameter(init_colors, requires_grad=True)

# 渲染RGB图像
pred_rgb = rgb_shader(rgb_rasterizer(deformed_mesh))

# 联合损失
loss = loss_silhouette + w_rgb * loss_rgb + w_color * loss_color_smooth
```

---

## 三、运行效果展示

### 3.1 基础实验效果

| 阶段 | 剪影损失 | 说明 |
|------|----------|------|
| 初始（Epoch 0） | 0.3255 | 球体剪影与奶牛差距大 |
| 中期（Epoch 150） | 0.0892 | 形状逐渐匹配 |
| 最终（Epoch 300） | 0.0767 | 成功变形为奶牛形状 |

**剪影优化过程：**
- 红色轮廓为预测剪影，灰色背景为目标剪影
- 随着迭代进行，红色轮廓逐渐与目标重合
<img width="938" height="680" alt="屏幕截图 2026-05-25 200312" src="https://github.com/user-attachments/assets/847bda9f-66d9-4545-802f-9fdd3115b9ee" />
<img width="940" height="382" alt="屏幕截图 2026-05-25 201036" src="https://github.com/user-attachments/assets/9c995f69-bff1-4a1f-8cc1-b05c51973958" />

### 3.2 选做实验效果

| 阶段 | 剪影损失 | RGB损失 | 说明 |
|------|----------|---------|------|
| 初始 | 0.2297 | 0.1823 | 灰色球体 |
| 中期 | 0.0605 | 0.0580 | 形状和颜色同时收敛 |
| 最终 | 0.0402 | 0.0346 | 彩色奶牛模型 |

**联合优化效果：**
- 形状逐渐从球体变为奶牛
- 颜色从灰色逐渐出现棕色渐变
- RGB图像与目标图像的差异不断减小
<img width="1361" height="1102" alt="屏幕截图 2026-05-25 201007" src="https://github.com/user-attachments/assets/dd790bd2-f41c-4d66-9e1c-6e4f4a8d3a33" />
<img width="1357" height="694" alt="屏幕截图 2026-05-25 201020" src="https://github.com/user-attachments/assets/068feef3-e5ca-4422-ba48-f6707eb4bc21" />

### 3.3 输出文件

| 文件 | 说明 |
|------|------|
| `cow_shape_only.obj` | 基础实验输出的奶牛形状模型 |
| `cow_textured_final.obj` | 选做实验输出的带颜色奶牛模型 |
| `cow_vertex_colors.npy` | 顶点颜色数据 |

---

## 四、关键技术要点

### 4.1 软光栅化原理

传统光栅化是不可微的，边界梯度为零。软光栅化通过Sigmoid函数产生平滑过渡：

$$A(d) = \text{sigmoid}\left(\frac{d}{\sigma}\right)$$

其中 $\sigma$ 控制边缘模糊程度，$\sigma$ 越大梯度传播范围越广。

### 4.2 正则化的重要性

仅靠图像损失会导致网格变成"刺猬"：

| 正则化类型 | 作用 |
|------------|------|
| 拉普拉斯平滑 | 约束相邻顶点，防止尖锐突起 |
| 边长惩罚 | 惩罚过长/过短边，防止三角形拉伸 |
| 法线一致性 | 约束相邻面法线接近，保持表面平滑 |

### 4.3 动态权重策略

```python
if epoch < 100:     # 前期：重形状
    w_sil, w_rgb = 1.0, 0.3
elif epoch < 200:   # 中期：平衡
    w_sil, w_rgb = 0.8, 0.5
else:               # 后期：重纹理
    w_sil, w_rgb = 0.5, 0.8
```

---

## 五、遇到的问题及解决方案

| 问题 | 解决方案 |
|------|----------|
| 梯度消失导致顶点不移动 | 使用 soft 光栅化，调整 blur_radius 参数 |
| 网格自交变成刺猬 | 增加正则化损失权重 |
| 颜色不连续 | 添加颜色平滑正则化 |
| save_obj不支持verts_colors | 自定义保存函数，写入带颜色的OBJ格式 |

---

## 六、实验总结

本次实验深入理解了可微渲染的核心原理：

1. **软光栅化**是解决渲染不可微问题的关键，通过在边界引入平滑过渡，使梯度能够传播到边缘外部的顶点。

2. **正则化**在网格优化中起决定性作用，没有正则化时网格会迅速崩坏，合理设置正则化权重才能获得光滑合理的结果。

3. **多视角监督**提供了足够的约束，使2D图像能够反推3D形状。

4. **联合优化**展示了可微渲染的扩展性，不仅可以优化形状，还可以同时优化纹理/颜色。

剪影损失从 0.3255 降至 0.0767，优化效果显著，成功将球体变形为奶牛形状。

## 七、Git仓库链接

🔗 **https://github.com/char-math/CG-Lab/tree/experience/work6**

---

**实验完成日期：2026年5月26日**
