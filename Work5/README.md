## 实验五：光线追踪基础与GPU实现

| 项目 | 内容 |
|------|------|
| **姓名** | 王宇畅 |
| **学号** | 202311030025 |
| **授课教师** | 张鸿文 |
| **助教** | 张怡冉 |
| **日期** | 2026年5月12日 |

---

## 一、项目架构

采用标准的 src 布局，实现代码与配置的物理隔离：

```
Work5/
└── src/          # 演示用gif
    ├── main.py
    └── main1.py        # 备用版本
├── README.md        # 说明文档
└── assets/          # 演示用gif
    ├── w5_0.gif
    └── w5_1.gif
```

---

## 二、代码逻辑

### 2.1 场景定义

场景包含三个几何体：

| 物体 | 位置 | 半径 | 材质 | 颜色 |
|------|------|------|------|------|
| 玻璃球 | (-1.2, 0, 0) | 1.0 | 玻璃 | 透明 |
| 镜面球 | (1.2, 0, 0) | 1.0 | 镜面反射 | 银色 |
| 地面 | y = -1.0 | 无限大 | 漫反射 | 棋盘格纹理 |

### 2.2 核心函数结构

```
render() [GPU并行入口]
    └── trace_ray() [迭代光线追踪]
            ├── scene_intersect() [场景求交]
            │       ├── intersect_sphere()
            │       └── intersect_plane()
            ├── MAT_DIFFUSE → shade_diffuse() [漫反射着色+阴影]
            ├── MAT_MIRROR → reflect() [镜面反射]
            └── MAT_GLASS → fresnel() + refract() [玻璃材质]
```

### 2.3 光线追踪流程

```python
# 迭代式光线追踪（替代递归）
for bounce in range(max_bounces):
    t, N, albedo, mat_id = scene_intersect(ro, rd)
    
    if mat_id == MAT_DIFFUSE:
        # 计算漫反射光照，终止光线
        color = shade_diffuse(p, N, albedo, light_pos)
        final_color += throughput * color
        break
        
    elif mat_id == MAT_MIRROR:
        # 反射：更新射线起点和方向，继续追踪
        ro = p + N * 1e-4
        rd = reflect(rd, N)
        throughput *= 0.9
        
    elif mat_id == MAT_GLASS:
        # 玻璃：菲涅尔效应决定反射/折射
        kr = fresnel(rd, N, 1.5)
        if ti.random() < kr:
            ro = p + N * 1e-4      # 反射
            rd = reflect(rd, N)
        else:
            ro = p - N * 1e-4      # 折射
            rd = refract(rd, N, 1.5)
```

### 2.4 关键公式

**反射向量：**
$$R = L_{in} - 2(L_{in} \cdot N) N$$
**斯涅尔定律（折射）：**
$$n_1 \sin\theta_1 = n_2 \sin\theta_2$$

---

## 三、实现功能

### 3.1 基础功能

| 功能 | 说明 |
|------|------|
| 球体/平面求交 | 精确的光线-几何体相交检测 |
| 棋盘格地面 | 通过坐标奇偶性生成黑白格子 |
| 硬阴影 | 向光源发射暗影射线检测遮挡 |
| 镜面反射 | 递归追踪反射光线 |
| 交互控制 | GUI滑动条调节光源位置和弹射次数 |

### 3.2 选做功能

#### 玻璃材质（折射 + 菲涅尔效应）

- 引入斯涅尔定律计算折射方向
- 菲涅尔效应决定反射/折射比例
- 全反射现象处理

#### 抗锯齿 (MSAA)

- 每像素4次随机采样
- 平均颜色实现边缘平滑
- 可通过GUI开关控制

---

## 四、效果展示

### 4.1 基础光线追踪效果

> 左：红色漫反射球 / 右：银色镜面球 / 地面：棋盘格
![运行效果](Work5/assets/w5_0.gif)\

### 4.2 玻璃材质效果（选做）

> 玻璃球：透明、折射、菲涅尔边缘高光

### 4.3 抗锯齿对比（选做）

| 无MSAA | 4x MSAA |
|--------|---------|
| 边缘有明显锯齿 | 边缘平滑过渡 |

选做效果展示：
![运行效果](Work5/assets/w5_1.gif)\

---

## 五、技术亮点

1. **迭代替代递归**：适配GPU并行计算
2. **精度Bug修复**：射线起点沿法线偏移1e-4，避免自相交
3. **俄罗斯轮盘赌**：基于菲涅尔系数随机选择反射/折射路径
4. **实时交互**：光源位置和弹射次数可动态调节

---

## 六、Git仓库

🔗 **https://github.com/char-math/CG-Lab/tree/experiment/work5**

---

**实验完成日期**：2026年5月12日
