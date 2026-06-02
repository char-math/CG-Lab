# 计算机图形学实验报告

## 实验七：基于质点-弹簧模型的布料模拟系统

| 项目 | 内容 |
|------|------|
| 姓名 | 王宇畅 |
| 学号 | 202311030025 |
| 授课教师 | 张鸿文 |
| 助教 | 张怡冉 |
| 日期 | 2026年6月2日 |

---

## 一、项目架构

采用 Taichi 框架与 GGUI 结合，实现基于 GPU 并行的布料模拟系统：

```
CG-Lab/
├── work7/
   ├── cloth_basic.py           # 基础实验：三种积分方法对比
   ├── cloth_spring_compare.py  # 选做实验：弹簧类型对比
   ├── cloth_collision.py       # 选做实验：球体碰撞
   └── README.md
```

**代码组织结构：**

```python
# 基础实验：布料模拟
- init_positions()      # 初始化质点位置（固定两角悬挂）
- init_springs()        # 初始化结构弹簧连接
- compute_forces_on()   # 力计算（重力+阻尼+弹簧力）
- clamp_velocity()      # 速度钳制防数值爆炸

# 三种积分方法
- step_explicit()       # 显式欧拉
- step_semi_implicit()  # 半隐式欧拉
- step_implicit_iter()  # 隐式欧拉（定点迭代）

# 选做实验
- init_springs_full()   # 结构+剪切+弯曲弹簧
- handle_sphere_collision()  # 球体碰撞检测与响应
- step_cloth_compare()  # 三布料并排对比
```

---

## 二、核心代码逻辑

### 2.1 质点-弹簧模型（基础实验）

```python
# 网格参数
N = 20                    # 20x20网格
mass = 1.0                # 质点质量
k_structural = 10000.0    # 结构弹簧劲度系数
k_d = 5.0                 # 阻尼系数
dt = 5e-4                 # 时间步长

# 弹簧力计算（胡克定律）
@ti.func
def compute_forces_on(pos, vel, force):
    # 重力和阻尼
    for i in range(N*N):
        force[i] = gravity * mass - k_d * vel[i]
    
    # 弹簧力
    for i in range(num_springs[None]):
        d = pos_a - pos_b
        dist = d.norm()
        f_spring = -k_s * (dist - rest_len) * (d / dist)
        ti.atomic_add(force[idx_a], f_spring)
        ti.atomic_add(force[idx_b], -f_spring)
```

### 2.2 三种积分方法实现

| 方法 | 更新顺序 | 代码实现 |
|------|----------|----------|
| 显式欧拉 | 位置→速度 | `x += v*dt; v += a*dt` |
| 半隐式欧拉 | 速度→位置 | `v += a*dt; x += v*dt` |
| 隐式欧拉 | 定点迭代 | 复制状态→迭代3次求解→写回 |

```python
@ti.kernel
def step_implicit_iter():
    # 1. 复制当前状态
    for i in range(N*N):
        v_next[i] = v[i]
        x_next[i] = x[i]
    
    # 2. 定点迭代（编译期展开）
    for _ in ti.static(range(3)):
        compute_forces_on(x_next, v_next, f_next)
        for i in range(N*N):
            if is_fixed[i] == 0:
                v_next[i] = v[i] + (f_next[i]/mass) * dt
                x_next[i] = x[i] + v_next[i] * dt
    
    # 3. 写回收敛状态
    for i in range(N*N):
        v[i] = v_next[i]
        x[i] = x_next[i]
```

### 2.3 弹簧类型扩展（选做）

```python
# 三种弹簧类型
spring_types: 0=结构弹簧, 1=剪切弹簧, 2=弯曲弹簧

# 弹簧劲度系数配置
if test_mode == 0:      # 仅结构弹簧（弱）
    k_struct, k_shear, k_bend = 2000, 0, 0
elif test_mode == 1:    # 结构+剪切（强剪切）
    k_struct, k_shear, k_bend = 2000, 20000, 0
else:                   # 全部弹簧（强）
    k_struct, k_shear, k_bend = 20000, 20000, 20000
```

### 2.4 球体碰撞检测（选做）

```python
@ti.func
def handle_sphere_collision(pos, vel):
    dir_to_center = pos - sphere_center
    distance = dir_to_center.norm()
    
    if distance < sphere_radius:
        normal = dir_to_center / distance
        # 位置修正：推回球面外
        new_pos = sphere_center + normal * sphere_radius
        # 速度响应：法向反弹+能量损失
        vel_normal = vel.dot(normal)
        if vel_normal < 0:
            new_vel = vel - 1.2 * vel_normal * normal
    return new_pos, new_vel
```

---

## 三、运行效果展示

### 3.1 基础实验效果

**三种积分方法对比：**

![运行效果](Work7/assets/w7_0.gif)\

| 方法 | 稳定性 | 视觉效果 | 适用场景 |
|------|--------|----------|----------|
| 显式欧拉 | 差（易爆炸） | 数值发散快 | 小时间步长 |
| 半隐式欧拉 | 好 | 自然摆动 | **推荐** |
| 隐式欧拉 | 最好 | 运动偏粘滞 | 需要稳定 |

**阻尼系数对比：**
- 阻尼=1.0：布料摆动剧烈，能量衰减慢
- 阻尼=5.0：布料运动平滑，快速稳定

### 3.2 弹簧类型对比（选做）

三块布料并排对比效果：

![运行效果](Work7/assets/w7_1.gif)\
| 位置 | 颜色 | 弹簧配置 | 视觉效果 |
|------|------|----------|----------|
| 左 | 蓝色 | 仅结构弹簧（弱） | 像渔网，下端明显变宽 |
| 中 | 绿色 | 结构+剪切（强） | 保持方形，自然下垂 |
| 右 | 橙色 | 全部弹簧（强） | 硬挺，几乎不变形 |

**观察结论：**
- 剪切弹簧：抵抗对角线拉伸，防止布料变成菱形
- 弯曲弹簧：增加抗弯刚度，减少褶皱，表面更平滑

### 3.3 球体碰撞效果（选做）

![运行效果](Work7/assets/w7_2.gif)\

- 布料从上方下落，与红色球体碰撞
- 布料自然覆盖在球体表面并滑落
- 恢复系数控制碰撞弹性（0.0粘滞 / 1.0完全弹性）

---

## 四、关键技术要点

### 4.1 GPU 并行优化

```python
# 拆分为多个 kernel 确保 GPU 同步
init_positions()   # kernel 1
init_springs()     # kernel 2
init_spring_indices()  # kernel 3

# 合并力计算和积分到单个 kernel
@ti.kernel
def step_semi_implicit():
    compute_forces_on(x, v, f)  # 内联调用
    for i in range(N*N):
        v[i] += (f[i]/mass) * dt
        x[i] += v[i] * dt

# ti.func 强制内联，减少函数调用开销
@ti.func
def compute_forces_on(...):
    ...
```

### 4.2 数值稳定性处理

| 技术 | 作用 | 实现 |
|------|------|------|
| 速度钳制 | 防止数值爆炸 | `if vel_norm > max_velocity: vel /= vel_norm * max_velocity` |
| 定点迭代 | 隐式求解 | 重复计算3-4次收敛 |
| 时间步长 | 控制精度 | `dt = 5e-4` |

### 4.3 软光栅化与可微渲染对比

| 特性 | 实验六（可微渲染） | 实验七（质点-弹簧） |
|------|---------------------|---------------------|
| 核心问题 | 梯度传播 | 数值积分稳定性 |
| 关键技术 | 软光栅化、边缘模糊 | 速度钳制、定点迭代 |
| 优化目标 | 形状/纹理拟合 | 物理真实性 |
| 计算方式 | 梯度下降 | 时间步进 |

---

## 五、遇到的问题及解决方案

| 问题 | 解决方案 |
|------|----------|
| GPU 后端不可用 | 改用 `ti.init(arch=ti.cpu)` |
| 布料数值爆炸 | 速度钳制 + 减小 dt |
| GPU 状态不同步 | 拆分多个 kernel 顺序调用 |
| 中文字符显示乱码 | 改用英文 UI 文本 |
| Taichi 不支持 if 内 return | 改为先赋值后返回 |

---

## 六、实验总结

本次实验成功实现了基于质点-弹簧模型的布料模拟系统，完成以下内容：

**1. 基础功能**
- 20×20 网格布料，结构弹簧连接
- 三种积分方法可实时切换
- GGUI 交互面板（方法切换、参数调节、暂停/重置）

**2. GPU 优化**
- 采用 Taichi 框架实现并行计算
- kernel 拆分保证状态同步
- 力计算与积分合并减少启动开销

**3. 选做功能**
- 剪切弹簧和弯曲弹簧扩展
- 三布料并排对比弹簧效果
- 球体碰撞检测与响应

**4. 实验数据**
- 弹簧数量：2202 条（20×20网格）
- 质点数量：400 个
- FPS：约 30-60（取决于参数）

通过实验深刻理解了：
- 质点-弹簧模型的物理原理和参数影响
- 不同数值积分方法的稳定性差异
- GPU 并行计算的编程范式

---

## 七、GUI 控制面板功能

| 控件 | 功能 |
|------|------|
| Explicit/Semi-Implicit/Implicit | 切换积分方法 |
| Pause/Resume | 暂停/继续模拟 |
| Reset Cloth | 重置布料状态 |
| Structural/Shear/Bending 滑块 | 调节各弹簧劲度系数 |
| Damping 滑块 | 调节阻尼系数 |
| Enable Collision | 开关球体碰撞 |
| Show Springs/Sphere | 开关渲染选项 |

---

## 八、Git 仓库链接

🔗 **https://github.com/char-math/CG-Lab/tree/experience/work7/work7**

---

**实验完成日期：2026年6月2日**
