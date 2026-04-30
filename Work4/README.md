# 计算机图形学实验报告
## 实验四：Phong 光照模型与交互式渲染

| 项目 | 内容 |
|------|------|
| 姓名 | 王宇畅 |
| 学号 | 202311030025 |
| 授课教师 | 张鸿文 |
| 助教 | 张怡冉 |
| 日期 | 2026年4月30日 |

---

## 一、项目架构

采用标准的 src 布局，实现代码与配置的物理隔离：
```
Work4/
├── assets/ # 演示资源
│ ├── demo.gif # 基础版运行效果
│ └── demo1.gif # 选做功能演示
├── src/
│ └── main.py # 基础 Phong 模型实现
│ └── main1.py # 含 Blinn-Phong + 阴影的完整实现
├── .gitignore
├── pyproject.toml
└── README.md

```

---

## 二、核心代码逻辑

### 2.1 几何体相交测试 (intersect_sphere / intersect_cone)

使用光线投射（Ray Casting）数学隐式定义几何体：

```python
@ti.func
def intersect_sphere(ro, rd, center, radius):
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c
    if delta > 0:
        t = (-b - ti.sqrt(delta)) / 2.0
        if t > 0:
            p = ro + rd * t
            normal = normalize(p - center)
    return t, normal
```
#### 2.2 Z-Buffer 深度测试
实现类似 Z-buffer 的深度竞争逻辑，保证正确的遮挡关系：

``` python
min_t = 1e10
if 0 < t_sph < min_t:
    min_t = t_sph
    hit_normal = n_sph
    hit_color = sphere_color
if 0 < t_cone < min_t:
    min_t = t_cone
    hit_normal = n_cone
    hit_color = cone_color
```
#### 2.3 Phong 光照模型 (main.py)
```python
# 环境光
ambient = Ka * light_color * hit_color

# 漫反射
diff = ti.max(0.0, N.dot(L))
diffuse = Kd * diff * light_color * hit_color

# 镜面高光
R = normalize(reflect(-L, N))
spec = ti.max(0.0, R.dot(V)) ** shininess
specular = Ks * spec * light_color

color = ambient + diffuse + specular
```
#### 2.4 Blinn-Phong 模型（选做）
```python
# 使用半程向量替代反射向量
H = normalize(L + V)
spec = ti.max(0.0, N.dot(H)) ** shininess
```
#### 2.5 硬阴影（选做）
```python
@ti.func
def is_in_shadow(p, light_pos):
    shadow_dir = normalize(light_pos - p)
    shadow_ray_origin = p + shadow_dir * 0.01
    # 检测与其他物体的交点
    t_sph, _ = intersect_sphere(...)
    t_cone, _ = intersect_cone(...)
    light_dist = (light_pos - p).norm()
    return min(t_sph, t_cone) < light_dist
```
## 三、实验场景设计
| 物体 | 位置 | 尺寸 | 颜色 |
| --- | --- | --- | --- |
| 红色球体 | 中心 (-1.0, -0.5, 0) | 半径 1.0 | (0.8, 0.2, 0.2) |
| 紫色圆锥 | 顶点 (1.2, 1.0, 0)，底面 y = -1.0 | 半径 1.0 | (0.6, 0.2, 0.8) |
| 地面平面 | y = -1.2 | 无限大 | (0.3, 0.3, 0.4) |

| 参数 | 位置/颜色 |
| --- | --- |
| 相机位置 | (0, 0, 5) |
| 光源位置 | (1.5, 2.5, 2.0) |
| 光源颜色 | (1.0, 1.0, 1.0) 白光 |
| 背景色 | (0.02, 0.05, 0.08) 深蓝色 |

## 四、运行效果展示
#### 4.1 基础 Phong 模型

![运行效果](Work4/assets/demo.gif)\

| 参数 | 含义 | 范围 | 默认值 |
| --- | --- | --- | --- |
| Ka | 环境光系数 | 0.0 ~ 1.0 | 0.2 |
| Kd | 漫反射系数 | 0.0 ~ 1.0 | 0.7 |
| Ks | 镜面高光系数 | 0.0 ~ 1.0 | 0.5 |
| Shininess | 高光指数 | 1.0 ~ 128.0 | 32.0 |


#### 4.2 选做功能

![运行效果](Work4/assets/demo1.gif)\
Blinn-Phong 模型切换：高光区域更圆润柔和

硬阴影开关：物体间产生清晰阴影


## 五、Git 仓库链接
🔗 https://github.com/char-math/CG-Lab/tree/experiment/work4/Work4 \
实验完成日期：2026年4月30日
