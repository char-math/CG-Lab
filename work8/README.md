# 计算机图形学实验报告
## 实验八：LBS蒙皮实验

| | |
|---|---|
| **姓名** | 王宇畅 |
| **学号** | 202311030025 |
| **授课教师** | 张鸿文 |
| **助教** | 张怡冉 |
| **日期** | 2026年6月17日 |

---

## 一、项目架构

采用标准的 `src` 布局，实现代码与配置的物理隔离：

```
CG-Lab/
├── models/
│   └── smpl/
│        └── SMPL_NEUTRAL.pkl      # SMPL模型文件
└── work8/
    └── src/
    │    ├── main.py               # 基础实验主程序
    │    ├── main1.py              # 选做动画主程序
    ├── assets/
    │    ├── outputs_basic/        # 基础实验输出
    │       ├── stage_a_template_weights.png
    │       ├── stage_b_shaped_joints.png
    │       ├── stage_c_pose_offsets.png
    │       ├── stage_d_lbs_result.png
    │       ├── comparison_grid.png
    │       ├── all_joint_weights.png
    │       ├── pose_animation.gif
    │       ├── multi_joint_animation.gif
    │       └── walking_animation.gif
    └── README.md
```
---

## 二、核心代码逻辑

### 2.1 模型加载与基础信息

使用 `smplx.SMPL` 加载中性SMPL模型，并输出基础信息：

```python
model = smplx.SMPL(
    model_path=model_path,
    gender="neutral",
    ext="pkl",
    num_betas=args.num_betas,
).to(device)
```

**模型信息：**
- 顶点数：6,890
- 面片数：13,776
- 关节数：24
- 形状参数维度：10

### 2.2 LBS四个阶段实现

#### (a) 模板网格与蒙皮权重

加载模板网格 `v_template` 和蒙皮权重 `lbs_weights`，将指定关节的权重通过颜色映射可视化到网格表面。

```python
weight_scalar = to_numpy(model.lbs_weights[:, joint_id])
save_single_figure(
    "stage_a_template_weights.png",
    v_template, faces,
    vertex_scalar=weight_scalar,
    title=f"Template Mesh + Weight of Joint {joint_id}"
)
```

#### (b) 形状校正与关节回归

通过形状参数 β 改变人体体型，并利用关节回归器从形状后的网格回归关节位置：

```python
betas = torch.zeros((1, num_betas))
betas[0, 0] = 2.0   # 体型
betas[0, 1] = -1.2  # 身高

v_shaped = v_template + blend_shapes(betas, shapedirs)
J = vertices2joints(model.J_regressor, v_shaped)
```

#### (c) 姿态校正

将轴角姿态参数转为旋转矩阵，构造 `pose_feature = R - I`，通过 `posedirs` 线性映射得到姿态偏移量：

```python
rot_mats = batch_rodrigues(full_pose.view(-1, 3))
pose_feature = (rot_mats[:, 1:, :, :] - ident).view(1, -1)
pose_offsets = torch.matmul(pose_feature, posedirs).view(1, -1, 3)
v_posed = v_shaped + pose_offsets
```

#### (d) LBS最终蒙皮

根据运动学树计算关节全局刚体变换，用 `lbs_weights` 加权得到最终顶点位置：

```python
J_transformed, A = batch_rigid_transform(rot_mats, J, model.parents)
T = torch.matmul(W, A.view(1, num_joints, 16)).view(1, -1, 4, 4)
v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))
verts = v_homo[:, :, :3, 0]
```

### 2.3 手写LBS与官方前向一致性验证

将手写实现的 `verts` 与官方 `model()` 前向结果逐顶点比较：

```python
def compare_with_official_forward(model, betas, global_orient, body_pose, manual_verts):
    output = model(betas=betas, global_orient=global_orient, body_pose=body_pose)
    official_verts = output.vertices
    diff = torch.abs(manual_verts - official_verts)
    return diff.mean().item(), diff.max().item()
```

**验证结果：**
- 平均绝对误差：**0.0000000000**
- 最大绝对误差：**0.0000000000**

### 2.4 选做：姿态动画

实现单关节旋转、多关节协同运动、行走循环三种动画：

```python
def create_pose_animation(model, betas, joint_id, output_path, num_frames=30):
    for i in range(num_frames):
        angle = (i / (num_frames - 1)) * max_angle
        body_pose[0, start:start+3] = torch.tensor([0.0, angle, 0.0])
        data = compute_manual_lbs(model, betas, global_orient, body_pose)
        # 渲染并保存每一帧
    imageio.mimsave(output_path, images, fps=10, loop=0)
```

---

## 三、运行效果展示

### 3.1 四阶段对比图

![comparison_grid](work8/assets/outputs_animation/comparison_grid.png)

### 3.2 全关节主导权重分布

![all_joint_weights](work8/assets/outputs_animation/all_joint_weights.png)

*不同颜色代表不同关节的主导控制区域，颜色强度表示主导权重的强弱。*

### 3.3 姿态动画（选做）

单关节旋转
![pose](work8/assets/outputs_animation/pose_animation.gif) 
多关节协同
![multi](work8/assets/outputs_animation/multi_joint_animation.gif) 
行走循环
![walk](work8/assets/outputs_animation/walking_animation.gif)

---

## 四、思考题解答

### 任务2：模板网格与蒙皮权重

**1. 为什么一个顶点不只受一个关节影响？**

人体皮肤是连续的弹性体，关节附近的顶点需要平滑过渡。多关节加权能产生更自然的形变，避免关节处出现"断裂"或"纸板剪影"效果，使皮肤看起来像是柔软地包裹在骨骼上。

**2. 如果一个顶点的权重几乎全给了某一个关节？**

该顶点会完全跟随该关节的刚体变换，在关节连接处会产生"裂缝"或"重叠"的视觉伪影，皮肤表面失去连续性，看起来像断裂的关节。

**3. 如果权重分布很平均？**

顶点会被多个关节同时拉动，产生"糖果包装纸"效应（收缩或膨胀），形变过于平滑，失去肌肉和骨骼的刚性特征，细节（如膝盖骨突出）被平滑掉。

### 任务3：形状校正与关节回归

**1. 为什么关节位置要从形状后的网格回归，而不是固定不变？**

体型变化（如高矮胖瘦）会改变关节的实际位置——胖人的关节因皮下脂肪增厚而向外扩张，瘦人则内收。固定关节位置会导致形状和骨骼不匹配，产生不自然的形变。

**2. 人物变胖/变瘦时，肩、膝、髋等关节的大致位置会不会变化？**

会变化。变胖时关节位置随体表扩张向外移动；变瘦时向内收缩；身高变化时关节沿垂直方向移动。

**3. v_template 与 v_shaped 的差别是什么？**

- `v_template`：标准T-pose下的平均人体模板（标准体型）
- `v_shaped`：根据β参数进行体型形变后的网格
- 差距在于体型特征的变化（高矮胖瘦等）

### 任务4：姿态校正

**1. 为什么LBS之前还要加pose corrective？**

纯刚体旋转变换无法模拟肌肉挤压和皮肤褶皱等非线性变形。例如弯曲肘部时，二头肌会隆起、肘部皮肤会折叠，这些都需要pose corrective来捕捉。

**2. 如果去掉pose_offsets，最终人体弯曲处会出现什么问题？**

关节弯曲处出现"糖果包装纸"效果（收缩或膨胀），肌肉和脂肪的形变无法表现，看起来像"塑料模型"，动作越极端视觉伪影越明显。

**3. v_shaped 与 v_posed 的本质区别是什么？**

- `v_shaped`：只受体型参数β影响，无姿态变化
- `v_posed`：同时受体型和姿态参数影响，但还未应用骨骼变换
- 后者包含了姿态变化带来的局部几何形变（肌肉隆起、皮肤折叠）

### 任务5：完整LBS结果

**1. J 和 J_transformed 有什么区别？**

- `J`：在模型局部坐标系中的关节位置（由形状后的网格回归得到）
- `J_transformed`：经过姿态旋转和平移后的全局关节位置
- 后者是前者经过运动学链变换后的结果

**2. 为什么最终顶点要写成加权和，而不是只选择最大权重的关节？**

加权和能保证皮肤表面的连续性，实现关节区域的平滑过渡。只选择最大权重会导致顶点在关节处"突变"，产生不自然的断裂效果。

---

## 五、Git仓库链接

🔗 **https://github.com/char-math/CG-Lab**

---

**实验完成日期：2026年6月17日**
