import taichi as ti

ti.init(arch=ti.cpu, cpu_max_num_threads=8)

N = 12
mass = 1.0
dt = 5e-4
gravity = ti.Vector([0.0, -9.8, 0.0])
max_velocity = 50.0

# 三块布料
x0 = ti.Vector.field(3, dtype=float, shape=N * N)
v0 = ti.Vector.field(3, dtype=float, shape=N * N)
f0 = ti.Vector.field(3, dtype=float, shape=N * N)

x1 = ti.Vector.field(3, dtype=float, shape=N * N)
v1 = ti.Vector.field(3, dtype=float, shape=N * N)
f1 = ti.Vector.field(3, dtype=float, shape=N * N)

x2 = ti.Vector.field(3, dtype=float, shape=N * N)
v2 = ti.Vector.field(3, dtype=float, shape=N * N)
f2 = ti.Vector.field(3, dtype=float, shape=N * N)

is_fixed = ti.field(dtype=int, shape=N * N)

max_springs = N * N * 8
spring_pairs = ti.Vector.field(2, dtype=int, shape=max_springs)
spring_lengths = ti.field(dtype=float, shape=max_springs)
spring_types = ti.field(dtype=int, shape=max_springs)
num_springs = ti.field(dtype=int, shape=())


@ti.kernel
def init_positions():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        # 三块布料：左、中、右
        x0[idx] = ti.Vector([j * 0.06 - 0.5, i * 0.06 + 0.8, 0.0])
        x1[idx] = ti.Vector([j * 0.06 + 0.0, i * 0.06 + 0.8, 0.0])
        x2[idx] = ti.Vector([j * 0.06 + 0.5, i * 0.06 + 0.8, 0.0])

        v0[idx] = ti.Vector([0.0, 0.0, 0.0])
        v1[idx] = ti.Vector([0.0, 0.0, 0.0])
        v2[idx] = ti.Vector([0.0, 0.0, 0.0])

        if i == 0 and (j == 0 or j == N - 1):
            is_fixed[idx] = 1
        else:
            is_fixed[idx] = 0


@ti.kernel
def init_springs():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j

        if i < N - 1:
            idx_right = (i + 1) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_right])
            spring_lengths[c] = 0.06
            spring_types[c] = 0

        if j < N - 1:
            idx_down = i * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_down])
            spring_lengths[c] = 0.06
            spring_types[c] = 0

        # 剪切弹簧
        if i < N - 1 and j < N - 1:
            idx_diag = (i + 1) * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_diag])
            spring_lengths[c] = 0.08485
            spring_types[c] = 1
        if i < N - 1 and j > 0:
            idx_diag = (i + 1) * N + (j - 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_diag])
            spring_lengths[c] = 0.08485
            spring_types[c] = 1

        # 弯曲弹簧
        if i < N - 2:
            idx_far = (i + 2) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_far])
            spring_lengths[c] = 0.12
            spring_types[c] = 2
        if j < N - 2:
            idx_far = i * N + (j + 2)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_far])
            spring_lengths[c] = 0.12
            spring_types[c] = 2


def init_all():
    num_springs[None] = 0
    init_positions()
    init_springs()


@ti.func
def compute_spring_force(pos, idx_a, idx_b, rest_len, spring_k):
    pos_a = pos[idx_a]
    pos_b = pos[idx_b]
    d = pos_a - pos_b
    dist = d.norm()
    force = ti.math.vec3(0.0)
    if dist > 1e-6:
        f = -spring_k * (dist - rest_len) * (d / dist)
        force = f
    return force


@ti.func
def step_cloth(pos, vel, force, k_struct, k_shear_val, k_bend, k_damp):
    for i in range(N * N):
        force[i] = gravity * mass - k_damp * vel[i]

    for s in range(num_springs[None]):
        idx_a = spring_pairs[s][0]
        idx_b = spring_pairs[s][1]
        spring_type = spring_types[s]

        spring_k = 0.0
        if spring_type == 0:
            spring_k = k_struct
        elif spring_type == 1:
            spring_k = k_shear_val
        else:
            spring_k = k_bend

        if spring_k > 0.0:
            f = compute_spring_force(pos, idx_a, idx_b, spring_lengths[s], spring_k)
            ti.atomic_add(force[idx_a], f)
            ti.atomic_add(force[idx_b], -f)

    for i in range(N * N):
        if is_fixed[i] == 0:
            acc = force[i] / mass
            vel[i] += acc * dt
            vel_norm = vel[i].norm()
            if vel_norm > max_velocity:
                vel[i] = vel[i] / vel_norm * max_velocity
            pos[i] += vel[i] * dt


@ti.kernel
def step_all():
    # 左边蓝色：只有结构弹簧（弱）
    step_cloth(x0, v0, f0, 2000.0, 0.0, 0.0, 2.0)
    # 中间绿色：结构+剪切（强剪切）
    step_cloth(x1, v1, f1, 2000.0, 20000.0, 0.0, 2.0)
    # 右边橙色：结构+剪切+弯曲（全部强）
    step_cloth(x2, v2, f2, 20000.0, 20000.0, 20000.0, 2.0)


@ti.kernel
def reset_all():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        if is_fixed[idx] == 0:
            x0[idx] = ti.Vector([j * 0.06 - 0.5, i * 0.06 + 0.8, 0.0])
            x1[idx] = ti.Vector([j * 0.06 + 0.0, i * 0.06 + 0.8, 0.0])
            x2[idx] = ti.Vector([j * 0.06 + 0.5, i * 0.06 + 0.8, 0.0])
            v0[idx] = ti.Vector([0.0, 0.0, 0.0])
            v1[idx] = ti.Vector([0.0, 0.0, 0.0])
            v2[idx] = ti.Vector([0.0, 0.0, 0.0])


def main():
    init_all()

    window = ti.ui.Window("Three Cloths Comparison", (1200, 800))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.0, 0.6, 3.2)
    camera.lookat(0.0, 0.5, 0.0)

    paused = False

    while window.running:
        window.GUI.begin("Control", 0.02, 0.02, 0.35, 0.25)

        window.GUI.text("LEFT (BLUE)   : Structural only (weak)")
        window.GUI.text("CENTER (GREEN): Structural + Shear")
        window.GUI.text("RIGHT (ORANGE): All springs (stiff)")
        window.GUI.text("")

        if window.GUI.button("Pause"):
            paused = not paused
        if window.GUI.button("Reset"):
            reset_all()

        window.GUI.end()

        if not paused:
            for _ in range(30):
                step_all()

        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.4, 0.4, 0.4))
        scene.point_light(pos=(0, 2, 0), color=(1, 1, 1))

        scene.particles(x0, radius=0.008, color=(0.2, 0.4, 1.0))
        scene.particles(x1, radius=0.008, color=(0.2, 0.8, 0.3))
        scene.particles(x2, radius=0.008, color=(1.0, 0.5, 0.2))

        canvas.scene(scene)
        window.show()


if __name__ == '__main__':
    print("=" * 60)
    print("THREE CLOTHS SIDE BY SIDE")
    print("=" * 60)
    print("")
    print("LEFT   (BLUE)  : Weak structural springs")
    print("       -> Will stretch and become wider at bottom")
    print("")
    print("CENTER (GREEN) : Strong shear springs")
    print("       -> Resists diagonal stretch, keeps shape")
    print("")
    print("RIGHT  (ORANGE): All springs very strong")
    print("       -> Very stiff, almost no deformation")
    print("=" * 60)
    main()