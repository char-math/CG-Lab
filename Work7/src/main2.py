import taichi as ti

ti.init(arch=ti.cpu, cpu_max_num_threads=8)

N = 20
mass = 1.0
dt = 5e-4
k_structural = 8000.0
k_shear = 3000.0
k_bending = 2000.0
k_d = 5.0
gravity = ti.Vector([0.0, -9.8, 0.0])
max_velocity = 50.0

sphere_center = ti.Vector([0.0, 0.6, 0.0])
sphere_radius = 0.35
enable_collision = True
collision_restitution = 0.7

x = ti.Vector.field(3, dtype=float, shape=N * N)
v = ti.Vector.field(3, dtype=float, shape=N * N)
f = ti.Vector.field(3, dtype=float, shape=N * N)
is_fixed = ti.field(dtype=int, shape=N * N)

x_next = ti.Vector.field(3, dtype=float, shape=N * N)
v_next = ti.Vector.field(3, dtype=float, shape=N * N)
f_next = ti.Vector.field(3, dtype=float, shape=N * N)

max_springs = N * N * 8
spring_indices = ti.field(dtype=int, shape=max_springs * 2)
spring_pairs = ti.Vector.field(2, dtype=int, shape=max_springs)
spring_lengths = ti.field(dtype=float, shape=max_springs)
spring_types = ti.field(dtype=int, shape=max_springs)
num_springs = ti.field(dtype=int, shape=())


@ti.kernel
def init_positions():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        x[idx] = ti.Vector([i * 0.04 - 0.4, 1.2, j * 0.04 - 0.4])
        v[idx] = ti.Vector([0.0, 0.0, 0.0])
        f[idx] = ti.Vector([0.0, 0.0, 0.0])
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
            spring_lengths[c] = (x[idx] - x[idx_right]).norm()
            spring_types[c] = 0

        if j < N - 1:
            idx_down = i * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_down])
            spring_lengths[c] = (x[idx] - x[idx_down]).norm()
            spring_types[c] = 0

        if i < N - 1 and j < N - 1:
            idx_diag = (i + 1) * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_diag])
            spring_lengths[c] = (x[idx] - x[idx_diag]).norm()
            spring_types[c] = 1

        if i < N - 1 and j > 0:
            idx_diag = (i + 1) * N + (j - 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_diag])
            spring_lengths[c] = (x[idx] - x[idx_diag]).norm()
            spring_types[c] = 1

        if i < N - 2:
            idx_far = (i + 2) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_far])
            spring_lengths[c] = (x[idx] - x[idx_far]).norm()
            spring_types[c] = 2

        if j < N - 2:
            idx_far = i * N + (j + 2)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_far])
            spring_lengths[c] = (x[idx] - x[idx_far]).norm()
            spring_types[c] = 2


@ti.kernel
def init_spring_indices():
    for i in range(num_springs[None]):
        spring_indices[i * 2] = spring_pairs[i][0]
        spring_indices[i * 2 + 1] = spring_pairs[i][1]


def init_cloth():
    num_springs[None] = 0
    init_positions()
    init_springs()
    init_spring_indices()


@ti.func
def get_spring_k(spring_type: int) -> ti.f32:
    result = 0.0
    if spring_type == 0:
        result = k_structural
    elif spring_type == 1:
        result = k_shear
    else:
        result = k_bending
    return result


@ti.func
def compute_forces_on(pos: ti.template(), vel: ti.template(), force: ti.template()):
    for i in range(N * N):
        force[i] = gravity * mass - k_d * vel[i]

    for i in range(num_springs[None]):
        idx_a = spring_pairs[i][0]
        idx_b = spring_pairs[i][1]
        pos_a = pos[idx_a]
        pos_b = pos[idx_b]
        d = pos_a - pos_b
        dist = d.norm()
        if dist > 1e-6:
            d_normalized = d / dist
            k = get_spring_k(spring_types[i])
            f_spring = -k * (dist - spring_lengths[i]) * d_normalized
            ti.atomic_add(force[idx_a], f_spring)
            ti.atomic_add(force[idx_b], -f_spring)


@ti.func
def handle_sphere_collision(pos: ti.math.vec3, vel: ti.math.vec3) -> ti.math.vec3:
    result_vel = vel
    if enable_collision:
        dir_to_center = pos - sphere_center
        distance = dir_to_center.norm()
        if distance < sphere_radius:
            normal = dir_to_center / distance
            vel_normal = vel.dot(normal)
            if vel_normal < 0:
                vel_tangent = vel - vel_normal * normal
                result_vel = vel_tangent - vel_normal * collision_restitution * normal
    return result_vel


@ti.func
def clamp_velocity(vel: ti.template(), idx: int):
    vel_norm = vel[idx].norm()
    if vel_norm > max_velocity:
        vel[idx] = vel[idx] / vel_norm * max_velocity


@ti.kernel
def step_explicit():
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            x[i] += v[i] * dt
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
            v[i] = handle_sphere_collision(x[i], v[i])


@ti.kernel
def step_semi_implicit():
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
            x[i] += v[i] * dt
            if enable_collision:
                dir_to_center = x[i] - sphere_center
                dist = dir_to_center.norm()
                if dist < sphere_radius:
                    normal = dir_to_center / dist
                    x[i] = sphere_center + normal * sphere_radius
                    v[i] = handle_sphere_collision(x[i], v[i])


@ti.kernel
def step_implicit_iter():
    for i in range(N * N):
        v_next[i] = v[i]
        x_next[i] = x[i]

    for _ in ti.static(range(4)):
        compute_forces_on(x_next, v_next, f_next)
        for i in range(N * N):
            if is_fixed[i] == 0:
                v_next[i] = v[i] + (f_next[i] / mass) * dt
                clamp_velocity(v_next, i)
                x_next[i] = x[i] + v_next[i] * dt

                if enable_collision:
                    dir_to_center = x_next[i] - sphere_center
                    dist = dir_to_center.norm()
                    if dist < sphere_radius:
                        normal = dir_to_center / dist
                        x_next[i] = sphere_center + normal * sphere_radius
                        v_next[i] = handle_sphere_collision(x_next[i], v_next[i])

    for i in range(N * N):
        v[i] = v_next[i]
        x[i] = x_next[i]


@ti.kernel
def reset_simulation():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        if is_fixed[idx] == 0:
            x[idx] = ti.Vector([i * 0.04 - 0.4, 1.2, j * 0.04 - 0.4])
            v[idx] = ti.Vector([0.0, 0.0, 0.0])
            f[idx] = ti.Vector([0.0, 0.0, 0.0])


def main():
    print("Initializing cloth simulation...")
    init_cloth()
    print(f"Done! Particles: {N * N}, Springs: {num_springs[None]}")

    window = ti.ui.Window("Cloth Simulation - Mass Spring System", (1024, 1024))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.0, 1.0, 2.2)
    camera.lookat(0.0, 0.6, 0.0)

    current_method = 1
    paused = False
    show_springs = True
    show_sphere = True

    import time
    last_time = time.time()
    frame_count = 0
    fps = 0.0

    while window.running:
        frame_count += 1
        current_time = time.time()
        if current_time - last_time >= 1.0:
            fps = frame_count
            frame_count = 0
            last_time = current_time

        window.GUI.begin("Control Panel", 0.02, 0.02, 0.4, 0.6)

        window.GUI.text("Integration Method:")
        if window.GUI.button("Explicit Euler"):
            current_method = 0
            init_cloth()
        if window.GUI.button("Semi-Implicit Euler"):
            current_method = 1
            init_cloth()
        if window.GUI.button("Implicit Euler"):
            current_method = 2
            init_cloth()

        window.GUI.text("")

        if window.GUI.button("Pause/Resume"):
            paused = not paused
        if window.GUI.button("Reset Cloth"):
            reset_simulation()

        window.GUI.text("")

        window.GUI.text("Spring Parameters:")
        global k_structural, k_shear, k_bending, k_d
        k_structural = window.GUI.slider_float("Structural", k_structural, 1000.0, 15000.0)
        k_shear = window.GUI.slider_float("Shear", k_shear, 500.0, 8000.0)
        k_bending = window.GUI.slider_float("Bending", k_bending, 500.0, 8000.0)
        k_d = window.GUI.slider_float("Damping", k_d, 0.0, 20.0)

        window.GUI.text("")

        window.GUI.text("Collision Parameters:")
        global enable_collision, collision_restitution
        enable_collision = window.GUI.checkbox("Enable Collision", enable_collision)
        collision_restitution = window.GUI.slider_float("Restitution", collision_restitution, 0.0, 1.0)

        window.GUI.text("")

        window.GUI.text("Display Options:")
        show_springs = window.GUI.checkbox("Show Springs", show_springs)
        show_sphere = window.GUI.checkbox("Show Sphere", show_sphere)

        window.GUI.text("")
        window.GUI.text(f"Spring Count: {num_springs[None]}")
        window.GUI.text(f"FPS: {fps:.1f}")

        window.GUI.end()

        if not paused:
            for _ in range(20):
                if current_method == 0:
                    step_explicit()
                elif current_method == 1:
                    step_semi_implicit()
                else:
                    step_implicit_iter()

        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.4, 0.4, 0.4))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
        scene.point_light(pos=(-0.5, 1.5, 1.0), color=(0.8, 0.8, 1.0))

        scene.particles(x, radius=0.012, color=(0.2, 0.7, 1.0))

        if show_springs:
            scene.lines(x, indices=spring_indices, width=1.0, color=(0.6, 0.6, 0.8))

        if show_sphere:
            temp_sphere = ti.Vector.field(3, dtype=float, shape=1)
            temp_sphere[0] = sphere_center
            scene.particles(temp_sphere, radius=sphere_radius, color=(1.0, 0.3, 0.3))

        canvas.scene(scene)
        window.show()


if __name__ == '__main__':
    main()