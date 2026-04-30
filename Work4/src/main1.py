import taichi as ti

ti.init(arch=ti.gpu)

res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

Ka = ti.field(ti.f32, shape=())
Kd = ti.field(ti.f32, shape=())
Ks = ti.field(ti.f32, shape=())
shininess = ti.field(ti.f32, shape=())

use_blinn_phong = ti.field(ti.i32, shape=())
enable_shadow = ti.field(ti.i32, shape=())


@ti.func
def normalize(v):
    return v / v.norm(1e-5)


@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N


@ti.func
def intersect_sphere(ro, rd, center, radius):
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c
    if delta > 0:
        t1 = (-b - ti.sqrt(delta)) / 2.0
        t2 = (-b + ti.sqrt(delta)) / 2.0
        if t1 > 1e-5 and t2 > 1e-5:
            t = ti.min(t1, t2)
        elif t1 > 1e-5:
            t = t1
        elif t2 > 1e-5:
            t = t2
        if t > 0:
            p = ro + rd * t
            normal = normalize(p - center)
    return t, normal


@ti.func
def intersect_cone(ro, rd, apex, base_y, radius):
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    H = apex.y - base_y
    k = (radius / H) ** 2

    ro_local = ro - apex

    A = rd.x ** 2 + rd.z ** 2 - k * rd.y ** 2
    B = 2.0 * (ro_local.x * rd.x + ro_local.z * rd.z - k * ro_local.y * rd.y)
    C = ro_local.x ** 2 + ro_local.z ** 2 - k * ro_local.y ** 2

    if ti.abs(A) > 1e-5:
        delta = B ** 2 - 4.0 * A * C
        if delta > 0:
            t1 = (-B - ti.sqrt(delta)) / (2.0 * A)
            t2 = (-B + ti.sqrt(delta)) / (2.0 * A)

            t_first = t1
            t_second = t2
            if t1 > t2:
                t_first, t_second = t_second, t_first

            y1 = ro_local.y + t_first * rd.y
            if t_first > 1e-5 and -H <= y1 <= 0:
                t = t_first
            else:
                y2 = ro_local.y + t_second * rd.y
                if t_second > 1e-5 and -H <= y2 <= 0:
                    t = t_second

            if t > 0:
                p_local = ro_local + rd * t
                normal = normalize(ti.Vector([p_local.x, -k * p_local.y, p_local.z]))

    return t, normal


@ti.func
def intersect_plane(ro, rd, plane_y):
    t = -1.0
    normal = ti.Vector([0.0, 1.0, 0.0])
    if rd.y != 0:
        t = (plane_y - ro.y) / rd.y
        if t < 1e-5:
            t = -1.0
    return t, normal


@ti.func
def is_in_shadow(p, light_pos):
    shadow_dir = normalize(light_pos - p)
    shadow_ray_origin = p + shadow_dir * 0.01
    shadow_t = 1e10

    t_sph, _ = intersect_sphere(shadow_ray_origin, shadow_dir, ti.Vector([-1.0, -0.5, 0.0]), 1.0)
    if 0 < t_sph < shadow_t:
        shadow_t = t_sph

    t_cone, _ = intersect_cone(shadow_ray_origin, shadow_dir, ti.Vector([1.2, 1.0, 0.0]), -1.0, 1.0)
    if 0 < t_cone < shadow_t:
        shadow_t = t_cone

    light_dist = (light_pos - p).norm()
    return shadow_t < light_dist


@ti.kernel
def render():
    for i, j in pixels:
        u = (i - res_x / 2.0) / res_y * 2.0
        v = (j - res_y / 2.0) / res_y * 2.0

        ro = ti.Vector([0.0, 0.0, 5.0])
        rd = normalize(ti.Vector([u, v, -1.0]))

        min_t = 1e10
        hit_normal = ti.Vector([0.0, 0.0, 0.0])
        hit_color = ti.Vector([0.0, 0.0, 0.0])
        hit_point = ti.Vector([0.0, 0.0, 0.0])

        t_sph, n_sph = intersect_sphere(ro, rd, ti.Vector([-1.0, -0.5, 0.0]), 1.0)
        if 0 < t_sph < min_t:
            min_t = t_sph
            hit_normal = n_sph
            hit_color = ti.Vector([0.8, 0.2, 0.2])
            hit_point = ro + rd * min_t

        t_cone, n_cone = intersect_cone(ro, rd, ti.Vector([1.2, 1.0, 0.0]), -1.0, 1.0)
        if 0 < t_cone < min_t:
            min_t = t_cone
            hit_normal = n_cone
            hit_color = ti.Vector([0.6, 0.2, 0.8])
            hit_point = ro + rd * min_t

        t_plane, n_plane = intersect_plane(ro, rd, -1.2)
        if 0 < t_plane < min_t:
            min_t = t_plane
            hit_normal = n_plane
            hit_color = ti.Vector([0.3, 0.3, 0.4])
            hit_point = ro + rd * min_t

        color = ti.Vector([0.02, 0.05, 0.08])

        if min_t < 1e9:
            N = hit_normal
            light_pos = ti.Vector([1.5, 2.5, 2.0])
            light_color = ti.Vector([1.0, 1.0, 1.0])

            L = normalize(light_pos - hit_point)
            V = normalize(ro - hit_point)

            shadow_factor = 1.0
            if enable_shadow[None] == 1:
                if is_in_shadow(hit_point, light_pos):
                    shadow_factor = 0.0

            ambient = Ka[None] * light_color * hit_color

            diff = ti.max(0.0, N.dot(L))
            diffuse = Kd[None] * diff * light_color * hit_color

            spec = 0.0
            if use_blinn_phong[None] == 1:
                H = normalize(L + V)
                spec = ti.max(0.0, N.dot(H)) ** shininess[None]
            else:
                R = normalize(reflect(-L, N))
                spec = ti.max(0.0, R.dot(V)) ** shininess[None]

            specular = Ks[None] * spec * light_color

            if shadow_factor < 0.5:
                color = ambient
            else:
                color = ambient + diffuse + specular

        pixels[i, j] = ti.math.clamp(color, 0.0, 1.0)


def main():
    window = ti.ui.Window("Phong Shading - Sphere, Cone & Shadow", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()

    Ka[None] = 0.2
    Kd[None] = 0.7
    Ks[None] = 0.5
    shininess[None] = 32.0
    use_blinn_phong[None] = 0
    enable_shadow[None] = 0

    while window.running:
        render()
        canvas.set_image(pixels)

        with gui.sub_window("Material Parameters", 0.65, 0.05, 0.33, 0.28):
            Ka[None] = gui.slider_float('Ka (Ambient)', Ka[None], 0.0, 1.0)
            Kd[None] = gui.slider_float('Kd (Diffuse)', Kd[None], 0.0, 1.0)
            Ks[None] = gui.slider_float('Ks (Specular)', Ks[None], 0.0, 1.0)
            shininess[None] = gui.slider_float('N (Shininess)', shininess[None], 1.0, 128.0)

            gui.text("")
            use_blinn_phong[None] = gui.checkbox('Blinn-Phong (vs Phong)', use_blinn_phong[None])
            enable_shadow[None] = gui.checkbox('Hard Shadow (ON/OFF)', enable_shadow[None])

            if use_blinn_phong[None] == 1:
                gui.text("  > Blinn-Phong Active")
            else:
                gui.text("  > Phong Active")

            if enable_shadow[None] == 1:
                gui.text("  > Shadow Enabled")
            else:
                gui.text("  > Shadow Disabled")

        window.show()


if __name__ == '__main__':
    main()