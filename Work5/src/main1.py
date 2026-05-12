import taichi as ti

ti.init(arch=ti.gpu)

res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

light_pos_x = ti.field(ti.f32, shape=())
light_pos_y = ti.field(ti.f32, shape=())
light_pos_z = ti.field(ti.f32, shape=())
max_bounces = ti.field(ti.i32, shape=())
enable_aa = ti.field(ti.i32, shape=())

MAT_DIFFUSE = 0
MAT_MIRROR = 1
MAT_GLASS = 2


@ti.func
def normalize(v):
    return v / (v.norm() + 1e-5)


@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N


@ti.func
def refract(I, N, ior):
    cosi = ti.math.clamp(I.dot(N), -1.0, 1.0)
    etai = 1.0
    etat = ior

    n = etai / etat

    if cosi < 0:
        cosi = -cosi
    else:
        n = etat / etai
        N = -N

    k = 1.0 - n * n * (1.0 - cosi * cosi)

    refracted = ti.Vector([0.0, 0.0, 0.0])
    can_refract = False

    if k >= 0:
        refracted = n * I + (n * cosi - ti.sqrt(k)) * N
        refracted = normalize(refracted)
        can_refract = True

    return refracted, can_refract


@ti.func
def fresnel(I, N, ior):
    cosi = ti.math.clamp(I.dot(N), -1.0, 1.0)
    etai = 1.0
    etat = ior

    if cosi > 0:
        etai, etat = etat, etai
        cosi = ti.abs(cosi)

    sint = etai / etat * ti.sqrt(ti.max(0.0, 1.0 - cosi * cosi))

    result = 0.04

    if sint < 1.0:
        cost = ti.sqrt(ti.max(0.0, 1.0 - sint * sint))
        Rs = ((etat * cosi - etai * cost) / (etat * cosi + etai * cost)) ** 2
        Rp = ((etai * cosi - etat * cost) / (etai * cosi + etat * cost)) ** 2
        result = (Rs + Rp) / 2.0
        result = ti.min(0.8, result)

    return result


@ti.func
def intersect_sphere(ro, rd, center, radius):
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c
    if delta > 0:
        sqrt_delta = ti.sqrt(delta)
        t1 = (-b - sqrt_delta) / 2.0
        if t1 > 1e-5:
            t = t1
            p = ro + rd * t
            normal = normalize(p - center)
        else:
            t2 = (-b + sqrt_delta) / 2.0
            if t2 > 1e-5:
                t = t2
                p = ro + rd * t
                normal = normalize(p - center)
    return t, normal


@ti.func
def intersect_plane(ro, rd, plane_y):
    t = -1.0
    normal = ti.Vector([0.0, 1.0, 0.0])
    if ti.abs(rd.y) > 1e-5:
        t1 = (plane_y - ro.y) / rd.y
        if t1 > 1e-5:
            t = t1
    return t, normal


@ti.func
def scene_intersect(ro, rd):
    min_t = 1e10
    hit_n = ti.Vector([0.0, 0.0, 0.0])
    hit_c = ti.Vector([0.0, 0.0, 0.0])
    hit_mat = MAT_DIFFUSE

    t, n = intersect_sphere(ro, rd, ti.Vector([-1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([1.0, 1.0, 1.0])
        hit_mat = MAT_GLASS

    t, n = intersect_sphere(ro, rd, ti.Vector([1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.9, 0.9, 0.9])
        hit_mat = MAT_MIRROR

    t, n = intersect_plane(ro, rd, -1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_mat = MAT_DIFFUSE
        p = ro + rd * t
        grid_scale = 2.0
        ix = ti.floor(p.x * grid_scale)
        iz = ti.floor(p.z * grid_scale)
        if (ix + iz) % 2 == 0:
            hit_c = ti.Vector([0.3, 0.3, 0.3])
        else:
            hit_c = ti.Vector([0.8, 0.8, 0.8])

    return min_t, hit_n, hit_c, hit_mat


@ti.func
def shade_diffuse(p, N, albedo, light_pos):
    L = normalize(light_pos - p)

    shadow_orig = p + N * 1e-4
    shadow_t, _, _, _ = scene_intersect(shadow_orig, L)
    dist_to_light = (light_pos - p).norm()
    in_shadow = shadow_t < dist_to_light - 1e-4

    ambient = 0.2 * albedo
    direct_light = ambient

    if not in_shadow:
        diff = ti.max(0.0, N.dot(L))
        diffuse = 0.8 * diff * albedo
        direct_light += diffuse

    return direct_light


@ti.func
def trace_ray(ro, rd, max_bounce):
    final_color = ti.Vector([0.0, 0.0, 0.0])
    throughput = ti.Vector([1.0, 1.0, 1.0])

    for bounce in range(max_bounce):
        t, N, albedo, mat_id = scene_intersect(ro, rd)

        if t > 1e9:
            final_color += throughput * ti.Vector([0.05, 0.1, 0.15])
            break

        p = ro + rd * t

        if mat_id == MAT_DIFFUSE:
            color = shade_diffuse(p, N, albedo, ti.Vector([light_pos_x[None], light_pos_y[None], light_pos_z[None]]))
            final_color += throughput * color
            break

        elif mat_id == MAT_MIRROR:
            ro = p + N * 1e-4
            rd = reflect(rd, N)
            throughput *= 0.95

        elif mat_id == MAT_GLASS:
            ior = 1.5
            kr = fresnel(rd, N, ior)

            reflected_ray = reflect(rd, N)
            refracted_ray, can_refract = refract(rd, N, ior)

            if can_refract:
                # 简单的俄罗斯轮盘赌
                if ti.random() < kr:
                    # 反射光线
                    ro = p + N * 1e-4
                    rd = reflected_ray
                    # 不衰减能量，让玻璃更亮
                else:
                    # 折射光线
                    ro = p - N * 1e-4
                    rd = refracted_ray
            else:
                # 全反射
                ro = p + N * 1e-4
                rd = reflected_ray

    return final_color


@ti.kernel
def render():
    for i, j in pixels:
        if enable_aa[None] == 1:
            final_color = ti.Vector([0.0, 0.0, 0.0])
            samples = 4

            for s in range(samples):
                offset_x = (ti.random() - 0.5) * 0.8
                offset_y = (ti.random() - 0.5) * 0.8

                u = (i + offset_x - res_x / 2.0) / res_y * 2.0
                v = (j + offset_y - res_y / 2.0) / res_y * 2.0

                ro = ti.Vector([0.0, 1.0, 5.0])
                rd = normalize(ti.Vector([u, v - 0.2, -1.0]))

                color = trace_ray(ro, rd, max_bounces[None])
                final_color += color

            pixels[i, j] = ti.math.clamp(final_color / samples, 0.0, 1.0)
        else:
            u = (i - res_x / 2.0) / res_y * 2.0
            v = (j - res_y / 2.0) / res_y * 2.0

            ro = ti.Vector([0.0, 1.0, 5.0])
            rd = normalize(ti.Vector([u, v - 0.2, -1.0]))

            color = trace_ray(ro, rd, max_bounces[None])
            pixels[i, j] = ti.math.clamp(color, 0.0, 1.0)


def main():
    window = ti.ui.Window("Ray Tracing - Glass Material", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()

    light_pos_x[None] = 2.0
    light_pos_y[None] = 5.0
    light_pos_z[None] = 3.0
    max_bounces[None] = 8
    enable_aa[None] = 1

    while window.running:
        render()
        canvas.set_image(pixels)

        with gui.sub_window("Controls", 0.7, 0.05, 0.28, 0.28):
            gui.text("Light Position")
            light_pos_x[None] = gui.slider_float("X", light_pos_x[None], -5.0, 5.0)
            light_pos_y[None] = gui.slider_float("Y", light_pos_y[None], 1.0, 8.0)
            light_pos_z[None] = gui.slider_float("Z", light_pos_z[None], -5.0, 5.0)

            gui.text(f"Max Bounces: {max_bounces[None]}")
            max_bounces[None] = gui.slider_int("", max_bounces[None], 1, 10)

            gui.text("Anti-aliasing")
            enable_aa[None] = 1 if gui.checkbox("Enable MSAA (4x)", enable_aa[None] == 1) else 0

        window.show()


if __name__ == '__main__':
    main()