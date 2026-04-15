import taichi as ti
import numpy as np
import time

ti.init(arch=ti.gpu)

WIDTH = 800
HEIGHT = 800
MAX_CONTROL_POINTS = 100
NUM_SEGMENTS = 1000

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
gui_indices = ti.field(dtype=ti.i32, shape=MAX_CONTROL_POINTS * 2)

curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)


def de_casteljau(points, t):
    if len(points) == 1:
        return points[0]

    next_points = []
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]
        x = (1.0 - t) * p0[0] + t * p1[0]
        y = (1.0 - t) * p0[1] + t * p1[1]
        next_points.append([x, y])

    return de_casteljau(next_points, t)


def compute_bezier_curve(control_points, num_segments):
    if len(control_points) < 2:
        return []

    curve_points = []
    for i in range(num_segments + 1):
        t = i / num_segments
        point = de_casteljau(control_points, t)
        curve_points.append(point)

    return curve_points


def compute_bspline_curve(control_points, num_segments):
    n = len(control_points)
    if n < 4:
        return []

    curve_points = []
    segments_per_span = num_segments // (n - 3)

    for i in range(n - 3):
        p0 = control_points[i]
        p1 = control_points[i + 1]
        p2 = control_points[i + 2]
        p3 = control_points[i + 3]

        for j in range(segments_per_span + 1):
            if i == n - 4 and j == segments_per_span:
                continue

            t = j / segments_per_span

            t2 = t * t
            t3 = t2 * t

            b0 = (1 - t) ** 3 / 6
            b1 = (3 * t ** 3 - 6 * t ** 2 + 4) / 6
            b2 = (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6
            b3 = t ** 3 / 6

            x = b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0]
            y = b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1]

            curve_points.append([x, y])

    last_t = 1.0
    i = n - 4
    p0 = control_points[i]
    p1 = control_points[i + 1]
    p2 = control_points[i + 2]
    p3 = control_points[i + 3]

    t = last_t
    t2 = t * t
    t3 = t2 * t

    b0 = (1 - t) ** 3 / 6
    b1 = (3 * t ** 3 - 6 * t ** 2 + 4) / 6
    b2 = (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6
    b3 = t ** 3 / 6

    x = b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0]
    y = b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1]

    curve_points.append([x, y])

    return curve_points


@ti.kernel
def clear_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def draw_bezier_kernel(n: ti.i32):
    for i in range(n):
        pt = curve_points_field[i]
        x_pixel = ti.cast(pt[0] * WIDTH, ti.i32)
        y_pixel = ti.cast(pt[1] * HEIGHT, ti.i32)
        if 0 <= x_pixel < WIDTH and 0 <= y_pixel < HEIGHT:
            pixels[x_pixel, y_pixel] = ti.Vector([0.0, 1.0, 0.0])


@ti.kernel
def draw_bspline_kernel(n: ti.i32):
    for i in range(n):
        pt = curve_points_field[i]
        x_pixel = ti.cast(pt[0] * WIDTH, ti.i32)
        y_pixel = ti.cast(pt[1] * HEIGHT, ti.i32)
        if 0 <= x_pixel < WIDTH and 0 <= y_pixel < HEIGHT:
            pixels[x_pixel, y_pixel] = ti.Vector([0.0, 0.5, 1.0])


@ti.kernel
def draw_antialiased_bezier_kernel(n: ti.i32):
    for i in range(n):
        pt = curve_points_field[i]
        x_float = pt[0] * WIDTH
        y_float = pt[1] * HEIGHT

        x_center = ti.cast(x_float, ti.i32)
        y_center = ti.cast(y_float, ti.i32)

        for dx in range(-2, 3):
            for dy in range(-2, 3):
                px = x_center + dx
                py = y_center + dy
                if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                    dist = ti.sqrt(float(dx * dx + dy * dy))
                    if dist <= 2.0:
                        alpha = 1.0 - dist / 2.5
                        if alpha > 0.0:
                            pixels[px, py] = pixels[px, py] + ti.Vector([0.0, 1.0, 0.0]) * alpha


@ti.kernel
def draw_antialiased_bspline_kernel(n: ti.i32):
    for i in range(n):
        pt = curve_points_field[i]
        x_float = pt[0] * WIDTH
        y_float = pt[1] * HEIGHT

        x_center = ti.cast(x_float, ti.i32)
        y_center = ti.cast(y_float, ti.i32)

        for dx in range(-2, 3):
            for dy in range(-2, 3):
                px = x_center + dx
                py = y_center + dy
                if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                    dist = ti.sqrt(float(dx * dx + dy * dy))
                    if dist <= 2.0:
                        alpha = 1.0 - dist / 2.5
                        if alpha > 0.0:
                            pixels[px, py] = pixels[px, py] + ti.Vector([0.0, 0.5, 1.0]) * alpha


@ti.kernel
def draw_point_marker(x: ti.f32, y: ti.f32):
    x_pixel = ti.cast(x * WIDTH, ti.i32)
    y_pixel = ti.cast(y * HEIGHT, ti.i32)

    for dx in range(-2, 3):
        for dy in range(-2, 3):
            px = x_pixel + dx
            py = y_pixel + dy
            if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                pixels[px, py] = ti.Vector([1.0, 1.0, 0.0])


@ti.kernel
def draw_line_kernel(x1: ti.f32, y1: ti.f32, x2: ti.f32, y2: ti.f32):
    x1_pixel = ti.cast(x1 * WIDTH, ti.i32)
    y1_pixel = ti.cast(y1 * HEIGHT, ti.i32)
    x2_pixel = ti.cast(x2 * WIDTH, ti.i32)
    y2_pixel = ti.cast(y2 * HEIGHT, ti.i32)

    dx = ti.abs(x2_pixel - x1_pixel)
    dy = ti.abs(y2_pixel - y1_pixel)

    if dx >= dy:
        if x1_pixel > x2_pixel:
            x1_pixel, x2_pixel = x2_pixel, x1_pixel
            y1_pixel, y2_pixel = y2_pixel, y1_pixel

        y = ti.cast(y1_pixel, ti.f32)
        y_step = 1.0 if y2_pixel > y1_pixel else -1.0
        error = 0.0
        delta_err = dy / dx if dx != 0 else 0.0

        for x in range(x1_pixel, x2_pixel + 1):
            if 0 <= x < WIDTH and 0 <= ti.cast(y, ti.i32) < HEIGHT:
                pixels[x, ti.cast(y, ti.i32)] = ti.Vector([0.5, 0.5, 0.5])
            error += delta_err
            while error >= 0.5:
                y += y_step
                error -= 1.0
    else:
        if y1_pixel > y2_pixel:
            x1_pixel, x2_pixel = x2_pixel, x1_pixel
            y1_pixel, y2_pixel = y2_pixel, y1_pixel

        x = ti.cast(x1_pixel, ti.f32)
        x_step = 1.0 if x2_pixel > x1_pixel else -1.0
        error = 0.0
        delta_err = dx / dy if dy != 0 else 0.0

        for y in range(y1_pixel, y2_pixel + 1):
            if 0 <= ti.cast(x, ti.i32) < WIDTH and 0 <= y < HEIGHT:
                pixels[ti.cast(x, ti.i32), y] = ti.Vector([0.5, 0.5, 0.5])
            error += delta_err
            while error >= 0.5:
                x += x_step
                error -= 1.0


def main():
    window = ti.ui.Window("Bezier & B-Spline - B键切换 | A键抗锯齿 | C键清空", (WIDTH, HEIGHT))
    canvas = window.get_canvas()

    control_points = []
    last_point_count = 0
    last_click_time = 0
    antialiasing = False
    mode = 0

    print("=" * 60)
    print("贝塞尔曲线 & B样条曲线绘制程序")
    print("=" * 60)
    print("操作说明：")
    print("  - 鼠标左键点击：添加控制点")
    print("  - B键：切换贝塞尔曲线/B样条曲线模式")
    print("  - A键：切换抗锯齿开关")
    print("  - C键：清空所有控制点")
    print("=" * 60)
    print("当前模式：贝塞尔曲线 (绿色)")
    print("B样条曲线使用均匀三次B样条公式，蓝色显示")

    while window.running:
        for event in window.get_events(ti.ui.PRESS):
            if event.key == 'c':
                control_points.clear()
                last_point_count = -1
                print("已清空所有控制点")
            elif event.key == 'b':
                mode = 1 - mode
                if mode == 0:
                    print("切换到贝塞尔曲线模式 (绿色)")
                else:
                    print("切换到B样条曲线模式 (蓝色) - 至少需要4个控制点")
                last_point_count = -1
            elif event.key == 'a':
                antialiasing = not antialiasing
                if antialiasing:
                    print("启用抗锯齿")
                else:
                    print("禁用抗锯齿")
                last_point_count = -1

        current_time = time.time()
        if window.is_pressed(ti.ui.LMB) and (current_time - last_click_time) > 0.15:
            if len(control_points) < MAX_CONTROL_POINTS:
                pos = window.get_cursor_pos()
                x = max(0.01, min(0.99, pos[0]))
                y = max(0.01, min(0.99, pos[1]))

                is_duplicate = False
                for p in control_points:
                    if abs(p[0] - x) < 0.005 and abs(p[1] - y) < 0.005:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    control_points.append([x, y])
                    print(f"添加控制点 {len(control_points)}: ({x:.3f}, {y:.3f})")
                    last_point_count = -1
                    last_click_time = current_time

        clear_pixels()

        for i in range(len(control_points) - 1):
            p1 = control_points[i]
            p2 = control_points[i + 1]
            draw_line_kernel(p1[0], p1[1], p2[0], p2[1])

        for point in control_points:
            draw_point_marker(point[0], point[1])

        current_count = len(control_points)

        need_update = current_count != last_point_count

        if mode == 0:
            if current_count >= 2:
                if need_update:
                    curve_points = compute_bezier_curve(control_points, NUM_SEGMENTS)
                    if curve_points:
                        curve_points_np = np.zeros((len(curve_points), 2), dtype=np.float32)
                        for i, point in enumerate(curve_points):
                            curve_points_np[i] = point
                        curve_points_field.from_numpy(curve_points_np)
                        last_point_count = current_count
                if antialiasing:
                    draw_antialiased_bezier_kernel(NUM_SEGMENTS + 1)
                else:
                    draw_bezier_kernel(NUM_SEGMENTS + 1)
        else:
            if current_count >= 4:
                if need_update:
                    curve_points = compute_bspline_curve(control_points, NUM_SEGMENTS)
                    if curve_points:
                        curve_points_np = np.zeros((len(curve_points), 2), dtype=np.float32)
                        for i, point in enumerate(curve_points):
                            curve_points_np[i] = point
                        curve_points_field.from_numpy(curve_points_np)
                        last_point_count = current_count
                if antialiasing:
                    draw_antialiased_bspline_kernel(len(curve_points))
                else:
                    draw_bspline_kernel(len(curve_points))
            elif current_count < 4 and current_count >= 2 and need_update:
                print("B样条曲线至少需要4个控制点，当前控制点数量: {}".format(current_count))
                last_point_count = current_count

        if current_count > 0:
            np_points = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
            np_points[:current_count] = np.array(control_points, dtype=np.float32)
            gui_points.from_numpy(np_points)
            canvas.circles(gui_points, radius=0.008, color=(1.0, 0.0, 0.0))

            if current_count >= 2:
                np_indices = np.zeros(MAX_CONTROL_POINTS * 2, dtype=np.int32)
                indices = []
                for i in range(current_count - 1):
                    indices.extend([i, i + 1])
                np_indices[:len(indices)] = np.array(indices, dtype=np.int32)
                gui_indices.from_numpy(np_indices)
                canvas.lines(gui_points, width=0.002, indices=gui_indices, color=(0.5, 0.5, 0.5))

        canvas.set_image(pixels)
        window.show()


if __name__ == '__main__':
    main()