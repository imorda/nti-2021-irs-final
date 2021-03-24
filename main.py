import math
import collections
from threading import Thread
from robot_library.robot import *
import time as tm
from scripts.artag_detector import *


DEBUG = False


class RobotConst:  # константы
    cell_size = 0.304
    exact_cell_size = 0.3
    wheel_d = 0.17999999999999999
    track = 0.29999999999999999
    cpr = 2 * math.pi

    forward_laser_delta = 0.154  # на столько м вынесен дальномер вперед центра оси
    min_lasers_on_edge = 15  # столько лазеров должно попасть на ребро, чтобы считать его препятствием
    laser_node_collision_dist = 0.10  # расстояние от точки на ребре до узла должно быть не меньше этого (в клетках)
    laser_collision_dist = 0.05  # точка на ребре, если расхождение ее координаты и настоящей координаты ребра не больше этого (в клетках)
    lasers_on_one_node = 4  # столько лазеров проверяет один узел на пустоту; будет взято x2+1
    min_length_delta = 0.2  # каждый лазер должен проходить дальше пустого узла хотя бы на столько м

    v = 0.35
    vslow = 0.04
    acceleration = 0.2  # points per second
    pid_window_size = 10
    decel_start_offset = 1 # расстояние до начала замедления
    rotate_decel_start_offset = 29000  # расстояние до начала замедления
    maze_size = 15

    hamming_length = 12
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    colors = {
        2: 'green',
        4: 'blue',
        0: 'red',
        1: 'yellow'
    }


class Data:  # одометрия
    cur_angle = 0
    pid_error_deque = [0.0 for i in range(RobotConst.pid_window_size)]
    pid_deque_index = -1
    field = np.full((RobotConst.maze_size * 2 + 1, RobotConst.maze_size * 2 + 1), -1)
    field_cells = [[-1 for _ in range((RobotConst.maze_size + 1) * 2 + 1)]
                   for _ in range((RobotConst.maze_size + 1) * 2 + 1)]
    robot_pos = (RobotConst.maze_size, RobotConst.maze_size)
    borders = [1, 1, field.shape[0] - 2, field.shape[1] - 2]
    enc = {"left": 0, "right": 0}
    sensors = {'time_stamp': 0, 'angle': 0.0, 'angle_increment': 0.0, 'values': tuple([0.0 for _ in range(683)])}
    encL = 0
    encR = 0
    min_robot_pos = (RobotConst.maze_size, RobotConst.maze_size)
    max_robot_pos = (RobotConst.maze_size, RobotConst.maze_size)
    finish = False
    threads = []


# -1 - неизвестная клетка
#  0 - пустота
#  1 - препятствие
#  2 - граница поля


# Shortcut methods
robot = Robot()
detector = ARTagDetector(debug=DEBUG)
hamming = Hamming(RobotConst.hamming_length, debug=DEBUG)
eL = lambda: Data.enc['right']
eR = lambda: Data.enc['left']
sF = lambda: min(Data.sensors["values"][(683 // 2) - 40:(683 // 2) + 41])
sR = lambda: Data.sensors['values'][31]
sR90 = lambda: Data.sensors['values'][84]
sL = lambda: Data.sensors['values'][-32]
sL90 = lambda: Data.sensors['values'][-85]
# sA = lambda x: robot.getLaser()['values'][x]
node_value = lambda cell_dist: (cell_dist % 1 <= RobotConst.laser_node_collision_dist or
                                cell_dist % 1 >= (1 - RobotConst.laser_node_collision_dist))
edge_value = lambda cell_dist: (cell_dist % 1 <= RobotConst.laser_collision_dist or
                                cell_dist % 1 >= (1 - RobotConst.laser_collision_dist))
pi = math.pi
sleep = lambda x: tm.sleep(x / 1000)
inf = float('inf')
eps = 0.00000001

# getLasers() output:
# {'time_stamp': int,
#  'angle': float,
#  'angle_increment': float,
#  'values': tuple(683 floats)
# }
#

# Base methods
sign = lambda x: -1 if x < 0 else 1
time = lambda: tm.time() * 1000
getYaw = lambda: math.degrees(robot.getDirection()) * -1000
log = lambda *args, sep=' ', end='\n', file=None: print(*args, sep=sep, end=end, file=file) if DEBUG else None


def encoder_polling(delay=200):
    while True:
        if Data.finish:
            break

        try:
            sleep(delay)
            x = robot.getEncoders()
            if x is not None:
                Data.enc = x
        except rospy.exceptions.ROSInterruptException:
            break


def laser_polling(delay=200):
    while True:
        if Data.finish:
            break

        try:
            sleep(delay)
            x = robot.getLaser()
            if x is not None:
                Data.sensors = x
        except rospy.exceptions.ROSInterruptException:
            break


class SpiralGenerator:
    def __init__(self, size):
        self.dir = 1
        self.size = size
        self.iter_lim = size * size
        self.iteration = 0
        self.start_dir = (0, 0)
        self.cur = (0, -1)

    def __next__(self):
        if self.iter_lim == self.iteration:
            raise StopIteration
        if self.dir == 1:
            if self.cur[1] == self.start_dir[1] + self.size - 1:
                self.dir = 2
                self.cur = (self.cur[0] + 1, self.cur[1])
                self.start_dir = self.cur
                self.size -= 1
            else:
                self.cur = (self.cur[0], self.cur[1] + 1)
        elif self.dir == 2:
            if self.cur[0] == self.start_dir[0] + self.size - 1:
                self.dir = 3
                self.cur = (self.cur[0], self.cur[1] - 1)
                self.start_dir = self.cur
            else:
                self.cur = (self.cur[0] + 1, self.cur[1])
        elif self.dir == 3:
            if self.cur[1] == self.start_dir[1] - self.size + 1:
                self.dir = 4
                self.cur = (self.cur[0] - 1, self.cur[1])
                self.start_dir = self.cur
                self.size -= 1
            else:
                self.cur = (self.cur[0], self.cur[1] - 1)
        elif self.dir == 4:
            if self.cur[0] == self.start_dir[0] - self.size + 1:
                self.dir = 1
                self.cur = (self.cur[0], self.cur[1] + 1)
                self.start_dir = self.cur
            else:
                self.cur = (self.cur[0] - 1, self.cur[1])
        self.iteration += 1
        return self.cur

    def __iter__(self):
        return self


def print_field():
    for i in Data.field:
        for j in i:
            log(('-' if j == -1 else j), end=' ')
        log()
    log('=' * len(Data.field_cells) * 2)
    for i in Data.field_cells:
        for j in i:
            log(('-' if j == -1 else j), end=' ')
        log()


def motors(vL=None, vR=None):
    vL = RobotConst.v if vL is None else vL
    vR = vL if vR is None else vR

    delta = vL - vR
    linear = vR + delta / 2
    angular = (2 * pi) / (pi * RobotConst.track / (delta / 2)) if abs(delta) > eps else 0  # градусы в секунду.
    # Подразумевается, что скорость вращения
    # колёс в gazebo считается как сумма скоростей колёс при linear_speed+angular_speed. Плюс, считается, что все
    # линейные величины считаются в метрах, linear_speed в м/с, angular_speed в рад/с
    # log(linear, angular)
    robot.setVelosities(linear, angular)


def _drive_enc_(power=RobotConst.v):
    sgnL = sign(Data.encL - eL())
    sgnR = sign(Data.encR - eR())
    flagL = 0 if eL() < Data.encL else 1
    flagR = 0 if eR() < Data.encR else 1
    while [eL(), Data.encL][flagL] < [Data.encL, eL()][flagL] or [eR(), Data.encR][flagR] < [Data.encR, eR()][flagR]:
        if abs(Data.encL - eL()) < RobotConst.decel_start_offset:
            powerL = RobotConst.vslow * sgnL if [eL(), Data.encL][flagL] < [Data.encL, eL()][
                flagL] else 0
        else:
            powerL = max(0, min(power, RobotConst.v * 2)) * sgnL if [eL(), Data.encL][flagL] < [Data.encL, eL()][
                flagL] else 0
        if abs(Data.encR - eR()) < RobotConst.decel_start_offset:
            powerR = RobotConst.vslow * sgnR if [eR(), Data.encR][flagR] < [Data.encR, eR()][
                flagR] else 0
        else:
            powerR = max(0, min(power, RobotConst.v * 2)) * sgnR if [eR(), Data.encR][flagR] < [Data.encR, eR()][
                flagR] else 0
        # log(eL(), Data.encL, eR(), Data.encR)
        motors(powerL, powerR)
        if Data.finish:
            exit(1)
        sleep(10)
    motors(0)


def forward_enc_simplified(cm=RobotConst.cell_size):
    path = (cm / (pi * RobotConst.wheel_d)) * RobotConst.cpr
    Data.encL += path  # Add start encoder value
    Data.encR += path  # Difference correction
    _drive_enc_()


def rotate_enc_simplified_relative(angle):
    if abs(angle) < 200:
        angle *= 1000
    if abs(angle) >= 180000:
        angle -= 360000
    cm = RobotConst.track * pi * (angle / 360000)
    Data.encL += (cm / (pi * RobotConst.wheel_d)) * RobotConst.cpr
    Data.encR -= (cm / (pi * RobotConst.wheel_d)) * RobotConst.cpr
    Data.cur_angle += angle
    _drive_enc_()


def rotate_enc_simplified_absolute(angle):
    if abs(angle) < 200:
        angle *= 1000
    rotate_enc_simplified_relative(angle - Data.cur_angle)


def forward_enc_simplified_while(check_func, dir=1, power=RobotConst.v):
    """
    Едет по гироскому пока check_func возвращает True
    :param check_func: функция, по которой надо останавливать цикл (возвращает bool)
    :param dir: 1 - вперёд, -1 - назад
    :param power: скорость
    :return:
    """
    sgn = sign(dir)
    while check_func():
        motors(max(0, min(power, 100)) * sgn, max(0, min(power, 100)) * sgn)
        if Data.finish:
            exit(1)
        sleep(1)
    Data.encL = eL()
    Data.encR = eR()
    motors(0)


def forward_gyro(cm=float(RobotConst.cell_size), additional_corrections=lambda: 0.0):
    gyro_kp = 0.000005  # П коэффиент для езды по гироскопу
    path = (cm / (pi * RobotConst.wheel_d)) * RobotConst.cpr
    Data.encL += path  # Add start encoder value
    Data.encR += path  # Add start encoder value

    # start_time = time()
    sgn = sign(cm)
    # decel_offset = (Robot.decel_start_offset / (pi * Robot.wheelD)) * Robot.cpr
    # decel_start = None
    flag = 0 if eL() < Data.encL else 1
    while [eL(), Data.encL][flag] < [Data.encL, eL()][flag]:
        # power_delta = Robot.acceleration * ((time() - start_time) / 1000)
        # power_accel = min(Robot.vstart + power_delta, Robot.v)
        # power_decel = Robot.v
        # if decel_start is not None:
        #     power_delta_decel = Robot.acceleration * ((time() - decel_start) / 1000)
        #     power_decel = max(Robot.vstart, Robot.v - power_delta_decel)
        # power = min(power_accel, power_decel)
        extra_correction = additional_corrections()
        if abs(Data.encL - eL()) < RobotConst.decel_start_offset:
            power = RobotConst.vslow
        else:
            power = RobotConst.v
        error = Data.cur_angle - getYaw()  # Proportional steering
        if error > 180000:
            error -= 360000
        if error < -180000:
            error += 360000
        correction = error * gyro_kp
        motors(max(0, min(power + correction * sgn + extra_correction, RobotConst.v * 2)) * sgn,
               max(0, min(power - correction * sgn - extra_correction, RobotConst.v * 2)) * sgn)
        if Data.finish:
            exit(1)
        sleep(10)
    motors(0)
    # log(time() - decel_start)  # print the deceleration actual time after simulation


def forward_gyro_while(check_func, dir=1, power=RobotConst.v):
    """
    Едет по гироскому пока check_func возвращает True
    :param check_func: функция, по которой надо останавливать цикл (возвращает bool)
    :param dir: 1 - вперёд, -1 - назад
    :param power: скорость
    :return:
    """
    gyro_kp = 0.00002  # П коэффиент для езды по гироскопу
    sgn = sign(dir)
    while check_func():
        # log(Data.sensors["values"][(683 // 2) - 5:(683 // 2) + 6])
        error = Data.cur_angle - getYaw()  # Proportional steering
        if error > 180000:
            error -= 360000
        if error < -180000:
            error += 360000
        correction = error * gyro_kp
        motors(max(0, min(power + correction * sgn, 100)) * sgn, max(0, min(power - correction * sgn, 100)) * sgn)
        if Data.finish:
            exit(1)
        sleep(1)
    Data.encL = eL()
    Data.encR = eR()
    motors(0)


def rotate_gyro_absolute(angle):
    """
    Rotates robot to a given value (NOTE: accepts values from -180 to 180 degrees!!!)
    """
    if abs(angle) < 200:
        angle *= 1000
    delta = angle - Data.cur_angle
    if delta > 180000:
        delta -= 360000
    if delta < -180000:
        delta += 360000
    cm = RobotConst.track * pi * (delta / 360000)
    Data.encL += (cm / (pi * RobotConst.wheel_d)) * RobotConst.cpr
    Data.encR -= (cm / (pi * RobotConst.wheel_d)) * RobotConst.cpr
    Data.cur_angle = angle
    sgn = sign(angle - getYaw())
    if abs(angle - getYaw()) >= 180000:
        sgn *= -1
    if Data.finish:
        exit(1)
    while abs(angle - getYaw()) > 3000 and (angle > -180000 + 3000 or abs(-angle - getYaw()) > 3000):
        if abs(angle - getYaw()) < RobotConst.rotate_decel_start_offset or \
                (angle <= -180000 + RobotConst.rotate_decel_start_offset and
                 abs(-angle - getYaw()) < RobotConst.rotate_decel_start_offset):
            motors(RobotConst.vslow * 1.3 * sgn, RobotConst.vslow * 1.3 * -sgn)
        else:
            motors((RobotConst.v / 3 * 2) * sgn, (RobotConst.v / 3 * 2) * -sgn)
        sleep(1)
        if Data.finish:
            exit(1)
    motors(0)
    if abs(eL() - Data.encL) > 2 * math.pi or abs(eL() - Data.encL) > 2 * math.pi:
        Data.encL = eL()
        Data.encR = eR()
    log('Rotate', angle, 'done!')


def rotate_gyro_relative(angle):
    angle *= 1000
    rotate_gyro_absolute(((Data.cur_angle + angle + 180000) % 360000) - 180000)
    log('Rotate', angle, '(relative) done!')


def pid_controller(value: float, setpoint: float) -> float:
    """
    Calculates PID correction
    :param value: current sensor value
    :param setpoint: target sensor value
    :return: correction value
    """
    kp = 0.05  # П/ПИД регулятор коэффициент
    ki = 0.04  # И/ПИД регулятор коэффициент
    kd = 0.03  # Д/ПИД регулятор коэффициент
    err = value - setpoint
    Data.pid_deque_index = (Data.pid_deque_index + 1) % RobotConst.pid_window_size
    Data.pid_error_deque[Data.pid_deque_index] = err
    P = err * kp
    I = sum(Data.pid_error_deque) / 10 * ki
    D = (Data.pid_error_deque[Data.pid_deque_index] - Data.pid_error_deque[
        (Data.pid_deque_index + RobotConst.pid_window_size + 1) % RobotConst.pid_window_size]) * kd
    return P + I + D


def pid_correction_to_motors(base_speed: int, correction: float) -> None:
    """
    Runs motor based on calculated PID correction
    :param base_speed: default speed
    :param correction: PID correction value
    :return:
    """
    motors(min(max(base_speed * (1 - correction), 10), 100), min(max(base_speed * (1 + correction), 10), 100))
    if Data.finish:
        exit(1)


def get_neighbors(pos: tuple) -> list:
    """
    Получает смежные вершины графа (в лабиринте).
    Всё происходит в 3D
    :param pos: текущее положение (x, y, z)
    :return: [(x, y, z) * len(neighbors)]
    """
    ans = [((pos[0] + 1) % len(RobotConst.directions), pos[1], pos[2]),
           ((pos[0] - 1) % len(RobotConst.directions), pos[1], pos[2])]  # мы всегда можем повернуть в этой же клетке
    i = RobotConst.directions[pos[0]]
    if 0 <= pos[1] + i[0] < Data.field.shape[0] and 0 <= pos[2] + i[1] < Data.field.shape[1] \
            and Data.field[pos[1] + i[0], pos[2] + i[1]] < 1:
        ans.append((pos[0], pos[1] + i[0], pos[2] + i[1]))
    return ans


def get_orientation() -> int:
    return ((Data.cur_angle + 180000 - 45000) // 90000) % 4


def get_orientation_exact() -> int:
    return ((round(getYaw()) + 180000 - 45000) // 90000) % 4


def BFS3D(target: list) -> list:
    """
    По списку смежностей (трёхмерному)
    :param target: list(tuple(x,y,z)) - все целевые вершины
    :return: list(tuple) если найдено инчае пустой list
    """
    target_cp = target.copy()
    target = []
    for i in target_cp:
        target += [(j, i[0], i[1]) for j in range(len(RobotConst.directions))] if len(i) == 2 else [i]
    start = get_orientation(), Data.robot_pos[0], Data.robot_pos[1]
    log('START', start)
    log('TARGET', target)
    queue = [start]
    done = np.full((len(RobotConst.directions), *Data.field.shape), False)  # посмотрели ли вершину
    marked = np.full((len(RobotConst.directions), *Data.field.shape), -1, dtype="i,i,i")  # прошлая вершина
    while len(queue) > 0:
        if Data.finish:
            exit(1)
        cur = queue[0]
        queue.pop(0)
        if not done[cur]:
            done[cur] = True
            for i in get_neighbors(cur):
                if not done[i]:
                    queue.append(i)
                    marked[i] = cur
                    if i in target:  # если нашли
                        ans = []
                        j = i
                        ans.append(i)
                        while j != start:  # восстановление ответа
                            j = tuple(marked[j])
                            ans.append(j)
                            if Data.finish:
                                exit(1)
                        ans.reverse()
                        log("PATH:", ans)
                        return ans
    return []


def localize() -> None:
    while Data.borders[2] - Data.borders[0] >= RobotConst.maze_size:
        if Data.finish:
            exit(1)
        target = Data.max_robot_pos[0] + 1, -1
        for i in range(Data.field.shape[1]):
            if Data.field[Data.max_robot_pos[0] + 1, i] < 1:
                if target[1] == -1 or abs(target[1] - Data.robot_pos[1]) > abs(i - Data.robot_pos[1]):
                    target = (Data.max_robot_pos[0] + 1, i)
        path = BFS3D([target])
        if len(path) == 0 or target == (Data.max_robot_pos[0] + 1, -1):
            Data.borders[2] = Data.max_robot_pos[0]
            update_borders()
            log("No path found X")
        ride_path(path)
    log('X loc done!')
    while Data.borders[3] - Data.borders[1] >= RobotConst.maze_size:
        if Data.finish:
            exit(1)
        target = -1, Data.min_robot_pos[1] - 1
        for i in range(Data.field.shape[0]):
            if Data.field[i, Data.min_robot_pos[1] - 1] < 1:
                if target[0] == -1 or abs(target[0] - Data.robot_pos[0]) > abs(i - Data.robot_pos[0]):
                    target = (i, Data.min_robot_pos[1] - 1)
        path = BFS3D([target])
        if len(path) == 0 or target == (-1, Data.min_robot_pos[1] - 1):
            Data.borders[1] = Data.min_robot_pos[1]
            update_borders()
            log("No path found Y", Data.min_robot_pos[1] - 1)
        ride_path(path)
    log('Y loc done!')


def update_node(x, y, value):
    if Data.borders[0] <= x <= Data.borders[2] and Data.borders[1] <= y <= Data.borders[3]:
        Data.field[x, y] = value


def update_cell(ax, ay, bx, by, value):
    if ax > bx:
        ax, ay, bx, by = bx, by, ax, ay
    if ay > by:
        ay, by = ay - 1, by + 1
    if Data.borders[0] <= bx <= Data.borders[2] + 1 and Data.borders[1] <= by <= Data.borders[3] + 1:
        Data.field_cells[bx][by] = value


def update_cell_relative(ax, ay, bx, by, value):
    ax, ay = convert_relative_to_absolute(ax, ay)
    bx, by = convert_relative_to_absolute(bx, by)
    update_cell(ax, ay, bx, by, value)


def convert_relative_to_absolute(dx, dy):
    rx, ry = Data.robot_pos[0], Data.robot_pos[1]
    cur_dir = get_orientation()
    if cur_dir == 0:
        x, y = rx - dx, ry - dy
    elif cur_dir == 2:
        x, y = rx + dx, ry + dy
    elif cur_dir == 1:
        x, y = rx - dy, ry + dx
    else:
        x, y = rx + dy, ry - dx
    return x, y


def convert_absolute_to_relative(x, y):
    rx, ry = Data.robot_pos[0], Data.robot_pos[1]
    cur_dir = get_orientation()
    if cur_dir == 0:
        dx = rx - x
        dy = ry - y
    elif cur_dir == 2:
        dx = x - rx
        dy = y - ry
    elif cur_dir == 1:
        dx = y - ry
        dy = rx - x
    else:
        dx = ry - y
        dy = x - rx
    return dx, dy


def update_node_relative(dx, dy, value):
    # dx - клеток вперед относительно робота
    # dy - клеток влево относительно робота
    x, y = convert_relative_to_absolute(dx, dy)
    update_node(x, y, value)


def update_obstacles():
    # -90 = #598  (~0.3)
    # -45 = #469  (~0.423 on left-forward, ~0.205 on forward)
    #   0 = #341  (не рекомендуется использовать для obstacles)
    #  45 = #213  (~0.423 on right-forward, ~0.205 on forward)
    #  90 = #85   (~0.3)

    start_time = time()

    # Важно: теперь лазеры слева направо
    data = robot.getLaser()
    lasers_delta_angle = data['angle_increment']
    distances = list(data['values'])[::-1]

    cell_rx, cell_ry = Data.robot_pos[0], Data.robot_pos[1]
    update_node(cell_rx, cell_ry, 0)
    update_cell(cell_rx, cell_ry, cell_rx - 1, cell_ry - 1, 0)
    update_cell(cell_rx, cell_ry, cell_rx - 1, cell_ry + 1, 0)
    update_cell(cell_rx, cell_ry, cell_rx + 1, cell_ry - 1, 0)
    update_cell(cell_rx, cell_ry, cell_rx + 1, cell_ry + 1, 0)

    edges = collections.defaultdict(int)
    for num, dist in enumerate(distances):
        if dist < 0.25:
            continue
        if dist == float('inf'):
            continue
        angle = lasers_delta_angle * (num - 85)
        # dx и dy - относительно робота (!) вперед и влево
        dx = math.sin(angle) * dist + RobotConst.forward_laser_delta
        dy = math.cos(angle) * dist

        cell_dx = dx / RobotConst.exact_cell_size
        cell_dy = dy / RobotConst.exact_cell_size
        if node_value(cell_dx) and node_value(cell_dy):
            # Laser endpoint is close to node
            continue

        if edge_value(cell_dx):
            cell_dx = round(cell_dx)
            ay = math.floor(cell_dy)
            by = math.ceil(cell_dy)
            edges[((cell_dx, ay), (cell_dx, by))] += 1
        elif edge_value(cell_dy):
            cell_dy = round(cell_dy)
            ax = math.floor(cell_dx)
            bx = math.ceil(cell_dx)
            edges[((ax, cell_dy), (bx, cell_dy))] += 1

    for points, cnt in edges.items():
        if cnt < RobotConst.min_lasers_on_edge:
            # Untrusted laser
            continue
        a, b = points
        ax, ay = a
        bx, by = b
        edges_now = [(a, b)]
        if ax == bx:
            if ax > 0:
                edges_now += [((ax + 1, ay), (bx + 1, by)),
                              ((ax + 2, ay), (bx + 2, by))]
                update_cell_relative(ax, ay, ax + 1, by, 1)
                update_cell_relative(ax + 1, ay, ax + 2, by, 1)
                update_cell_relative(ax - 1, ay, ax, by, 0)
            else:
                edges_now += [((ax - 1, ay), (bx - 1, by)),
                              ((ax - 2, ay), (bx - 2, by))]
                update_cell_relative(ax, ay, ax - 1, by, 1)
                update_cell_relative(ax - 1, ay, ax - 2, by, 1)
                update_cell_relative(ax + 1, ay, ax, by, 0)
        elif ay == by:
            if ay > 0:
                edges_now += [((ax, ay + 1), (bx, by + 1)),
                              ((ax, ay + 2), (bx, by + 2))]
                update_cell_relative(ax, ay, bx, by + 1, 1)
                update_cell_relative(ax, ay + 1, bx, by + 2, 1)
                update_cell_relative(ax, ay - 1, bx, by, 0)
            else:
                edges_now += [((ax, ay - 1), (bx, by - 1)),
                              ((ax, ay - 2), (bx, by - 2))]
                update_cell_relative(ax, ay, bx, by - 1, 1)
                update_cell_relative(ax, ay - 1, bx, by - 2, 1)
                update_cell_relative(ax, ay + 1, bx, by, 0)
        for a, b in edges_now:
            update_node_relative(*a, 1)
            update_node_relative(*b, 1)

    for x in range(Data.borders[0], Data.borders[2] + 1):
        for y in range(Data.borders[1], Data.borders[3] + 1):
            if Data.field[x, y] != -1:
                continue
            if x == cell_rx and y == cell_ry:
                continue

            dx, dy = convert_absolute_to_relative(x, y)
            distance = (((dx - RobotConst.exact_cell_size) ** 2 + dy ** 2) ** 0.5) * RobotConst.exact_cell_size
            angle = math.atan2((dx - RobotConst.exact_cell_size), dy)
            central_laser = 85 + round(angle / lasers_delta_angle)
            start, end = central_laser - RobotConst.lasers_on_one_node, central_laser + RobotConst.lasers_on_one_node
            if start < 0 or end >= len(distances) or distances[start] < 0.25 or distances[end] < 0.25:
                continue

            minimal, maximal = min(distances[start:end + 1]), max(distances[start:end + 1])
            if minimal >= distance + RobotConst.min_length_delta and maximal != float('inf'):
                update_node_relative(dx, dy, 0)
                update_cell_relative(dx, dy, dx + 1, dy + 1, 0)
                update_cell_relative(dx, dy, dx + 1, dy - 1, 0)
                update_cell_relative(dx, dy, dx - 1, dy + 1, 0)
                update_cell_relative(dx, dy, dx - 1, dy - 1, 0)

    print_field()
    log('CALC TIME:', time() - start_time)


def ride_path(path, need_to_exit=lambda: False):
    for i in range(1, len(path)):
        for j in range(i, len(path)):
            if Data.field[path[j][1:]] > 0:
                return
        if need_to_exit():
            return
        log(path[i - 1], '--->', path[i])
        if path[i][0] != path[i - 1][0]:
            if (path[i][0] - 1) % 4 == path[i - 1][0]:
                precise_90_degree_turn(1)
            elif (path[i][0] + 1) % 4 == path[i - 1][0]:
                precise_90_degree_turn(-1)
            else:
                raise AssertionError()
        else:
            forward_gyro((abs(path[i - 1][1] - path[i][1]) + abs(path[i - 1][2] - path[i][2])) * RobotConst.cell_size,
                         distance_corrector_proportional)

        sleep(250)

        Data.robot_pos = (path[i][1], path[i][2])
        Data.min_robot_pos = (min(path[i][1], Data.min_robot_pos[0]), min(path[i][2], Data.min_robot_pos[1]))
        Data.max_robot_pos = (max(path[i][1], Data.max_robot_pos[0]), max(path[i][2], Data.max_robot_pos[1]))
        update_obstacles()
        update_borders()
        log('POSITION:', Data.robot_pos)


def update_borders() -> None:
    """
    Обновляет массив поля согласно заданным границам (заполняет всё, что за пределами границ стенами)
    :return:
    """
    Data.borders[0] = max(Data.borders[0], Data.robot_pos[0] - (RobotConst.maze_size - 1))
    Data.borders[1] = max(Data.borders[1], Data.robot_pos[1] - (RobotConst.maze_size - 1))
    Data.borders[2] = min(Data.borders[2], Data.robot_pos[0] + (RobotConst.maze_size - 1))
    Data.borders[3] = min(Data.borders[3], Data.robot_pos[1] + (RobotConst.maze_size - 1))

    for i in range(Data.borders[0]):
        for j in range(Data.field.shape[1]):
            Data.field[i, j] = 2
    for j in range(Data.borders[1]):
        for i in range(Data.field.shape[0]):
            Data.field[i, j] = 2
    for i in range(Data.field.shape[0] - 1, Data.borders[2], -1):
        for j in range(Data.field.shape[1]):
            Data.field[i, j] = 2
    for j in range(Data.field.shape[1] - 1, Data.borders[3], -1):
        for i in range(Data.field.shape[0]):
            Data.field[i, j] = 2
    # ==============================
    for i in range(Data.borders[0]):
        for j in range(len(Data.field_cells[i])):
            Data.field_cells[i][j] = 2
    for j in range(Data.borders[1]):
        for i in range(len(Data.field_cells)):
            Data.field_cells[i][j] = 2
    for i in range(len(Data.field_cells) - 1, Data.borders[2] + 1, -1):
        for j in range(len(Data.field_cells[i])):
            Data.field_cells[i][j] = 2
    for j in range(len(Data.field_cells[0]) - 1, Data.borders[3] + 1, -1):
        for i in range(len(Data.field_cells)):
            Data.field_cells[i][j] = 2


def proximity_corrector():
    """
    Corrects the robot position according to the walls position
    :return:
    """
    if sF() <= 0.142:
        log("CORRECT TOO CLOSE FRONT")
        forward_gyro_while(lambda: sF() < 0.146, -1, RobotConst.vslow)
        sleep(250)
    if 0.152 <= sF() <= 0.3:
        log("CORRECT TOO FAR FRONT")
        forward_gyro_while(lambda: sF() > 0.148, 1, RobotConst.vslow)
        sleep(250)


def precise_90_degree_turn(dir):
    """
    Turns robot in a maze by 90 degrees precisely
    :param dir: 1 - clockwise, -1 - counterclockwise
    :return:
    """
    proximity_corrector()

    if sign(dir) > 0 and sL() < 0.437 and abs(0.317 - sL()) > 0.05:
        rotate_gyro_relative(-90)
        proximity_corrector()
        rotate_gyro_relative(90)
    elif sign(dir) < 0 and sR() < 0.437 and abs(0.317 - sR()) > 0.05:
        rotate_gyro_relative(90)
        proximity_corrector()
        rotate_gyro_relative(-90)
    rotate_gyro_relative(sign(dir) * 90)
    proximity_corrector()


def watchdog(time_limit):
    """
    Watches the encoders value. If their value doesn't change much for more than
    time_limit ms - it kills the program
    :param time_limit:
    :return:
    """
    last_val1 = eL()
    last_val2 = eR()
    watch_timestamp = time()
    while True:
        if abs(eL() - last_val1) > 0.001 and abs(eR() - last_val2) > 0.001:
            last_val1 = eL()
            last_val2 = eR()
            watch_timestamp = time()
        if time() - watch_timestamp > time_limit:
            Data.finish = True
            break
        sleep(100)


def distance_corrector_proportional():
    """
    For maintaining the good distance to the wall while riding through the maze cells
    :return: correction value
    """
    kp = 0.5
    correction = 0
    if sL90() < 0.38 and sL() < 0.437:
        correction = 0.317 - sL()
    elif sR90() < 0.38 and sR() < 0.437:
        correction = sR() - 0.317
    # log(sL(), sR())
    return correction * kp


# Main code
def main():
    log('Start')
    start_time = time()

    Data.cur_angle = (get_orientation_exact() - 1) * 90000
    rotate_gyro_relative(0)

    log('Localize')  # Локализация
    update_obstacles()
    update_borders()
    localize()
    log('Location succeed!')
    print_field()

    log('Search start')  # Поиск близжайшей тумбы к стене
    status = False
    pos = (-1, -1)  # Координаты близжайшей тумбы к стене
    while not status:
        if Data.finish:
            exit(1)
        log('Searching again...')
        status, pos = spiral_obstacle_search()

    status = False  # Определение цвета близжайшей тумбы
    color = -1
    center = (-1, -1)
    first_run = False
    while (not status) or (color not in RobotConst.colors):
        log("Try ride_close")
        status, area, center = ride_close(pos, first_run)
        first_run = True
        if not status:
            continue
        log("Color detect")
        status, color = detect_color(robot.getImage(), area)
    print('color', RobotConst.colors[color])

    if color != 1:
        full_localize()
        scan_boxes(1, center)

    targets = find_way_outs()  # Поиск артага на тумбе, куда подъехали

    if Data.robot_pos[0] % 2 != 1 or Data.robot_pos[1] % 2 != 1:
        log("DRIVE TO INT NODE")
        int_nodes = get_int_nodes()  # едем до близжайшей целой позиции
        path = BFS3D(int_nodes)
        if len(path) == 0:
            log("NO WAY FOUND INT NODE")
        ride_path(path)

    robot_pos = robot_abs_pos(color)
    print('coordinates', robot_pos[0] // 2, robot_pos[1] // 2)
    robot.sleep(10)

    status = -1
    code = ""
    while status < 0:
        while (get_orientation(), Data.robot_pos[0], Data.robot_pos[1]) not in targets:
            path = BFS3D(targets)
            if len(path) == 0:
                log("NO WAY FOUND")
                exit(0)
            ride_path(path)
        status, code = detector.detect(robot.getImage())
        targets.remove((get_orientation(), Data.robot_pos[0], Data.robot_pos[1]))

    print('found marker')
    robot.sleep(10)

    while status > 0:  # считывание артага
        if status == 1:
            rotate_gyro_relative(-10)
        elif status == 2:
            rotate_gyro_relative(10)
        sleep(100)
        status, code = detector.detect(robot.getImage())
    x, y = -1, -1
    if status == 0:
        status, parsed_code = hamming.decode(code)
        if not status:
            log("ERROR PARSING", code)
            exit(0)
        x, y = parse_artag_coordinates(parsed_code)
        print('marker', x, y)
    else:
        log("Error, detector returned code", status)
        exit(0)
    rotate_gyro_absolute(0)
    if x == -1 or y == -1:
        log("Error, coordinates not valid")

    target = robot_abs_pos_to_relative(color, (x*2, y*2))
    target = target[0], target[1]
    log("Finish coords:", target)

    while (Data.robot_pos[0], Data.robot_pos[1]) != target:
        path = BFS3D([target])
        if len(path) == 0:
            log("NO WAY FOUND")
            exit(0)
        ride_path(path)

    print('finish')

    log('Success!!!')
    log('Exec time:', (time() - start_time) / 1000)


def scan_boxes(target_color, closest_center):
    center_nodes = centers_of_boxes()
    if closest_center in center_nodes:
        center_nodes.remove(closest_center)
    target_list = get_near_points_from_centers(center_nodes)
    while len(target_list) > 0:
        path = BFS3D(target_list)
        if len(path) == 0:
            break
        ride_path(path)
        if (get_orientation(), Data.robot_pos[0], Data.robot_pos[1]) in target_list:
            status, color = detect_color(robot.getImage(), 0)
            if status and (color == target_color):
                log("SCAN SUCCESS")
                return
            elif status:
                box_nodes = find_way_outs()
                for i in box_nodes:
                    if i in target_list:
                        target_list.remove(i)
    log("SCAN FAIL")


def get_int_nodes():
    ans = []
    for i in range(Data.borders[0], Data.borders[2] + 1):
        for j in range(Data.borders[1], Data.borders[3] + 1):
            if i % 2 == 1 and j % 2 == 1:
                ans.append((i, j))
    return ans


def full_localize():
    log("START FULL LOCALIZE")
    while True:
        unknown_nodes = []
        for i in range(Data.borders[0], Data.borders[2] + 1):
            for j in range(Data.borders[1], Data.borders[3] + 1):
                if Data.field[i, j] < 0:
                    unknown_nodes.append((i, j))
        if len(unknown_nodes) == 0:
            break
        path = BFS3D(unknown_nodes)
        if len(path) == 0:
            break
        ride_path(path, lambda: True if Data.field[path[-1][1], path[-1][2]] > -1 else False)
    log("FULL LOCALIZE DONE")


def robot_abs_pos(color):
    if Data.borders[0] < RobotConst.maze_size // 2:
        if Data.borders[1] < RobotConst.maze_size // 2:
            green = RobotConst.maze_size - Data.robot_pos[0], RobotConst.maze_size - Data.robot_pos[1]
        else:
            green = Data.robot_pos[1] - RobotConst.maze_size, RobotConst.maze_size - Data.robot_pos[0]
    elif Data.borders[1] < RobotConst.maze_size // 2:
        green = RobotConst.maze_size - Data.robot_pos[1], Data.robot_pos[0] - RobotConst.maze_size
    else:
        green = Data.robot_pos[0] - RobotConst.maze_size, Data.robot_pos[1] - RobotConst.maze_size

    if color == 2:  # green
        return green
    elif color == 4:  # blue
        return green[1], (RobotConst.maze_size - 1) - green[0]
    elif color == 0:  # red
        return (RobotConst.maze_size - 1) - green[0], (RobotConst.maze_size - 1) - green[1]
    elif color == 1:  # yellow
        return (RobotConst.maze_size - 1) - green[1], green[0]
    return green


def robot_abs_pos_to_relative(color, coords):
    green = (-1, -1)
    if color == 2:  # green
        green = coords
    elif color == 4:  # blue
        green = (RobotConst.maze_size - 1) - coords[1], coords[0]
    elif color == 0:  # red
        green = (RobotConst.maze_size - 1) - coords[0], (RobotConst.maze_size - 1) - coords[1]
    elif color == 1:  # yellow
        green = coords[1], (RobotConst.maze_size - 1) - coords[0]
    else:
        return green

    if Data.borders[0] < RobotConst.maze_size // 2:
        if Data.borders[1] < RobotConst.maze_size // 2:  # вверх влево
            return RobotConst.maze_size - green[0], RobotConst.maze_size - green[1]
        else:  # вверх вправо
            return RobotConst.maze_size - green[1], RobotConst.maze_size + green[0]
    elif Data.borders[1] < RobotConst.maze_size // 2:  # вниз влево
        return RobotConst.maze_size + green[1], RobotConst.maze_size - green[0]
    else:  # вниз вправо
        return RobotConst.maze_size + green[0], RobotConst.maze_size + green[1]


def centers_of_boxes():
    res = []
    for x in range(Data.borders[0], Data.borders[2] + 1):
        for y in range(Data.borders[1], Data.borders[3] + 1):
            if Data.field_cells[x][y] == Data.field_cells[x][y + 1] == \
                    Data.field_cells[x + 1][y] == Data.field_cells[x + 1][y + 1] == 1:
                res.append((x, y))
    return res


def detect_color(image: np.ndarray, search_area):
    status, code = detector.detect(image)
    if search_area == 0 and status > -1:
        return True, 1

    src = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    left = src[720 // 2 - 2: 720 // 2 + 3, (1280 // 2 - 160) - 2:(1280 // 2 - 160) + 3]
    right = src[720 // 2 - 2: 720 // 2 + 3, (1280 // 2 + 160) - 2:(1280 // 2 + 160) + 3]
    left_hsv = cv2.cvtColor(left, cv2.COLOR_BGR2HSV)
    right_hsv = cv2.cvtColor(right, cv2.COLOR_BGR2HSV)
    mean_left = left_hsv.mean(axis=0).mean(axis=0)
    mean_right = right_hsv.mean(axis=0).mean(axis=0)
    if (search_area == 1 and mean_left[1] < 127) or (search_area == 2 and mean_right[1] < 127) \
            or (search_area == 0 and (mean_left[1] < 127 or mean_right[1] < 127)):
        return False, -1
    left_type = round(mean_left[0] / 30)
    right_type = round(mean_right[0] / 30)
    if search_area == 0 and right_type != left_type:
        return False, -1
    if search_area == 0:
        return True, left_type
    if search_area == 1:
        return True, left_type
    if search_area == 2:
        return True, right_type


def get_near_points_from_centers(centers: list) -> list:
    res = set()
    for c in centers:
        centerX, centerY = c

        res.add((0, centerX + 2, centerY))
        res.add((1, centerX, centerY - 2))
        res.add((2, centerX - 2, centerY))
        res.add((3, centerX, centerY + 2))
    return list(res)


def find_way_outs():
    posX, posY = Data.robot_pos
    direction = get_orientation()
    centerX = centerY = -1
    if direction == 0:
        centerX = posX - 2
        centerY = posY
    elif direction == 1:
        centerY = posY + 2
        centerX = posX
    elif direction == 2:
        centerX = posX + 2
        centerY = posY
    elif direction == 3:
        centerY = posY - 2
        centerX = posX

    coords = [
        (0, centerX + 2, centerY),
        (1, centerX, centerY - 2),
        (2, centerX - 2, centerY),
        (3, centerX, centerY + 2)
    ]
    return coords


def get_cell(cell: tuple) -> int:
    px, py = cell
    if px < 0 or py < 0:
        return 2
    if len(Data.field_cells) <= px or len(Data.field_cells) <= py:
        return 2
    return Data.field_cells[px][py]


def find_lost_cells(cells: list):
    if len(cells) == 0:
        return True
    for i in cells:
        nodes = [i, (i[0] - 1, i[1]), (i[0], i[1] - 1), (i[0] - 1, i[1] - 1)]
        while Data.robot_pos not in nodes:
            path = BFS3D(nodes)
            if len(path) == 0:
                return False
            ride_path(path, lambda: True if Data.field_cells[i[0]][i[1]] > -1 else False)
    return True


def ride_close(pos: tuple, is_first_run: bool):
    """
    Подъехать максимально близко к тумбе на pos
    """
    # Посчитать все координаты тумбы
    cell_borders = [Data.borders[0], Data.borders[1], Data.borders[2] + 1, Data.borders[3] + 1]

    res_box = list()
    res_box.append(pos)
    px, py = pos
    minX = maxX = minY = maxY = False
    toWall = min(px - cell_borders[0],
                 cell_borders[2] - px,
                 py - cell_borders[1],
                 cell_borders[3] - py)
    if toWall == px - cell_borders[0]:
        minX = True
    elif toWall == cell_borders[2] - px:
        maxX = True
    elif toWall == py - cell_borders[1]:
        minY = True
    elif toWall == cell_borders[3] - py:
        maxY = True
    print_field()
    Xmove = Ymove = 0
    if minX:
        Xmove = 1
        res_box.append((px + 1, py))
    if minY:
        Ymove = 1
        res_box.append((px, py + 1))
    if maxX:
        Xmove = -1
        res_box.append((px - 1, py))
    if maxY:
        Ymove = -1
        res_box.append((px, py - 1))
    while len(res_box) != 4:
        lost_cells = []
        if minX or maxX:
            if get_cell((px, py + 1)) == 0\
                    or get_cell((px + Xmove, py + 1)) == 0 or get_cell((px, py - 1)) == 1:
                res_box.append((px, py - 1))
                res_box.append((px + Xmove, py - 1))

            elif get_cell((px, py - 1)) == 0\
                    or get_cell((px + Xmove, py - 1)) == 0 or get_cell((px, py + 1)) == 1:
                res_box.append((px, py + 1))
                res_box.append((px + Xmove, py + 1))
            else:
                lost_cells.append((px, py + 1))
                lost_cells.append((px, py - 1))

        if minY or maxY:
            if get_cell((px + 1, py)) == 0 or\
                    get_cell((px + 1, py + Ymove)) == 0 or get_cell((px - 1, py)) == 1:
                res_box.append((px - 1, py))
                res_box.append((px - 1, py + Ymove))

            elif get_cell((px - 1, py)) == 0 \
                    or get_cell((px - 1, py + Ymove)) == 0 or get_cell((px + 1, py)):
                res_box.append((px + 1, py))
                res_box.append((px + 1, py + Ymove))
            else:
                lost_cells.append((px + 1, py))
                lost_cells.append((px - 1, py))
        status = find_lost_cells(lost_cells)
        if not status:
            return False, -1, (-1, -1)
    target_list = []

    target_dict = {}

    p1, p2, p3, p4 = res_box
    centerX = min((p1[0], p2[0], p3[0], p4[0]))
    centerY = min((p1[1], p2[1], p3[1], p4[1]))


    path = []
    for i in range(1, 3):
        # ЦЕНТРАЛЬНЫЕ
        target_list.append((3, centerX, centerY + 1 + i))  # -Y
        target_dict[(3, centerX, centerY + 1 + i)] = 0

        target_list.append((0, centerX + 1 + i, centerY))  # -X
        target_dict[(0, centerX + 1 + i, centerY)] = 0

        target_list.append((1, centerX, centerY - 1 - i))  # +Y
        target_dict[(1, centerX, centerY - 1 - i)] = 0

        target_list.append((2, centerX - 1 - i, centerY))  # +X
        target_dict[(2, centerX - 1 - i, centerY)] = 0

        if i == 1:
            if not is_first_run and (get_orientation(), Data.robot_pos[0], Data.robot_pos[1]) in target_dict:
                target_dict.pop((get_orientation(), Data.robot_pos[0], Data.robot_pos[1]))
                target_list.remove((get_orientation(), Data.robot_pos[0], Data.robot_pos[1]))
            path = BFS3D(target_list)
            if len(path) > 0:
                break

        # Низ БОКОВЫЕ (+y)
        target_list.append((1, centerX - 1, centerY - i - 1))
        target_dict[(1, centerX - 1, centerY - i - 1)] = 2

        target_list.append((1, centerX + 1, centerY - i - 1))
        target_dict[(1, centerX + 1, centerY - i - 1)] = 1

        # ЛЕВО БОКОВЫЕ (+x)
        target_list.append((2, centerX - 1 - i, centerY - 1))
        target_dict[(2, centerX - 1 - i, centerY - 1)] = 1

        target_list.append((2, centerX - 1 - i, centerY + 1))
        target_dict[(2, centerX - 1 - i, centerY + 1)] = 2

        # Верх БОКОВЫЕ (-y)
        target_list.append((3, centerX + 1, centerY + 1 + i))
        target_dict[(3, centerX + 1, centerY + 1 + i)] = 2

        target_list.append((3, centerX - 1, centerY + i + 1))
        target_dict[(3, centerX - 1, centerY + 1 + i)] = 1

        # ПРАВО БОКОВЫЕ (-x)
        target_list.append((0, centerX + 1 + i, centerY + 1))
        target_dict[(0, centerX + 1 + i, centerY + 1)] = 1

        target_list.append((0, centerX + 1 + i, centerY - 1))
        target_dict[(0, centerX + 1 + i, centerY - 1)] = 2

        if not is_first_run and (get_orientation(), Data.robot_pos[0], Data.robot_pos[1]) in target_dict:
            target_dict.pop((get_orientation(), Data.robot_pos[0], Data.robot_pos[1]))
            target_list.remove((get_orientation(), Data.robot_pos[0], Data.robot_pos[1]))
        path = BFS3D(target_list)
        if len(path) > 0:
            break

    if len(path) == 0:
        return False, -1, (centerX, centerY)
    ride_path(path)

    if (get_orientation(), Data.robot_pos[0], Data.robot_pos[1]) not in target_dict:
        return False, -1, (centerX, centerY)

    return True, target_dict[(get_orientation(), Data.robot_pos[0], Data.robot_pos[1])], (centerX, centerY)


def parse_artag_coordinates(code: str):
    return int(code[2::-1], 2), int(code[5:2:-1], 2)


def spiral_obstacle_search():
    """
    Проходит по спирали от границы поля. (Выполняет одну итерацию поиска!)
    Если находит неизвестную клетку - пытается построить до неё путь.
    Находит препятствие - выводит ответ
    :return: True если успешно найден ответ, False если нет
    """
    cell_borders = [Data.borders[0], Data.borders[1], Data.borders[2] + 1, Data.borders[3] + 1]
    for i in SpiralGenerator(cell_borders[2] - cell_borders[0] + 1):
        position = cell_borders[0] + i[0], cell_borders[1] + i[1]
        cell_type = Data.field_cells[position[0]][position[1]]
        if cell_type == -1:
            path = BFS3D([position, (position[0] - 1, position[1]),
                          (position[0], position[1] - 1), (position[0] - 1, position[1] - 1)])
            if len(path) == 0:
                continue
            ride_path(path, lambda: True if Data.field_cells[position[0]][position[1]] > -1 else False)
            return False, (-1, -1)
        elif cell_type == 1:
            ans = min(position[0] - cell_borders[0],
                      cell_borders[2] - position[0],
                      position[1] - cell_borders[1],
                      cell_borders[3] - position[1]) * RobotConst.exact_cell_size
            log('FOUND:', i, 'ANSWER:', ans)
            print('distance', round(ans, 1))
            return True, position


if __name__ == '__main__':
    # Thread(target=watchdog, args=(10000,), daemon=True).start()
    Data.threads.append(Thread(target=encoder_polling, args=(1,)))
    Data.threads.append(Thread(target=laser_polling, args=(100,)))
    for i in Data.threads:
        i.start()
    sleep(100)
    try:
        main()
    finally:
        robot.setVelosities(0, 0)
        Data.finish = True
        for i in Data.threads:
            i.join()

    log("Done")
