import numpy as np
import cv2


class Hamming:
    def __init__(self, code_length, debug=False):
        self.code_length = code_length
        self.debug = debug
        self._calc_powers()

    log = lambda self, *args, sep=' ', end='\n', file=None: \
        print(*args, sep=sep, end=end, file=file) if self.debug else None

    def _calc_powers(self):
        self.powers = []
        x = 1
        while x <= self.code_length:
            self.powers.append(x)
            x *= 2

    def _separate_code(self, code: list):
        check, info = [], []
        for i in range(len(code)):
            if i + 1 in self.powers:
                check.append(code[i])
            else:
                info.append(code[i])
        return check, info

    def _combine_check_and_info(self, check: list, info: list):
        code = []
        cur_check_index, cur_info_index = 0, 0
        while len(code) < len(check) + len(info):
            if len(code) + 1 in self.powers:
                code.append(check[cur_check_index])
                cur_check_index += 1
            else:
                code.append(info[cur_info_index])
                cur_info_index += 1
        return code

    def _calculate_check(self, code: list):
        check = []
        for p in self.powers:
            now = 0
            for i in range(p - 1, len(code), p * 2):
                if i == p - 1:
                    now += sum(code[i + 1:i + p])
                else:
                    now += sum(code[i:i + p])
            check.append(now % 2)
        return check

    @staticmethod
    def _find_difference(a: list, b: list):
        if len(a) != len(b):
            raise NotImplementedError
        res = []
        for i in range(len(a)):
            if a[i] != b[i]:
                res.append(i)
        return res

    def _find_corrupted_bit(self, checks_diff: list):
        t = [self.powers[i] for i in checks_diff]
        return sum(t) - 1

    def decode(self, raw_code: str):
        """
        Accepts a byte-encoded hamming code and decodes a message (with error correction)
        :param raw_code: hamming code string (example: "110101011000")
        :return: status: bool (False if decoding failed, True if successful), parsed_code: str - decoded code in the
                same style as a given raw_code parameter
        """
        if len(raw_code) > self.code_length:
            raise NotImplementedError
        # while len(raw_code) < self.word_length:
        #     raw_code += '0'

        code = list(map(int, raw_code))
        check, info = self._separate_code(code)
        self.log(f"found: {code} = {check} and {info}")
        real_check = self._calculate_check(code)
        checks_diff = self._find_difference(check, real_check)
        if checks_diff:
            self.log(f"broke: {check} & {real_check} = {checks_diff}")
            corrupted = self._find_corrupted_bit(checks_diff)
            code[corrupted] = (code[corrupted] + 1) % 2
            check, info = self._separate_code(code)
            self.log(f"fixed: {code} = {check} + {info}")
            real_check = self._calculate_check(code)
            checks_diff = self._find_difference(check, real_check)
            if checks_diff:
                self.log(f"broke: {check} & {real_check} = {checks_diff}")
                return False, raw_code
        self.log(f" real: {code} = {check} and {info}")
        return True, ''.join(map(str, info))


class ARTagDetector:
    cell_count = 6
    threshold = 10  # порог бинаризации
    homo_size = 100

    def __init__(self, debug=True):
        self.debug = debug
        self.width = 720
        self.height = 1280
        self.shape = np.array([(0, 0), (self.homo_size, 0), (self.homo_size, self.homo_size), (0, self.homo_size)])

    log = lambda self, *args, sep=' ', end='\n', file=None: \
        print(*args, sep=sep, end=end, file=file) if self.debug else None

    @staticmethod
    def edge_detect(binary):
        edges = []
        _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for i, contour in zip(hierarchy[0], contours):
            approx = cv2.approxPolyDP(contour, cv2.arcLength(contour, True)*0.02, True)
            if len(approx) == 4 and cv2.isContourConvex(approx) and cv2.contourArea(approx) > 1000:
                if i[0] == -1 and i[1] == -1 and i[3] != -1:
                    edges.append(approx.reshape(-1, 2))
        return edges

    @staticmethod
    def stripe_detect(binary):
        edges = []
        _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) >= 4 and cv2.contourArea(contour) > 800:
                edges.append(contour)
        return edges

    def extract_ar_tag(self, binary, edges):
        ratios, _ = cv2.findHomography(edges[0], self.shape)
        tag = cv2.warpPerspective(binary.copy(), ratios, (self.homo_size, self.homo_size))
        tag_natural_size = cv2.resize(tag,
                                      dsize=None,
                                      fx=self.cell_count/self.homo_size,
                                      fy=self.cell_count/self.homo_size)
        return tag_natural_size

    @staticmethod
    def orient(tag: np.ndarray):
        if tag[1, 1] == 255:
            return True, cv2.rotate(tag, cv2.ROTATE_180)
        elif tag[1, -2] == 255:
            return True, cv2.rotate(tag, cv2.ROTATE_90_CLOCKWISE)
        elif tag[-2, 1] == 255:
            return True, cv2.rotate(tag, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif tag[-2, -2] == 255:
            return True, tag
        else:
            return False, tag

    def read_raw_value(self, tag: np.ndarray):
        ans = ''
        for j in range(2, self.cell_count - 2):
            ans += str(int(tag[1][j] == 0))
        for i in range(2, self.cell_count - 2):
            for j in range(1, self.cell_count - 1):
                ans += str(int(tag[i][j] == 0))
        for j in range(2, self.cell_count - 2):
            ans += str(int(tag[-2][j] == 0))
        return ans

    def detect(self, image):
        """
        Detects ARTag and parses it.
        :param image: Robot() raw image
        :return: status: int (0 - successful, -1 - no ARTag detected, 1 - go left, 2 - go right),
                 code: str ("" if status != 0)
        """
        src = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)[:-87, 4:-4]
        self.height, self.width = src.shape[:2]
        # grayscale = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        blur_color = cv2.medianBlur(src, 3)
        blur_color_hsv = cv2.cvtColor(blur_color, cv2.COLOR_BGR2HSV)
        green_binary = cv2.inRange(blur_color_hsv, (40, 100, 20), (50, 255, 255))
        blue_binary = cv2.inRange(blur_color_hsv, (107, 100, 20), (117, 255, 255))
        # blur = cv2.medianBlur(grayscale, 3)
        # _, binary = cv2.threshold(blur, self.threshold, 255, cv2.THRESH_BINARY)
        binary = cv2.bitwise_not(cv2.inRange(blur_color_hsv, (0, 0, 0), (0, 1, 30)))

        edges = self.edge_detect(binary)

        preview = src.copy()
        if len(edges) < 1 or len(edges[0]) != 4:
            green_cnts = self.stripe_detect(green_binary)
            blue_cnts = self.stripe_detect(blue_binary)
            if self.debug:
                cv2.drawContours(preview, green_cnts, -1, (0, 0, 255), 1)
                cv2.drawContours(preview, blue_cnts, -1, (0, 255, 255), 1)
            if len(green_cnts) == 0 and len(blue_cnts) == 0:
                return -1, ""
            elif len(green_cnts) > len(blue_cnts):
                return 2, ""
            else:
                return 1, ""

        if self.debug:
            cv2.drawContours(preview, edges, -1, (0, 255, 0), 2)
            # cv2.namedWindow('preview', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow("preview", preview)
            # cv2.waitKey(1)
            # return 1, ""

        tag_raw = self.extract_ar_tag(binary, edges)
        status, tag = self.orient(tag_raw)

        if not status:
            return 1, ""

        code_raw_str = self.read_raw_value(tag)
        self.log("ARTAG FOUND:", code_raw_str)
        return 0, code_raw_str
