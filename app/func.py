import math
import numpy as np
import time

class YWQZ:
    """仰卧起坐动作评估类，用于分析关键点数据并评估动作标准性"""
    def __init__(self, back_ground_threshold=20, knee_angle_low=90, knee_angle_high=120,
                 raise_ratio=2.5, neck_force_threshold=30):
        """
        初始化评估参数

        参数:
            back_ground_threshold: 背部贴地判断的阈值（像素）
            knee_angle_low: 膝盖弯曲角度下限（度）
            knee_angle_high: 膝盖弯曲角度上限（度）
            raise_ratio: 起身高度达标的比例（相对于起始姿势）
            neck_force_threshold: 颈部发力判断的距离阈值（像素）
        """
        self.back_ground_threshold = back_ground_threshold
        self.knee_angle_low = knee_angle_low
        self.knee_angle_high = knee_angle_high
        self.raise_ratio = raise_ratio
        self.neck_force_threshold = neck_force_threshold
        self.start_keypoints = None  # 存储起始帧关键点（参考姿势）

    @staticmethod
    def calculate_angle(p1, p2, p3):
        """计算三点形成的角度（p2为顶点）"""
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        # noinspection PyTypeChecker
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    def is_back_on_ground(self, keypoints):
        """判断背部是否贴地"""
        left_shoulder_y = keypoints[5][1]
        right_shoulder_y = keypoints[6][1]
        left_hip_y = keypoints[11][1]
        right_hip_y = keypoints[12][1]

        shoulder_hip_diff = (abs(left_shoulder_y - left_hip_y) +
                             abs(right_shoulder_y - right_hip_y)) / 2
        return shoulder_hip_diff < self.back_ground_threshold

    def is_knee_bent(self, keypoints):
        """判断膝盖是否正确弯曲"""
        # 计算左膝角度（左髋-左膝-左脚踝）
        left_knee_angle = self.calculate_angle(
            keypoints[11][:2], keypoints[13][:2], keypoints[15][:2]
        )
        # 计算右膝角度（右髋-右膝-右脚踝）
        right_knee_angle = self.calculate_angle(
            keypoints[12][:2], keypoints[14][:2], keypoints[16][:2]
        )

        return (self.knee_angle_low < left_knee_angle < self.knee_angle_high and
                self.knee_angle_low < right_knee_angle < self.knee_angle_high)

    def is_raised_enough(self, current_keypoints):
        """判断上半身抬起高度是否达标"""
        if self.start_keypoints is None:
            raise ValueError("请先设置起始帧关键点（调用set_start_keypoints方法）")

        # 起始帧肩髋垂直距离
        start_diff = (abs(self.start_keypoints[5][1] - self.start_keypoints[11][1]) +
                      abs(self.start_keypoints[6][1] - self.start_keypoints[12][1])) / 2
        # 当前帧肩髋垂直距离
        current_diff = (abs(current_keypoints[5][1] - current_keypoints[11][1]) +
                        abs(current_keypoints[6][1] - current_keypoints[12][1])) / 2

        return current_diff > start_diff * self.raise_ratio

    def is_neck_force(self, keypoints):
        """判断是否使用颈部发力（错误动作）"""
        # 鼻子与左肩的距离
        nose_shoulder_dist = np.linalg.norm(
            np.array(keypoints[0][:2]) - np.array(keypoints[5][:2])
        )
        return nose_shoulder_dist < self.neck_force_threshold

    def set_start_keypoints(self, start_keypoints):
        """设置起始帧关键点（作为参考姿势）"""
        self.start_keypoints = start_keypoints

    def count_complete_reps(self, all_keypoints):
        """统计完整动作次数"""
        if not all_keypoints:
            return 0

        # 若未手动设置起始帧，则用第一帧作为起始参考
        if self.start_keypoints is None:
            self.set_start_keypoints(all_keypoints[0])

        rep_count = 0
        in_rep = False  # 是否处于一次动作周期中

        for keypoints in all_keypoints:
            is_start = self.is_back_on_ground(keypoints)
            is_raised = self.is_raised_enough(keypoints)

            if not in_rep and is_raised:
                # 从起始状态进入抬起状态，标记动作开始
                in_rep = True
            elif in_rep and is_start:
                # 从抬起状态回到起始状态，标记动作完成
                rep_count += 1
                in_rep = False

        return rep_count

    def evaluate_frame(self, keypoints):
        """评估单帧动作的各项指标"""
        return {
            "back_on_ground": self.is_back_on_ground(keypoints),
            "knee_bent": self.is_knee_bent(keypoints),
            "raised_enough": self.is_raised_enough(keypoints) if self.start_keypoints else None,
            "neck_force": self.is_neck_force(keypoints)
        }


    def generate_advice(self,all_keypoints):
        """根据多帧评估结果生成综合改进建议"""
        # 统计各指标的错误出现次数
        error_counts = {
            "back_not_ground": 0,
            "knee_not_bent": 0,
            "not_raised_enough": 0,
            "neck_force": 0
        }
        metrics_list = []
        for keypoints in all_keypoints:
            metrics_list.append(self.evaluate_frame(keypoints))
        for metrics in metrics_list:
            if not metrics["back_on_ground"]:
                error_counts["back_not_ground"] += 1
            if not metrics["knee_bent"]:
                error_counts["knee_not_bent"] += 1
            if metrics["raised_enough"] is False:
                error_counts["not_raised_enough"] += 1
            if metrics["neck_force"]:
                error_counts["neck_force"] += 1

        # 生成建议
        advice = []
        if error_counts["back_not_ground"] > len(metrics_list) * 0.3:
            advice.append("起始和回落时请将背部贴紧地面，避免腰部拱起")
        if error_counts["knee_not_bent"] > len(metrics_list) * 0.3:
            advice.append("保持膝盖弯曲90-120度，双脚平放地面")
        if error_counts["not_raised_enough"] > len(metrics_list) * 0.3:
            advice.append("起身时请抬高上半身，确保肩胛骨离开地面")
        if error_counts["neck_force"] > len(metrics_list) * 0.2:
            advice.append("注意不要用颈部发力，双手轻放胸前即可")

        return advice if advice else ["动作标准，继续保持！"]

def ywqz_count(y):
    # 处理空列表或长度不足的情况
    if not y or len(y) < 2:
        return 0

    count = 0
    flag = False
    count_list = [y[0]]  # 用第一个元素初始化，避免空列表问题

    # 只遍历到倒数第二个元素，确保i+1有效
    for i in range(len(y) - 1):
        # 基本条件判断
        if (y[i] <= y[i + 1] and not flag) or (y[i] >= y[i + 1] and flag):
            continue

        # 边界检查：确保不会访问i-3, i+3等越界索引
        valid_indices = True
        # 检查左侧索引
        for offset in [-3, -2, -1]:
            if i + offset < 0:
                valid_indices = False
                break
        # 检查右侧索引
        for offset in [1, 2, 3]:
            if i + offset >= len(y):
                valid_indices = False
                break

        # 如果索引有效才进行抖动检查
        if valid_indices:
            # 检查是否有明显抖动
            shake = any([
                abs(count_list[-1] - y[i - 3]) > 100,
                abs(count_list[-1] - y[i - 2]) > 100,
                abs(count_list[-1] - y[i - 1]) > 100,
                abs(count_list[-1] - y[i]) > 100,
                abs(count_list[-1] - y[i + 1]) > 100,
                abs(count_list[-1] - y[i + 2]) > 100,
                abs(count_list[-1] - y[i + 3]) > 100
            ])

            if shake:
                count += 1
                count_list.append(y[i])
                flag = not flag
    return math.floor(count / 2)


class PlankEvaluator:
    """平板支撑动作评估类，用于分析关键点数据、评估动作标准性并计时"""

    def __init__(self,
                 body_angle_tolerance=30,  # 身体直线角度容忍度（度）
                 hip_angle_low=140,  # 髋部角度下限（接近直线）
                 hip_angle_high=180,  # 髋部角度上限（不超过直线）
                 shoulder_angle_low=70,  # 肩部角度下限（不小于80度）
                 shoulder_angle_high=110):  # 肩部角度上限（不大于100度）
        """初始化评估参数"""
        self.body_angle_tolerance = body_angle_tolerance
        self.hip_angle_low = hip_angle_low
        self.hip_angle_high = hip_angle_high
        self.shoulder_angle_low = shoulder_angle_low
        self.shoulder_angle_high = shoulder_angle_high

        # 计时相关变量
        self.start_time = None
        self.end_time = None
        self.is_valid = False  # 当前是否为有效平板支撑姿态
        self.total_duration = 0  # 总有效时长（秒）

        # 评估结果缓存
        self.evaluation_history = []
        self.valid_frame = 0
    @staticmethod
    def calculate_angle(p1, p2, p3):
        """计算三点形成的角度（p2为顶点）"""
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    @staticmethod
    def calculate_line_slope(p1, p2):
        """计算两点连线的斜率（用于判断身体是否保持直线）"""
        if p2[0] - p1[0] == 0:
            return np.inf  # 垂直直线
        return (p2[1] - p1[1]) / (p2[0] - p1[0])

    def is_body_aligned(self, keypoints):
        """判断身体是否保持直线（肩、髋、踝连线）"""
        # 取左肩、左髋、左脚踝三点（右侧可作为备用）
        left_shoulder = keypoints[5][:2]
        left_hip = keypoints[11][:2]
        left_ankle = keypoints[15][:2]

        # 计算肩-髋和髋-踝的斜率差
        slope1 = self.calculate_line_slope(left_shoulder, left_hip)
        slope2 = self.calculate_line_slope(left_hip, left_ankle)

        # 斜率差在容忍范围内视为直线（转为角度差判断）
        angle_diff = abs(np.degrees(np.arctan(slope1)) - np.degrees(np.arctan(slope2)))
        return angle_diff < self.body_angle_tolerance

    def is_hip_angle_valid(self, keypoints):
        """判断髋部角度是否标准（不塌陷、不上翘）"""
        # 髋部角度：肩-髋-踝
        left_shoulder = keypoints[5][:2]
        left_hip = keypoints[11][:2]
        left_ankle = keypoints[15][:2]
        hip_angle = self.calculate_angle(left_shoulder, left_hip, left_ankle)

        return self.hip_angle_low < hip_angle < self.hip_angle_high

    def is_shoulder_angle_valid(self, keypoints):
        """判断肩部角度是否标准（不耸肩、不塌肩）"""
        # 肩部角度：肘-肩-髋
        left_wrist = keypoints[8][:2]
        left_shoulder = keypoints[6][:2]
        left_hip = keypoints[12][:2]
        shoulder_angle = self.calculate_angle(left_wrist, left_shoulder, left_hip)

        return self.shoulder_angle_low < shoulder_angle < self.shoulder_angle_high

    def is_elbow_position_valid(self, keypoints):
        """判断肘部是否位于肩部正下方"""
        # 肘部x坐标应接近肩部x坐标
        left_shoulder_x = keypoints[5][0]
        left_elbow_x = keypoints[7][0]
        x_diff = abs(left_shoulder_x - left_elbow_x)

        # 允许的x方向偏差（根据实际分辨率调整，这里用肩部宽度的1/3作为参考）
        shoulder_width = abs(keypoints[5][0] - keypoints[6][0])
        return x_diff < shoulder_width / 3

    def evaluate_frame(self, keypoints):
        """评估单帧动作是否标准"""
        # 检查关键点位是否有效（置信度>0.5）
        critical_points = [5, 6, 7, 8, 9, 10, 11, 12, 15, 16]  # 肩、肘、腕、髋、踝
        if any(keypoints[kp][2] < 0.2 for kp in critical_points):
            return {
                "valid": False,
                "reason": "关键点位识别不清",
                "body_aligned": None,
                "hip_angle": None,
                "shoulder_angle": None,
                "elbow_position": None
            }

        # 各项指标评估
        body_aligned = self.is_body_aligned(keypoints)
        hip_angle_valid = self.is_hip_angle_valid(keypoints)
        shoulder_angle_valid = self.is_shoulder_angle_valid(keypoints)
        elbow_valid = self.is_elbow_position_valid(keypoints)

        #print(shoulder_angle_valid)
        # 综合判断
        valid = body_aligned and hip_angle_valid and shoulder_angle_valid

        result = {
            "valid": valid,
            "reason": [] if valid else [
                "身体未保持直线" if not body_aligned else "",
                "髋部塌陷或上翘" if not hip_angle_valid else "",
                "肩部角度不标准" if not shoulder_angle_valid else "",
                "肘部未在肩部正下方" if not elbow_valid else ""
            ],
            "body_aligned": body_aligned,
            "hip_angle": hip_angle_valid,
            "shoulder_angle": shoulder_angle_valid,
            "elbow_position": elbow_valid
        }

        # 过滤空字符串原因
        result["reason"] = [r for r in result["reason"] if r]
        self.evaluation_history.append(result)
        #print(result)
        return result

    def update_timer(self, is_valid,current_timestamp: float):
        """根据当前帧是否有效更新计时器"""


        # 状态1：从无效→有效（开始计时）
        if is_valid and not self.is_valid:
            self.start_time = current_timestamp  # 记录开始时间
            self.is_valid = True

        # 状态2：从有效→无效（结束计时并累加）
        elif not is_valid and self.is_valid:
            if self.start_time is not None:
                # 计算当前有效时段的时长（避免start_time未初始化）
                duration = current_timestamp - self.start_time
                self.total_duration += max(0.0, duration)  # 确保时长非负
            self.is_valid = False

    def get_current_duration(self):
        """获取当前累计有效时长（秒）"""
        if self.is_valid and self.start_time is not None:
            # 若当前仍在有效状态，实时计算
            return self.total_duration + (time.time() - self.start_time)
        return self.total_duration
    def get_time(self,is_valid):
        if is_valid:
            self.valid_frame+=1


    def generate_advice(self):
        """根据历史评估结果生成综合建议"""
        if not self.evaluation_history:
            return ["暂无评估数据"]

        # 统计各错误出现频率
        error_counts = {
            "body_aligned": 0,
            "hip_angle": 0,
            "shoulder_angle": 0,
            "elbow_position": 0
        }

        for eval in self.evaluation_history:
            if not eval["body_aligned"]:
                error_counts["body_aligned"] += 1
            if not eval["hip_angle"]:
                error_counts["hip_angle"] += 1
            if not eval["shoulder_angle"]:
                error_counts["shoulder_angle"] += 1
            if not eval["elbow_position"]:
                error_counts["elbow_position"] += 1

        # 生成建议（优先显示最频繁的错误）
        advice = []
        total_frames = len(self.evaluation_history)

        if error_counts["body_aligned"] > total_frames * 0.3:
            advice.append("保持身体成一条直线，避免腰部塌陷或臀部上翘")
        if error_counts["hip_angle"] > total_frames * 0.3:
            advice.append("调整髋部位置，保持与身体其他部位成直线")
        if error_counts["shoulder_angle"] > total_frames * 0.3:
            advice.append("肩部不要过度前倾或后仰，保持自然弯曲")
        if error_counts["elbow_position"] > total_frames * 0.3:
            advice.append("确保肘部位于肩部正下方，不要外扩")

        return advice if advice else ["动作标准，继续保持！"]

    def reset(self):
        """重置评估器状态（用于新一轮评估）"""
        self.start_time = None
        self.end_time = None
        self.is_valid = False
        self.total_duration = 0
        self.evaluation_history = []
