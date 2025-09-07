import math
import numpy as np


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
