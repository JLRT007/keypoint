import os
import time
import cv2
import math
import numpy as np
import json
from python.infer import Detector
from python.keypoint_infer import KeyPointDetector
from python.det_keypoint_unite_infer import KeypointSmoothing

class UniModel():
    def __init__(self):
        dirpath = os.path.dirname(os.path.abspath(__file__))
        self.detector = Detector(
            model_dir= os.path.join(dirpath, '../output_inference/picodet_v2_s_320_pedestrian'), 
            device='GPU')
        self.keypoint_detector = KeyPointDetector(
            model_dir=os.path.join(dirpath, '../output_inference/tinypose_256x192'), 
            device='GPU')
        
    def infer_img(self, img_file, vis=False):
        if type(img_file) == str:
            image = cv2.imread(img_file)
        else:
            image = img_file
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.predict_image([image], visual=False)
        results = self.detector.filter_box(results, threshold=0.5)
        if results['boxes_num'] > 0:
            keypoint_res = self.predict_with_given_det(image, results)
            keypoint_res['boxes_num'] = results['boxes_num']
        else:
            keypoint_res = {"keypoint": [[], []], "boxes_num": 0}
        if vis:
            canvas = self.visualize(img_file, keypoint_res)
            return canvas, keypoint_res
        return keypoint_res
        
    def predict_with_given_det(self, image, det_res):
        keypoint_res = {}
        rec_images, records, det_rects = self.keypoint_detector.get_person_from_rect(image, det_res)
        if len(det_rects) == 0:
            keypoint_res['keypoint'] = [[], []]
            return keypoint_res

        kp_results = self.keypoint_detector.predict_image(rec_images, visual=False)
        kp_results['keypoint'][..., 0] += np.array(records)[:, 0:1]
        kp_results['keypoint'][..., 1] += np.array(records)[:, 1:2]
        keypoint_res['keypoint'] = [
            kp_results['keypoint'].tolist(), kp_results['score'].tolist()
            ] if len(kp_results['keypoint']) > 0 else [[], []]
        keypoint_res['bbox'] = det_rects
        return keypoint_res

    def infer_vid(self, video_file=None, camera_id=-1):
        #ywqz_list = []
        results_list = []
        frame_timestamp_list = []
        if camera_id != -1:
            capture = cv2.VideoCapture(camera_id)
        else:
            capture = cv2.VideoCapture(video_file)
        out_path = './output.mp4'
        save_path = "keypoints_data.json"#关键点数据保存路径
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        print("fps: %d, frame_count: %d" % (fps, frame_count))
        fourcc = cv2.VideoWriter_fourcc(* 'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        index = 0
        keypoint_smoothing = KeypointSmoothing(width, height, filter_type='OneEuro', beta=0.05)
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            index += 1
            print('detect frame: %d' % (index))
            results = self.infer_img(frame, vis=False)
            #print(results)
            if results['boxes_num'] == 0:
                writer.write(frame)
                continue
            if len(results['keypoint'][0]) == 1:
                current_keypoints = np.array(results['keypoint'][0][0])
                smooth_keypoints = keypoint_smoothing.smooth_process(current_keypoints)
                results['keypoint'][0][0] = smooth_keypoints.tolist()
                results_list.append(results['keypoint'][0][0])#存关键点数据

            frame = self.visualize(frame, results)
            writer.write(frame)
        writer.release()
        dirpath = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(dirpath, 'output.mp4')
        print('output_video saved to: {}'.format(out_path))
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results_list, f, ensure_ascii=False, indent=2)
        print(f"关键点数据已保存到 {save_path}")
        return out_path,results_list,fps
                
    def visualize(self, img_file, results, visual_thresh=0.5):
        skeletons, scores = results['keypoint']
        if len(skeletons) == 0:
            return img_file
        skeletons = np.array(skeletons)
        kpt_nums = 17
        if len(skeletons) > 0:
            kpt_nums = skeletons.shape[1]
        if kpt_nums == 17:  #plot coco keypoint
            EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8),
                    (7, 9), (8, 10), (5, 11), (6, 12), (11, 13), (12, 14),
                    (13, 15), (14, 16), (11, 12)]
        else:  #plot mpii keypoint
            EDGES = [(0, 1), (1, 2), (3, 4), (4, 5), (2, 6), (3, 6), (6, 7), (7, 8),
                    (8, 9), (10, 11), (11, 12), (13, 14), (14, 15), (8, 12),
                    (8, 13)]
        NUM_EDGES = len(EDGES)
        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
            [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
            [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
        img = cv2.imread(img_file) if type(img_file) == str else img_file
        bboxs = results['bbox']
        for rect in bboxs:
            xmin, ymin, xmax, ymax = rect
            color = colors[0]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        canvas = img.copy()
        for i in range(kpt_nums):
            for j in range(len(skeletons)):
                if skeletons[j][i, 2] < visual_thresh:
                    continue
                cv2.circle(
                    canvas,
                    tuple(skeletons[j][i, 0:2].astype('int32')),
                    2,
                    colors[i],
                    thickness=-1)
        for i in range(NUM_EDGES):
            for j in range(len(skeletons)):
                edge = EDGES[i]
                if skeletons[j][edge[0], 2] < visual_thresh or skeletons[j][edge[
                        1], 2] < visual_thresh:
                    continue

                cur_canvas = canvas.copy()
                X = [skeletons[j][edge[0], 1], skeletons[j][edge[1], 1]]
                Y = [skeletons[j][edge[0], 0], skeletons[j][edge[1], 0]]
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1])**2 + (Y[0] - Y[1])**2)**0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY), int(mX)),
                                        (int(length / 2), 2),
                                        int(angle), 0, 360, 1)
                cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        return canvas