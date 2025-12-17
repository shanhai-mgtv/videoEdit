import numpy as np
import json
import jsonlines
import cv2
import torchaudio



# Specify the path to the FFmpeg executable
# os.environ["TORCHAUDIO_EXTENSIONS_FFMPEG"] = "/usr/bin/ffmpeg"

def speech_file_to_array_fn(path):
    speech_array, sampling_rate = torchaudio.load(path, backend="ffmpeg")
    resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
    speech = resampler(speech_array).squeeze().numpy()  # 左声道，加载视频片段的全部音频数据，numpy 类型
    if len(speech.shape) == 2:
        speech = speech[0]
    return speech


# def index_generation(crt_i, max_n, N, padding='replicate'):
#     '''
#     padding: replicate | reflection | new_info | circle
#     '''
#     max_n = max_n - 1
#     n_pad = N // 2

#     return_l = []

#     for i in range(crt_i - n_pad, crt_i + n_pad + 1):
#         if i < 0:
#             if padding == 'replicate':
#                 add_idx = 0
#             elif padding == 'reflection':
#                 add_idx = -i
#             elif padding == 'new_info':
#                 add_idx = (crt_i + n_pad) + (-i)
#             elif padding == 'circle':
#                 add_idx = N + i
#             else:
#                 raise ValueError('Wrong padding mode')
#         elif i > max_n:
#             if padding == 'replicate':
#                 add_idx = max_n
#             elif padding == 'reflection':
#                 add_idx = max_n * 2 - i
#             elif padding == 'new_info':
#                 add_idx = (crt_i - n_pad) - (i - max_n)
#             elif padding == 'circle':
#                 add_idx = i - N
#             else:
#                 raise ValueError('Wrong padding mode')
#         else:
#             add_idx = i
#         return_l.append(add_idx)
#     if N % 2 == 0:
#         return np.array(return_l[1:])
#     else:
#         return np.array(return_l)

def index_generation(crt_i, max_n, N, padding='replicate'):
    '''
    Generate indices based on the current index and a window size.

    Parameters:
    - crt_i: Current central index
    - max_n: Maximum valid index (exclusive)
    - N: Size of the window
    - padding: Padding strategy ('replicate', 'reflection', 'new_info', 'circle')

    Returns:
    - A numpy array of generated indices.
    '''
    n_pad = N // 2
    return_l = []

    for i in range(crt_i - n_pad, crt_i + n_pad + 1):
        if i < 0:  # Handle negative indices
            add_idx = {
                'replicate': 0,
                'reflection': -i,
                'new_info': (crt_i + n_pad) + (-i),
                'circle': max_n + i
            }.get(padding, None)
        elif i > max_n - 1:  # Handle overflow indices
            add_idx = {
                'replicate': max_n - 1,
                'reflection': (max_n - 1) * 2 - i,
                'new_info': (crt_i - n_pad) - (i - max_n + 1),
                'circle': i - max_n
            }.get(padding, None)
        else:  # In-range indices
            add_idx = i

        if add_idx is None:
            raise ValueError(f"Invalid padding mode: {padding}")
        return_l.append(add_idx)

    # Adjust for even N by slicing
    return np.array(return_l[1:] if N % 2 == 0 else return_l)



def pose_to_bbox(keypoints: np.ndarray, expansion: float = 1.2) -> np.ndarray:
    """Get bounding box from keypoints.

    Args:
        keypoints (np.ndarray): Keypoints of person.
        expansion (float): Expansion ratio of bounding box.

    Returns:
        np.ndarray: Bounding box of person.
    """
    x = keypoints[:, 0]
    y = keypoints[:, 1]
    bbox = np.array([x.min(), y.min(), x.max(), y.max()])
    center = np.array([bbox[0] + bbox[2], bbox[1] + bbox[3]]) / 2
    bbox = np.concatenate([
        center - (center - bbox[:2]) * expansion,
        center + (bbox[2:] - center) * expansion
    ])
    return bbox

def process_pose_jsonl(json_file):

    face_landmarks = []
    face_bboxes = []
    null_indices = []
    
    with open(json_file, "r+", encoding="utf8") as f:
        for line in jsonlines.Reader(f):
            info = json.loads(line)

            face_score = np.array(info['score'][0][24:92])  # [24, 91] 68 points
            face_keypoint = np.array(info['keypoint'][0][24:92])
            face_bbox = pose_to_bbox(face_keypoint[face_score > 2])
            
            if len(face_keypoint[face_score > 2]) < 68:
                null_indices.append(info['frame_idx'])

            face_landmarks.append(face_keypoint.tolist())
            face_bboxes.append(face_bbox.tolist())

    return np.array(face_landmarks), face_bboxes, null_indices


def get_max_bbox(face_bboxes, frame_indices):
    
    max_bbox = get_bouding_box([face_bboxes[index] for index in frame_indices])
    return max_bbox

def get_max_bbox_all(face_bboxes):
    
    max_bbox = get_bouding_box([face_bboxes[index] for index in range(len(face_bboxes))])
    return max_bbox

def get_bouding_box(face_points):
    
    # 过滤掉None数据
    filtered_points = [point for point in face_points if point and None not in point]

    # 计算边界框
    if filtered_points:  # 确保至少有一个有效点
        min_x = min(p[0] for p in filtered_points)
        min_y = min(p[1] for p in filtered_points)
        max_x = max(p[2] for p in filtered_points)
        max_y = max(p[3] for p in filtered_points)
    else:
        raise ValueError("没有有效的关键点数据")

    cut_info = [min_x, min_y, max_x, max_y]
    return cut_info




def detect_blur_fft(image, size=60, thresh=20):
    image = cv2.cvtColor(image.asnumpy(), cv2.COLOR_RGB2GRAY)
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))

    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)

    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    return (mean, mean <= thresh)


def detect_blur_fft_cropped(image, size=60, thresh=20):
    image = cv2.cvtColor(image.permute(1, 2, 0).numpy() * 255, cv2.COLOR_RGB2GRAY)
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))

    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)

    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    return mean

def process_face_blur(indices, video_reader):
    blur_scores = []
    for index in indices:
        blur_score = detect_blur_fft(video_reader[index])
        blur_scores.append((index, blur_score))
    blur_scores = sorted(blur_scores, key=lambda x: x[1], reverse=True)

    return blur_scores[0][0]  # return index



def draw_facepose(canvas, lmks):
    
    eps = 0.01
    points = []
    for lmk in lmks:
        x, y = lmk
        x = int(x)
        y = int(y)
        points.append([x, y])
        if x > eps and y > eps:
            cv2.circle(canvas, (x, y), 3, (255, 0, 0), thickness=-1)
    canvas = draw_face_line(canvas, points)
    return canvas


def draw_convex_hull(canvas, kpts, start, end):
    points = np.array(kpts[start:end + 1])
    cnts = cv2.convexHull(points)
    cv2.drawContours(canvas, [cnts], -1, (0, 255, 0), 1)
 
 
def draw_line(canvas, kpts, start, end):
    lines = np.array(kpts[start:end + 1])
    for line in range(1, len(lines)):
        pts1 = lines[line - 1]
        pts2 = lines[line]
        cv2.line(canvas, pts1, pts2, (0, 255, 0), 1)

def draw_face_line(canvas, kpts):
    draw_convex_hull(canvas, kpts, 36, 41)  # 绘制右眼凸包
    draw_convex_hull(canvas, kpts, 42, 47)  # 绘制左眼凸包
    # draw_convex_hull(canvas, kpts, 48, 59)  # 绘制嘴外部凸包
    draw_line(canvas, kpts, 48, 59)
    cv2.line(canvas, kpts[59], kpts[48], (0, 255, 0), 1)

    draw_convex_hull(canvas, kpts, 60, 67)  # 绘制嘴内部凸包
 
    draw_line(canvas, kpts, 0, 16)  # 绘制脸颊点线
    draw_line(canvas, kpts, 17, 21)  # 绘制左眉毛点线
    draw_line(canvas, kpts, 22, 26)  # 绘制右眉毛点线
    draw_line(canvas, kpts, 27, 30)  # 绘制鼻子点线
    draw_line(canvas, kpts, 31, 35)  # 绘制鼻子点线
    return canvas


def get_mouth_mask(img_size, landmarks):
    
    # Mouth landmarks (48-67)
    mouth_points = [(int(landmarks[n][0]), int(landmarks[n][1])) for n in range(48, 68)]

    mouth_points_array = np.array(mouth_points)
    x, y, w, h = cv2.boundingRect(mouth_points_array)

    w_, h_ = img_size
    mouth_mask = np.zeros((h_, w_), dtype=np.uint8)

    mouth_mask[y:y+h, x:x+w] = 255
    return mouth_mask

def get_eyes_mask(img_size, landmarks):
    
    # left eye landmarks (42-47)
    left_eye_points = [(int(landmarks[n][0]), int(landmarks[n][1])) for n in range(42, 48)]

    left_eye_points_array = np.array(left_eye_points)
    x, y, w, h = cv2.boundingRect(left_eye_points_array)

    w_, h_ = img_size
    eyes_mask = np.zeros((h_, w_), dtype=np.uint8)

    eyes_mask[y:y+h, x:x+w] = 255

    # right eye landmarks (36-41)
    right_eye_points = [(int(landmarks[n][0]), int(landmarks[n][1])) for n in range(36,42)]
    right_eye_points_array = np.array(right_eye_points)
    x, y, w, h = cv2.boundingRect(right_eye_points_array)

    eyes_mask[y:y+h, x:x+w] = 255

    return eyes_mask

def find_valid_sequence(seq, start, null_indices, clip_len):

    while start <= len(seq) - clip_len:
        end = start + clip_len
        subseq = seq[start:end]

        if all(item not in null_indices for item in subseq):
            return subseq
        
        start += 1
    return False


def draw_mask(img_size, landmarks):
    
    mask_points_idx = list(range(2, 15)) + [33]  # [2, 14]
    
    # 提取关键点坐标
    mask_points = np.array([landmarks[i] for i in mask_points_idx], dtype=np.int32)
    
    # 创建一个空白的 mask
    w_, h_ = img_size
    mask = np.zeros((h_, w_), dtype=np.uint8)
    
    # 绘制多边形
    cv2.fillPoly(mask, [mask_points], (255))

    return mask
