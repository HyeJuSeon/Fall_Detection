import os
import numpy as np
import cv2
import utils
from tqdm import tqdm

# BASE_DIR_URFD = os.path.join(os.getcwd(), 'urfd_pre_data')
BASE_DIR_URFD = os.path.join(os.getcwd(), 'urfd_pre_data_64')
# BASE_DIR_AIHUB_TRAIN = 'D:\Aihub_pre_data\Training\image'
# BASE_DIR_AIHUB_VAL = 'D:\Aihub_pre_data\Validation\image'
BASE_DIR_AIHUB_TRAIN = 'D:/aihub_pre_data_64/Training'
BASE_DIR_AIHUB_VAL = 'D:/aihub_pre_data_64/Validation'

def get_pose_point(image):
    # frame = cv2.imread(image)
    frame = utils.read_image(image)
    frame = cv2.resize(frame, dsize=(640, 480), interpolation=cv2.INTER_AREA) # urfd dataset frame size
    frame_height, frame_width = frame.shape[:2]
    threshold = 0.2
    num_points = 15
    # pretrained network setting
    proto_file = 'pose_deploy_linevec.prototxt'
    weights_file = 'pose_iter_160000.caffemodel'
    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)
    # GPU
    #net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    #net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    input_width = 368
    input_height = 368
    inp_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (input_width, input_height), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inp_blob)

    output = net.forward()

    output_height, output_width = output.shape[2:4]
    points = []
    for i in range(num_points):
        # confidence map
        prob_map = output[0, i, :, :]
        min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)
        # adjust the point position to match the original image
        x = int((frame_width * point[0]) / output_width)
        y = int((frame_height * point[1]) / output_height)
        if prob > threshold:
            points.append((x, y))
        else:
            points.append(None)
    return points

def draw_pose_point_n_line(image, background='image'):
    # frame = cv2.imread(image)
    pose_points = get_pose_point(image)
    if background == 'image':
        frame = utils.read_image(image)
        frame = cv2.resize(frame, dsize=(640, 480), interpolation=cv2.INTER_AREA) # urfd dataset frame size
    else:
        frame = np.zeros((480, 640, 3), np.uint8)

    for i in range(len(pose_points)):
        if pose_points[i]:
            x, y = int(pose_points[i][0]), int(pose_points[i][1])
            cv2.circle(frame, (x, y), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            # cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
    for pair in POSE_PAIRS:
        part_a, part_b = pair[0:2]
        if pose_points[part_a] and pose_points[part_b]:
            cv2.line(frame, pose_points[part_a], pose_points[part_b], (0, 255, 255), 2)
    return frame

def show_pose_points_n_lines(image):
    frame = draw_pose_point_n_line(image)
    winname = 'result'
    cv2.namedWindow(winname, flags=cv2.WINDOW_NORMAL)
    cv2.moveWindow(winname, 40, 30)
    cv2.imshow(winname, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindow()

def write_pose_image(image, path='', background='image'):
    frame = draw_pose_point_n_line(image, background=background)
    filename = path + image[-4:]
    cv2.imwrite(filename, frame)

def write_pose_image_urfd(background='image'):
    base_dir_pose = os.path.join(os.getcwd(), 'openpose_{}_urfd'.format(background))
    list_dir = os.listdir(BASE_DIR_URFD) # ['fall-01', 'fall-02', 'adl-01', 'adl-02', ...]

    for dir in tqdm(list_dir):
        path_origin = os.path.join(BASE_DIR_URFD, dir)
        path_pose = os.path.join(base_dir_pose, dir)
        utils.mkdir(path_pose)
        list_file = os.listdir(path_origin)
        for file in tqdm(list_file):
            filename_origin = os.path.join(path_origin, file)
            filename_pose = os.path.join(path_pose, file)
            write_pose_image(filename_origin, filename_pose, background=background)

def write_pose_image_aihub(phase='Training', background='image'):
    '''
    :param phase: 'Training' or 'Validation'.
    '''
    base_dir_pose = f'D:/openpose_{background}_aihub/{phase}'
    if phase == 'Training':
        base_dir_origin = BASE_DIR_AIHUB_TRAIN
    else:
        base_dir_origin = BASE_DIR_AIHUB_VAL
    list_dir = os.listdir(base_dir_origin) # ['[원천]inside', '[원천]outside', ...]
    for dir in tqdm(list_dir):
        path_origin = os.path.join(base_dir_origin, dir)
        path_pose = os.path.join(base_dir_pose, dir[4:])
        if os.path.isdir(path_pose):
            print(path_pose)
            continue
        utils.mkdir(path_pose)
        list_dir2 = os.listdir(path_origin) # ['FD_In_H11_01', 'FD_In_H11_02', ...]
        for dir2 in tqdm(list_dir2):
            path_origin2 = os.path.join(path_origin, dir2)
            path_pose2 = os.path.join(path_pose, dir2)
            utils.mkdir(path_pose2)
            list_file = os.listdir(path_origin2)
            for i, file in enumerate(list_file):
                filename_origin = os.path.join(path_origin2, file)
                filename_pose = os.path.join(path_pose2, str(i + 1))
                write_pose_image(filename_origin, filename_pose, background=background)

def detected_human(image):
    num_points = len(get_pose_point(image))
    if num_points < 10:
        return False
    return True


if __name__ == '__main__':
    POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7],
                  [1, 14], [14, 8], [8, 9], [9, 10], [14, 11], [11, 12], [12, 13]]
    show_pose_points_n_lines('test.jpg')
    show_pose_points_n_lines('test2.jpg')
    show_pose_points_n_lines('aihub/FD_In_H11H21H31_0001_20210112_09__083.0s.jpg')
    write_pose_image_urfd(background='black')
    write_pose_image_aihub()
    write_pose_image_aihub('Validation')
    write_pose_image_aihub(background='black')
    write_pose_image_urfd()
    write_pose_image_urfd(background='black')