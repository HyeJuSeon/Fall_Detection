import os
from tqdm import tqdm
import shutil
import numpy as np
import cv2
import utils
import openpose
import glob

BASE_DIR_URFD = os.path.join(os.getcwd(), 'urfd_data')
BASE_DIR_AIHUB = 'D:\senior_abnormal_video'

def get_base_dir_aihub(phase='Training', file='image'):
    '''
    :param phase: 'Training' or 'Validation'.
    :param file: 'image' or 'video'.
    '''
    base_dir = os.path.join(BASE_DIR_AIHUB, phase)
    return os.path.join(base_dir, file)

def EDA():
    base_dir = 'D:/balanced_openpose_aihub/Training'
    # base_dir = 'D:/balanced_openpose_aihub/Training'
    # base_dir = get_base_dir_aihub()
    num_fd_train = 0
    num_adl_train = 0
    num_fd_val = 0
    num_adl_val = 0
    list_dir = os.listdir(base_dir)
    for dir in tqdm(list_dir):
        path = os.path.join(base_dir, dir)
        if 'H' in dir:
            num_fd_train += len(os.listdir(path))
            # num_fd_train += len(os.listdir(os.path.join(path, dir[-9:])))
        else:
            num_adl_train += len(os.listdir(path))
            # num_adl_train += len(os.listdir(os.path.join(path, dir[7:])))
    base_dir = 'D:/balanced_openpose_aihub/Validation'
    # base_dir = 'D:/balanced_aihub/Validation'
    # base_dir = get_base_dir_aihub(phase='Validation')
    list_dir = os.listdir(base_dir)
    for dir in tqdm(list_dir):
        path = os.path.join(base_dir, dir)
        if 'H' in dir:
            num_fd_val += len(os.listdir(path))
            # num_fd_val += len(os.listdir(os.path.join(path, dir[-9:])))
        else:
            num_adl_val += len(os.listdir(path))
            # num_adl_val += len(os.listdir(os.path.join(path, dir[7:])))
    print('Num of FD train set:', num_fd_train)
    print('Num of ADL train set:', num_adl_train)
    print('Num of FD validation set:', num_fd_val)
    print('Num of ADL validation set:', num_adl_val)

def get_files_urfd(num_frames):
    ''' Get frames of URFD dataset as a 2-dim list.

    :param num_frames: Number of frames to extract.
    '''
    list_dir = os.listdir(BASE_DIR_URFD)
    result = []
    for dir in list_dir:
        path = os.path.join(BASE_DIR_URFD, dir)
        list_all_files = os.listdir(path)
        step = int(len(list_all_files) / num_frames)
        list_file = []
        for i in range(0, len(list_all_files), step):
            list_file.append(list_all_files[i])
        if len(list_file) > num_frames:
            start_frame = len(list_file) - num_frames
            list_file = list_file[start_frame:]
        result.append(list_file)
    return result

def preprocess_urfd(num_frames):
    base_dir_pre = os.path.join(BASE_DIR_URFD[:-9], 'urfd_pre_data_64')
    list_dir = get_files_urfd(num_frames)
    for dir in tqdm(list_dir):
        dir_name = dir[0][:-8]
        path_origin = f'{BASE_DIR_URFD}/{dir_name}'
        path_pre = f'{base_dir_pre}/{dir_name}'
        utils.mkdir(path_pre)
        for file in dir:
            filename_origin = f'{path_origin}/{file}'
            shutil.copy(filename_origin, path_pre)

def preprocess_aihub(num_frame, phase='Training'):
    ''' Extract and copy the frames(num_frame) through the start frame info from Aihub dataset.

    :param num_frame: Number of frames to extract.
    :param phase: Training or Validation.
    '''
    # base_dir_origin = get_base_dir_aihub(file='image')
    # base_dir_origin = f'D:/balanced_aihub/{phase}'
    base_dir_origin = f'D:/openpose_image_aihub/{phase}'
    if phase == 'Training':
        annotation_file = utils.read_json('./aihub_data.json')
        # base_dir_pre = 'D:\Aihub_pre_data\Training\image'
        base_dir_pre = 'D:/aihub_pre_data_64/Training'
    else:
        annotation_file = utils.read_json('./aihub_val_data.json')
        # base_dir_pre = 'D:\Aihub_pre_data\Validation\image'
        base_dir_pre = 'D:/aihub_pre_data_64/Validation'
    annotations = annotation_file['annotations']

    list_dir = os.listdir(base_dir_origin)
    list_dir = [d for d in list_dir if d.startswith('[원천]')]
    for dir in tqdm(list_dir):
        path_pre = os.path.join(base_dir_pre, dir)
        utils.mkdir(path_pre)
        if 'H' in dir:
            # dir = os.path.join(dir, dir[-9:]) # [원천]In_M_I_003/M_I_003
            path_origin = os.path.join(base_dir_origin, dir)
            list_dir2 = os.listdir(path_origin)
            for dir2 in list_dir2:
                filename_origin = os.path.join(path_origin, dir2)
                filename_pre = os.path.join(path_pre, dir2)
                if len(os.listdir(filename_pre)) == num_frame:
                    continue
                list_all_frames = os.listdir(filename_origin)
                start_frame = annotations[dir2]['startframe']
                if int(list_all_frames[1][-6]) != 0: # x.0s.jpg or x.5s.jpg
                    start_frame = start_frame * 2
                end_frame = start_frame + num_frame
                if len(list_all_frames[start_frame:end_frame]) < num_frame:
                    start_frame -= num_frame - (len(list_all_frames) - start_frame)
                list_file = list_all_frames[start_frame:end_frame]
                # copy frames
                utils.mkdir(filename_pre)
                for file in list_file:
                    shutil.copy(os.path.join(filename_origin, file), filename_pre)
        else:
            # dir = os.path.join(dir, dir[7:]) # [원천]In_M_I_003/M_I_003
            path_origin = os.path.join(base_dir_origin, dir)
            list_dir2 = os.listdir(path_origin)
            for dir2 in list_dir2:
                filename_origin = os.path.join(path_origin, dir2)
                filename_pre = os.path.join(path_pre, dir2)
                if len(os.listdir(filename_pre)) == num_frame:
                    continue
                list_all_frames = os.listdir(filename_origin)
                start_frame = 0
                num_all_frames = len(list_all_frames)
                step = 4
                for i in range(0, num_all_frames, step):
                    if openpose.detected_human(os.path.join(filename_origin, list_all_frames[i])) or num_all_frames <= start_frame + num_frame + step:
                        break
                    start_frame += step
                end_frame = start_frame + num_frame
                if len(list_all_frames[start_frame:end_frame]) < num_frame:
                    start_frame -= num_frame - (len(list_all_frames) - start_frame)
                list_file = list_all_frames[start_frame:end_frame]
                # copy frames
                utils.mkdir(filename_pre)
                for file in list_file:
                    shutil.copy(os.path.join(filename_origin, file), filename_pre)

def write_input_urfd_i3d(mode, phase):
    '''
    param:
        mode: rgb or flow or openpose
        phase: train or val or test
    '''
    x_data = []
    y_data = []
    path = f'urfd_{mode}_i3d'
    vid_list = os.listdir(f'{path}/{phase}')
    vid_list = [vid for vid in vid_list if os.path.isdir(f'{path}/{phase}/{vid}')]
    for i, vid in enumerate(tqdm(vid_list)):
        y = []
        if vid.startswith('adl'):
            y.append(np.ones(64, dtype=np.float32))
            y.append(np.zeros(64, dtype=np.float32))
        else:
            y.append(np.zeros(64, dtype=np.float32))
            y.append(np.ones(64, dtype=np.float32))
        y_data.append(y)
        if mode == 'flow':
            frames = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in glob.glob(f'{path}/{phase}/{vid}/*.png')[:]]
        else:
            frames = [cv2.imread(file) for file in glob.glob(f'{path}/{phase}/{vid}/*.png')[:]]
        res = [cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA) for img in frames]
        x_data.append(res)
    print(np.asarray(x_data).shape)
    print(np.asarray(y_data).shape)
    utils.write_pickle(f'{path}/vid_{mode}_{phase}.pkl', x_data)
    utils.write_pickle(f'{path}/label_{mode}_{phase}.pkl', y_data)

def write_input_aihub(num_frames, phase='Training'):
    # base_dir = 'D:/openpose_black_aihub/{}'.format(phase)
    # base_dir = 'D:/openpose_image_aihub/{}'.format(phase)
    # base_dir = 'D:/aihub_pre_data_64/{}'.format(phase)
    # base_dir = 'D:/balanced_openpose_aihub/{}'.format(phase)
    # base_dir = 'D:/balanced_aihub/{}'.format(phase)
    # base_dir = 'D:/Openpose_aihub/{}'.format(phase)
    base_dir = 'D:/openpose_image_aihub/{}'.format(phase)
    list_dir = os.listdir(base_dir)
    x_data = []
    y_data = []
    for dir in tqdm(list_dir, leave=True):
        path = os.path.join(base_dir, dir)
        list_dir2 = os.listdir(path)
        for dir2 in tqdm(list_dir2, leave=True):
            path2 = os.path.join(path, dir2)
            list_file = os.listdir(path2)
            x = []
            if len(list_file) != num_frames:
                print('\n', path2, len(list_file))
                continue
            for file in list_file:
                path3 = os.path.join(path2, file)
                file = utils.read_image(path3)
                x.append(cv2.resize(file, dsize=(256, 256), interpolation=cv2.INTER_AREA))
            x_data.append(x)
            tmp = []
            if dir2.startswith('FD'):
                tmp.append(np.zeros(64, dtype=np.float32))
                tmp.append(np.ones(64, dtype=np.float32))
            else:
                tmp.append(np.ones(64, dtype=np.float32))
                tmp.append(np.zeros(64, dtype=np.float32))
            y_data.append(tmp)
    print(np.array(x_data).shape)
    print(np.array(y_data).shape)
    utils.write_pickle(f'x_{phase}_aihub_64.pkl', x_data)
    utils.write_pickle(f'y_{phase}_aihub_64.pkl', y_data)

def test():
    video = '2021-10-24.mp4'
    vidcap = cv2.VideoCapture()
    vidcap.open(video)

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_cnt = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_cnt / fps
    print('영상 길이:', duration, '초')

    cnt = 1706
    increase_width = 0.5
    second = 0
    status = True

    while status and second <= duration:
        status, frame = vidcap.read()
        vidcap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        print(second, '초 에서 캡쳐')
        cv2.imwrite('2021-10-24_{}.jpg'.format(cnt), frame)
        cnt += 1
        second += increase_width
        if cv2.waitKey(10) == 27:
            break

def load_flow(dataset, phase):
    '''
    param:
        dataset: urfd or aihub
        phase: train or val or test
    '''
    flow_vid_list = []
    if dataset == 'urfd':
        path = f'gdrive/MyDrive/sw_capstone/{dataset}_flow_64'
    else:
        path =
    vid_list = utils.read_pickle(f'{path}/vid_flow_{phase}_1.pkl')
    for vid in tqdm(vid_list):
        flow_vid = []
        for img in vid:
            imgx = img
            imgy = img
            img = np.asarray([imgx, imgx]).transpose([1, 2, 0])
            flow_vid.append(img)
        flow_vid_list.append(flow_vid)
    utils.write_pickle(f'{path}/vid_flow_{phase}.pkl', flow_vid_list)
    print(np.array(flow_vid_list).shape)

def write_flow_aihub(phase='Training'):
    x_data = []
    y_data = []
    origin_base_dir = f'D:/aihub_pre_data_64/{phase}'
    # flow_base_dir = f'D:/aihub_flow/{phase}'
    list_dir = os.listdir(origin_base_dir)
    for dir in tqdm(list_dir, leave=True):
        path = os.path.join(origin_base_dir, dir)
        list_dir2 = os.listdir(path)
        # os.mkdir(f'{flow_base_dir}/{dir}')
        for dir2 in tqdm(list_dir2, leave=True):
            path2 = os.path.join(path, dir2)
            img_list = os.listdir(path2)
            if len(img_list) != 64:
                print('\n', path2, len(img_list))
                continue
            x = []
            imgs = [cv2.imread(file) for file in glob.glob(f'{origin_base_dir}/{dir}/{dir2}/*.jpg')]
            frame1 = imgs[0]
            prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            hsv = np.zeros_like(frame1)
            hsv[..., 1] = 255
            # flow_path = f'{flow_base_dir}/{dir}/{dir2}'
            # os.mkdir(flow_path)
            for i, frame2 in enumerate(tqdm(imgs)):
                next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                # cv2.imwrite(f'{flow_path}/{i}.png', bgr)
                x.append(cv2.resize(bgr, dsize=(256, 256), interpolation=cv2.INTER_AREA))
            x_data.append(x)
            print(np.array(x_data).shape)
            tmp = []
            if dir2.startswith('FD'):
                tmp.append(np.zeros(64, dtype=np.float32))
                tmp.append(np.ones(64, dtype=np.float32))
            else:
                tmp.append(np.ones(64, dtype=np.float32))
                tmp.append(np.zeros(64, dtype=np.float32))
            y_data.append(tmp)
    print(np.array(x_data).shape)
    print(np.array(y_data).shape)
    utils.write_pickle(f'vid_flow_{phase}.pkl', x_data)
    utils.write_pickle(f'label_flow_{phase}.pkl', y_data)

if __name__ == '__main__':
    preprocess_aihub(16)
    preprocess_aihub(16, phase='Validation')
    write_input_aihub()
    write_input_aihub('Validation')
    test()
    EDA()
    preprocess_urfd(64)
    preprocess_aihub(64)
    preprocess_aihub(64, phase='Validation')
    write_input_aihub(64)
    write_input_aihub(64, 'Validation')
    write_input_aihub(64, 'test')
    with open('x_Training_aihub_64.pkl', 'rb') as f:
        x_data = pickle.load(f)
    with open('y_Training_aihub_64.pkl', 'rb') as f:
        y_data = pickle.load(f)
    print(np.array(x_data).shape)
    print(np.array(y_data).shape)
    load_flow()
    write_input_urfd_i3d('openpose', 'train')
    write_input_urfd_i3d('openpose', 'val')
    write_input_urfd_i3d('openpose', 'test')
    write_input_aihub(64, 'Training')
    write_input_aihub(64, 'Validation')
    write_flow_aihub('Training')
    write_flow_aihub('Validation')