# Fall_Detection

[소프트웨어융합캡스톤디자인]

## 1. Overview

- 2025년 초고령 사회 진입할 것으로 예측
- 사고 발생 예측, 예방을 위한 데이터기반 스마트 돌봄 서비스 수요 증가
- 노약자의 경우 뇌졸중 등으로 인한 낙상 사고가 빈번하고 낙상으로 인한 2차 피해 발생 가능성이 높음
- 따라서 Openpose와 Action recognition 모델을 이용하여 낙상 감지 성능을 비교함

## 2. Dataset

- URFD Dataset

  ADL: 40 clip, Fall: 30 clip
  
  <img src="/img/urfd.png" width="400" height="250">
  
- AI Hub 시니어 이상행동 영상

  ADL: 700 clip, Fall: 700 clip

  <img src="/img/aihub.png" width="400" height="250">

## 3. Model

### 1) C3D

<img src="/img/c3d.png">

### 2) I3D

<img src="/img/i3d.png">

- 위 2개의 모델에 RGB 이미지와 Openpose 라이브러리를 이용하여 Pose line을 그린 이미지를 넣음(I3D는 Optical Flow도)

## 4. Training

### 1) C3D

- URFD(RGB)

  Best accuracy: 1

  <img src="/img/urfd/c3d_rgb_epoch100.png" width="500" height="300">

- URFD(Openpose)

  Best accuracy: 1

  <img src="/img/urfd/c3d_pose_epoch100.png" width="500" height="300">

- AI Hub(RGB)

  Best accuracy: 0.97

  <img src="/img/aihub/c3d_rgb_epoch10.png" width="500" height="300">

- AI Hub(Openpose)

  Best accuracy: 0.97

  <img src="/img/aihub/c3d_pose_epoch10.png" width="500" height="300">

### 2) I3D

- URFD(RGB)

  Best accuracy: 1

  <img src="/img/urfd/i3d_imagenet_rgb_lr0.01_epoch100.png" width="500" height="300">

- URFD(Optical Flow)

  Best accuracy: 1

  <img src="/img/urfd/i3d_imagenet_flow_lr0.01_epoch100.png" width="500" height="300">

- URFD(Openpose)

  Best accuracy: 1

  <img src="/img/urfd/i3d_imagenet_pose_lr0.01_epoch100.png" width="500" height="300">

- AI Hub(RGB)

  Best accuracy: 0.92

  <img src="/img/aihub/i3d_imagenet_rgb_lr0.0025_epoch4.png" width="500" height="300">

- AI Hub(Openpose)

  Best accuracy: 0.97

  <img src="/img/aihub/i3d_imagenet_pose_lr0.0025_epoch4.png" width="500" height="300">

## 4. Conclusion

- Openpose를 이용하여 이미지에 pose line을 그린 경우 최대 5% 성능 향상
- AI Hub 데이터의 경우 배경이 다양하고 차지하는 비율이 높아 비교적 성능이 떨어짐
- 낙상으로 인한 노약자 사고 및 사망 예방 효과를 기대할 수 있음
- 카메라를 이용한 노약자의 스마트 돌봄 서비스에 이용 가능

## Link

[발표자료](https://drive.google.com/file/d/1FVipTl91Kh3r8U1L-D2zCsWObi-QkAS6/view?usp=sharing)
