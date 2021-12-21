# Fall_Detection

[소프트웨어융합캡스톤디자인]

## 1. 과제 선정 배경 및 필요성

- 사고 발생 예측, 예방을 위한 데이터기반 스마트 돌봄 서비스 수요 증가
- 2025년 초고령 사회 진입할 것으로 예측
- 노약자의 경우 뇌졸중 등으로 인한 낙상 사고가 빈번하고 낙상으로 인한 2차 피해 발생 가능성이 높음

## 2. 데이터셋

- URFD Dataset
- AI Hub 시니어 이상행동 영상

## 3. Training

### C3D

- URFD(RGB)

<img src="/img/urfd/c3d_rgb_epoch100.png" width="400" height="250">

- URFD(Openpose)

<img src="/img/urfd/c3d_pose_epoch100.png" width="400" height="250">

- AI Hub(RGB)

<img src="/img/aihub/c3d_rgb_epoch10.png" width="400" height="250">

- AI Hub(Openpose)

<img src="/img/aihub/c3d_pose_epoch10.png" width="400" height="250">

### I3D

- URFD(RGB)

<img src="/img/urfd/i3d_imagenet_rgb_lr0.01_epoch100.png" width="400" height="250">

- URFD(Optical Flow)

<img src="/img/urfd/i3d_imagenet_flow_lr0.01_epoch100.png" width="400" height="250">

- URFD(Openpose)

<img src="/img/urfd/i3d_imagenet_pose_lr0.01_epoch100.png" width="400" height="250">

- AI Hub(RGB)

<img src="/img/aihub/i3d_imagenet_rgb_lr0.0025_epoch4.png" width="400" height="250">

- AI Hub(Openpose)

<img src="/img/aihub/i3d_imagenet_pose_lr0.0025_epoch4.png" width="400" height="250">


## Link

[발표자료](https://drive.google.com/file/d/1FVipTl91Kh3r8U1L-D2zCsWObi-QkAS6/view?usp=sharing)
