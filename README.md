# 1. Introduction
JoonMyung Choi's Package


# 2. ToDo List
### a. Library 
1. joonmyung/Script
 - [ ] 추가 스크립트, Queue 추가 

2. joonmyung/draw 
 - [ ] LinePlot 수정
 
### b. Playground


# 3. Previous
## Version 1.4.1
1. joonmyung/log
 - [X] wandb table 저장 오류 수정
2. joonmyung/utils
 - [X] str2list 띄어쓰기 오류 수정

## Version 1.4.0
1. joonmyung/app.py
 - [X] 실험 도중 스크립트 추가 기능
2. joonmyung/log
 - [X] 모델, 코드 저장 기능

## Version 1.3.2
1. joonmyung/Logger
 - [X] wandb_id 작업

## Version 1.3.2
1. joonmyung/Script
 - [X] Multi-GPU 적용
2joonmyung/draw
 - [X] rollout (Attention, Gradient) 추가   
3. joonmyung/log
 - [X] type에 대한 확인
4. playground/profiling
 - [X] 속도 측정 비교 메서드 구체화


## Version 1.3.1
### a. Library
1. joonmyung/draw 
 - [X] overlay 기능 추가

### b. Playground
1. playground/analysis
   - [X] data  관련 분석 코드 작성
   - [X] Model 관련 분석 코드 작성
   
## Version 1.3.0
### a. Library 
1. joonmyung/draw
 - [X] drawImgPlot 추가   

3. joonmyung/log
- [X] Wandb Log / Table 추가








[//]: # (CUDA_VISIBLE_DEVICES=2 python playground/models/fastsam/model.py --split 0)
# CUDA_VISIBLE_DEVICES=2 nohup python playground/models/fastsam/model.py --split 0 > 0.log 2>&1  &
# CUDA_VISIBLE_DEVICES=2 nohup python playground/models/fastsam/model.py --split 1 > 1.log 2>&1  &
# CUDA_VISIBLE_DEVICES=2 nohup python playground/models/fastsam/model.py --split 2 > 2.log 2>&1  &
# CUDA_VISIBLE_DEVICES=2 nohup python playground/models/fastsam/model.py --split 3 > 3.log 2>&1  &
# CUDA_VISIBLE_DEVICES=3 nohup python playground/models/fastsam/model.py --split 4 > 4.log 2>&1  &
# CUDA_VISIBLE_DEVICES=3 nohup python playground/models/fastsam/model.py --split 5 > 5.log 2>&1  &
# CUDA_VISIBLE_DEVICES=3 nohup python playground/models/fastsam/model.py --split 6 > 6.log 2>&1  &
# CUDA_VISIBLE_DEVICES=3 nohup python playground/models/fastsam/model.py --split 7 > 7.log 2>&1  &


# nohup python playground/saliency/opencv.py --split 0 > 0.log 2>&1  &
# nohup python playground/saliency/opencv.py --split 1 > 1.log 2>&1  &
# nohup python playground/saliency/opencv.py --split 2 > 2.log 2>&1  &
# nohup python playground/saliency/opencv.py --split 3 > 3.log 2>&1  &
# nohup python playground/saliency/opencv.py --split 4 > 4.log 2>&1  &
# nohup python playground/saliency/opencv.py --split 5 > 5.log 2>&1  &
# nohup python playground/saliency/opencv.py --split 6 > 6.log 2>&1  &
# nohup python playground/saliency/opencv.py --split 7 > 7.log 2>&1  &