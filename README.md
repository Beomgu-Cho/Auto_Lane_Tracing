# Auto_Lane_Tracing
###### 본 프로젝트는 2018년 7월 ~ 2019년 1월 까지 진행된 것으로 교육용 자료로 만들었습니다.
###### 해당 자료는 한국산업기술대학교에서 2019년 1월부터 약 한 달간 특강으로 교육이 시행되었습니다.

### 실행 환경
#### Windows 10
#### Python 3.7.3 ver
#### OpenCV 4.1.2 ver
#### 기타 라이브러리 matplotlib, numpy, time 등
#### PyCharm 가상환경
----------
# Lane_Tracing
### 자율주행의 핵심 기술이 되는 차선인식 및 추적기술 입니다.
#### 실제 자율주행자동차의 경우 Lidar 센서, Radar 센서 등을 다양한 방법을 이용하여 정확하고 많은 정보를 수용하는 방법을 이용하지만 
#### 이 프로젝트의 경우 기능 구현에 초점을 맞추었기 때문에 카메라를 통해 간단하게 기능을 구현하였습니다.

## opencv 라이브러리 설치
#### pip 명령어를 이용하여 설치가 가능합니다.
### 단, Linux OS 라면 다른 방법을 이용해야 합니다.
```
  [terminal]> pip install opencv-python
  [terminal]> pip install opencv-contrib-python
```
## 기타 라이브러리 설치
```
  [terminal]> pip install numpy
  [terminal]> pip install matplotlib
```
## 라이브러리 참조
```
  import cv2
  import numpy as np
  import time
```
## 1. 카메라 접근
### 1-1. Access
#### OpenCV 는 내/외부 접속되어있는 카메라에 연결할 수 있는 함수가 제공이 됩니다.
```
  cv2.VideoCapture()
```
#### 괄호 안에 0 혹은 -1 을 입력하면 하드웨어 내장 카메라를 출력해주고
#### 각 포트에 연결된 외장 카메라는 1, 2, ...
#### 동영상파일의 경로를 작은따옴표 안에 입력하면 해당 동영상 파일을 사용할 수 있습니다.
#### 해당 영상(카메라)의 정보를 return 합니다.
```
  예시
  cap = cv2.VideoCapture('project_video.mp4')
```
#### .py 파일과 같은 경로에 있는 project_video.mp4 파일을 access 하였습니다.

### 1-2. Frame Capture
#### `cap`으로 저장된 변수를 출력하기 위해선 `cap.read()` 로 이용할 수 있습니다.
```
  ret, frame = cap.read()
```
#### 위에서 접근한 영상은 기본적으로 영상의 각 프레임을 출력합니다.
#### ret 엔 cap 접속 정보, Frame엔 cap의 영상 각각의 프레임이 각각의 이미지로 출력됩니다.
##### 즉, 완전한 영상으로 출력하기 위해선 루프문을 이용하여야 합니다.
##### 이 경우엔 무한루프로 만들어 사용할 것입니다.
```
  cap = cv2.VideoCapture('project_video.mp4')
  
  while(True):
    ret, frame = cap.read()
    
    cv2.imshow('result', frame)
```

#### 정상적으로 영상이 잘 출력됩니다.
### 1-3. Frame Delay Check
#### time 라이브러리를 이용하여 쉽게 체크가 가능합니다.
#### time.time() 기준 시간 대비 현재 시간을 초 단위로 출력합니다.
```
  last_time = time.time()
```
#### 그리고 while 루프의 마지막에 time.time() - last_time 을 출력하면 1 루프에 걸린 시간을 알 수 있습니다.
#### 즉, frame 체크가 가능합니다.
```
  cap = cv2.VideoCapture('project_video.mp4')
  last_time = time.time()
  
  while(True):
    ret, frame = cap.read()
    
    cv2.imshow('result', frame)
    printf("frame took: {}".format(time.time() - last_time))
    last_time = time.time()
```
###### `(time.time() - last_time)/60` 으로 하면 fps 체크가 가능합니다.
### 1-4. Region Of Interest
#### 관심 영역을 지정합니다.
#### 대부분의 경우 관심영역 외의 부분을 빈공간으로 처리하는 경우가 대부분이었는데 이 경우 라인의 각도가 조금밖에 이루어지지 않는 것같아 최대한 실제 라인과 비슷한 환경을 만들어 주기 위해 위에서 바라보는 화면을 만들어 주었습니다.
##### 해당 관심영역을 이하 upper_view 라 칭하겠습니다.
### 1-4-1. 영역 지정 원리
###### 자율주행을 위해 차선을 본다고 하면 차량 앞으로 나와있는 차선과 옆 차선의 일부분이 같이 보이게 설정하였습니다.
###### 옆 차선이 모두 보일 정도가 되면 너무 많은 직선이 검출되어 차선 인식에 장애가 있을 수 있다고 판단하여 정면 차선과 양 옆의 차선 일부를 해당 영역으로 지정하였습니다.
#### 차선이 보이는 이미지에서 `matplotlib.pyplot` 라이브러리를 이용하여 원하는 위치의 픽셀좌표를 가져옵니다.
##### 저의 경우 차례대로 [300, 650], [580, 460], [720, 460], [1100, 650] 가 되었습니다.
## 2. 차선 검출
### 2-1. 이미지 전처리
#### opencv에서 불러온 이미지는 기본적으로 ndarray 배열의 형태이며 각 셀마다 `(Blue, Green, Red)` 의 3차원 데이터로 이루어져 있습니다.
#### 해당 데이터를 처리하기 쉽게 1차원 `GrayScale Image`로 변경합니다.
```
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```
#### `image`를 BGR을 GRAY Scale로 변환합니다.
#### 조금 더 처리하기 쉽게 1차원 데이터를 이진화 하였습니다.
```
  _, binary = cv2.threshold(gray, 197, 255, cv2.THRESH_BINARY)
  binary_gaussian = cv2.adaptiveThreshold(binary, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
```
#### 가우시안 이진화기법을 이용하는 함수입니다.
#### `cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)`
###### image를 이진화 할 때 가우시안 기법을 사용하고 0 혹은 255 로 변환합니다.
#### 생성된 이미지의 노이즈를 제거하기 위해 Blur 처리를 하였습니다.
#### 사용한 Blur 기법은 가우시안블러링 입니다.
```
  blur = cv2.GaussianBlur(binary_gaussian, (5, 5), 0)
```
##### 이상으로 이미지 전처리 단계는 끝입니다.
### 2-2. Canny
#### 차선을 검출하는 가장 대표적인 방법입니다.
#### Canny 기법은 배열의 주변 셀과 값의 차이가 클 때 그 부분들끼리 이어진다면 선으로 표시해주는 방법 입니다.
```
  Canny_image = cv2.Canny(blur, 80, 120)
```
#### 여기까지의 과정을 하나의 함수로 만들어 사용했습니다.
#### 현재까지의 내용을 담은 전체 코드입니다.
```
  # 라이브러리 참조
  import cv2
  import numpy as np
  import time
  
  
  # 이미지 전처리 과정
  def make_canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 197, 255, cv2.THRESH_BINARY)
    binary_gaussian = cv2.adaptiveThreshold(binary, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
    blur = cv2.GaussianBlur(binary_gaussian, (5, 5), 0)
  
    canny_image = cv2.Canny(blur, 80, 120)

    return canny_image
  
  
  # 동영상 파일 접근
  cap = cv2.VideoCapture('project_video.mp4')
  last_time = time.time()
  
  # 메인 루프
  while(True):
    ret, frame = cap.read()
    canny = make_canny(frame)
    
    cv2.imshow('result', canny)
    printf("frame took: {}".format(time.time() - last_time))
    last_time = time.time()
```
### 2-3. HoughLines 기법
```
  cv2.HoughLinesP(image, 3, np.pi/180, 100, np.array([]), minimum_linelength, maxium_line_gap)
```
#### 해당 이미지에서 직선을 검출합니다. 최소길이와 다른 직선들과의 거리를 설정할 수 있습니다.
#### canny화 한 이미지에서 직선을 찾아줍니다.
```
  lines = cv2.HoughLinesP(canny, 3, np.pi/180, 100, np.array([]), 100, 400)
```
