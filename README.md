# Auto_Lane_Tracing
###### 본 프로젝트는 2018년 7월 ~ 2019년 1월 까지 진행된 것으로 교육용 자료로 만들었습니다.
###### 해당 자료는 한국산업기술대학교에서 2019년 1월부터 약 한 달간 특강으로 교육이 시행되었습니다.

### 실행 환경
#### Windows 10
#### Python 3.7.3 ver
#### OpenCV 4.1.2 ver
#### 기타 라이브러리 matplotlib, numpy, time 등
#### PyCharm 가상환경


<div>
  <img width="300" src="https://github.com/Beomgu-Cho/Auto_Lane_Tracing/blob/main/capture/python.png">
  <img width="300" src="https://github.com/Beomgu-Cho/Auto_Lane_Tracing/blob/main/capture/opencv.png">
</div>


----------
# Lane_Tracing
### 자율주행의 핵심 기술이 되는 차선인식 및 추적기술 입니다.
#### 실제 자율주행자동차의 경우 Lidar 센서, Radar 센서 등을 다양한 방법을 이용하여 정확하고 많은 정보를 수용하는 방법을 이용하지만 
#### 이 프로젝트의 경우 기능 구현에 초점을 맞추었기 때문에 카메라를 통해 간단하게 기능을 구현하였습니다.
![screenshots autu_drive.png](https://github.com/Beomgu-Cho/Auto_Lane_Tracing/blob/main/capture/auto_drive.jpg)
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
<img width="400" src="https://github.com/Beomgu-Cho/Auto_Lane_Tracing/blob/main/capture/1.video_capture.png">


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
    print("frame took: {}".format(time.time() - last_time))
    last_time = time.time()
```
###### `(time.time() - last_time)/60` 으로 하면 fps 체크가 가능합니다.
![screenshots 2.time.time().png](https://github.com/Beomgu-Cho/Auto_Lane_Tracing/blob/main/capture/2.time.time().PNG)

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
    print("frame took: {}".format(time.time() - last_time))
    last_time = time.time()
```
<div>
  <img width="400" src="https://github.com/Beomgu-Cho/Auto_Lane_Tracing/blob/main/capture/3.gray.png">
  <img width="400" src="https://github.com/Beomgu-Cho/Auto_Lane_Tracing/blob/main/capture/4.binary.png">
</div>
<div>
  <img width="400" src="https://github.com/Beomgu-Cho/Auto_Lane_Tracing/blob/main/capture/5.blur.png">
  <img width="400" src="https://github.com/Beomgu-Cho/Auto_Lane_Tracing/blob/main/capture/6.canny.png">
</div>


### 2-3. Region Of Interest
#### 관심 영역을 지정합니다.
#### 대부분의 경우 관심영역 외의 부분을 빈공간으로 처리하는 경우가 대부분이었는데 이 경우 라인의 각도가 조금밖에 이루어지지 않는 것같아 최대한 실제 라인과 비슷한 환경을 만들어 주기 위해 위에서 바라보는 화면을 만들어 주었습니다.
##### 해당 관심영역을 이하 upper_view 라 칭하겠습니다.
### 영역 지정 원리
###### 자율주행을 위해 차선을 본다고 하면 차량 앞으로 나와있는 차선과 옆 차선의 일부분이 같이 보이게 설정하였습니다.
###### 옆 차선이 모두 보일 정도가 되면 너무 많은 직선이 검출되어 차선 인식에 장애가 있을 수 있다고 판단하여 정면 차선과 양 옆의 차선 일부를 해당 영역으로 지정하였습니다.

#### 차선이 보이는 이미지에서 `matplotlib.pyplot` 라이브러리를 이용하여 원하는 위치의 픽셀좌표를 가져옵니다.
```
  import cv2
  import matplotlib.pyplot as plt


  def canny(image):
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      blur = cv2.GaussianBlur(gray, (5, 5), 0)
      canny = cv2.Canny(blur, 50, 150)

      return canny


  cap = cv2.VideoCapture('project_video.mp4')

  while(True):
      _, frame = cap.read()
      canny = canny(frame)

      plt.imshow(canny)
      plt.show()

```
#### OpenCV가 아닌 Matplotlib.pyplot을 이용하여 출력하면 화면에 한 프레임의 이미지가 올라오게 되는데 이 때 마우스를 이미지에 가져다대면 마우스 끝이 있는 해당 셀의 (x, y) 좌표를 확인할 수 있습니다.
###### 저의 경우 차례대로 `[300, 650], [580, 460], [720, 460], [1100, 650]` 가 나왔습니다.
![screenshots 7.matplot.png](https://github.com/Beomgu-Cho/Auto_Lane_Tracing/blob/main/capture/7.matplot.PNG)
### 새 윈도우창 설정
#### 이제 upper_view 를 출력할 윈도우의 설정값을 지정해 줍니다.
```
  h = 640
  w = 800
  pts1 = np.float32([[300, 650], [580, 460], [720, 460], [1100, 650]])
  pts2 = np.float32([[200, 640], [200, 0], [600, 0], [600, 640]])
  M = cv2.getPerspectiveTransform(pts1, pts2)
```

||변수명|데이터|
|---|---|---|
|높이|h|640|
|너비|w|800|
|원래 이미지 좌표|pts1|[300, 650], [580, 460], [720, 460], [1100, 650]|
|새 윈도우로 치환할 좌표|pts2|[200, 640], [200, 0], [600, 0], [600, 640]|

#### 여기서 `cv2.getPerspectiveTransform(pts1, pts2)` 는 pts1의 배열 속 좌표를 새로운 이미지의 pts2 배열로 옮기게 하는 정보를 변수 `M` 에 담아줍니다.
#### 이때 늘어나서 비어있게 되는 공간은 양 옆 두 셀 값의 평균값을로 채워집니다.
#### upper_view 이미지를 새롭게 만듭니다.
#### frame의 각 프레임에 적용시켜야하므로 루프안에 적용시킵니다.
```
  img2 = cv2.warpPerspective(frame, M, (w, h), borderValue=(255, 255, 255))
```
<img width="400" src="https://github.com/Beomgu-Cho/Auto_Lane_Tracing/blob/main/capture/8.img2.png">


#### 이제 upper_view 화면이 정상적으로 출력이 가능합니다.
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
  
  # upper_view 설정
  h = 640
  w = 800
  pts1 = np.float32([[300, 650], [580, 460], [720, 460], [1100, 650]])
  pts2 = np.float32([[200, 640], [200, 0], [600, 0], [600, 640]])
  M = cv2.getPerspectiveTransform(pts1, pts2)
  
  # 메인 루프
  while(True):
    ret, frame = cap.read()
    img2 = cv2.warpPerspective(frame, M, (w, h), borderValue=(255, 255, 255))    # frame --> upper_view
    canny = make_canny(img2)
    
    cv2.imshow('result', canny)
    print("frame took: {}".format(time.time() - last_time))
    last_time = time.time()
```
### 2-4. HoughLines 기법
```
  cv2.HoughLinesP(image, 3, np.pi/180, 100, np.array([]), minimum_linelength, maxium_line_gap)
```
#### 해당 이미지에서 직선을 검출합니다. 최소길이와 다른 직선들과의 거리를 설정할 수 있습니다.
#### canny화 한 이미지에서 직선을 찾아줍니다.
```
  lines = cv2.HoughLinesP(canny, 3, np.pi/180, 100, np.array([]), 100, 400)
```
#### lines 변수에는 canny 이미지에서 찾아낸 직선들의 `(x1, y1), (x2, y2)` 정보가 담겨있습니다.
> x1, y1 : 직선의 시작점 좌표
> x2, y2 : 직선의 끝점 좌표
```
  # 메인 루프
  while(True):
    ret, frame = cap.read()
    img2 = cv2.warpPerspective(frame, M, (w, h), borderValue=(255, 255, 255))    # frame --> upper_view
    canny = make_canny(img2)
    lines = cv2.HoughLinesP(canny, 3, np.pi/180, 100, np.array([]), 100, 400)
    
    cv2.imshow('result', canny)
    print("frame took: {}".format(time.time() - last_time))
    last_time = time.time()
```
###### lines 변수는 직선들의 좌표 정보만을 가지고 있기 때문에 이것만으로 출력이 불가능 합니다.
### 2-5. 직선 표시
#### 우선 차선을 인식할 때 왼쪽 차선과 오른쪽 차선을 구분해야 합니다. 구분을 위해 빈 배열을 생성합니다.
```
  left_fit = []
  right_fit = []
```
#### 실제 노면에서 검출되는 직선들은 차선뿐만 아니라 노면의 울퉁불퉁한 정도에 따라 이상한 직선들이 검출되기도 하고 옆차선을 달리는 자동차에서도 검출될 수 있습니다.
#### 따라서, 각도가 어느정도 일치하는 직선들만 검출할 필요가 있습니다.
#### 각도에 관한 정보는 따로 없으니 좌표를 계산하여 각도를 도출했습니다.
```
  for line in lines:
     x1, y1, x2, y2 = line.reshape(4)

     x = np.array([x1, x2])
     y = np.array([y1, y2])
     A = np.vstack([x, np.ones(len(x))]).T

     slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
```
#### lines의 각 요소마다 계산을 실시합니다.
#### (x1, y1), (x2, y2) 좌표를 지나는 직선의 방정식을 구하고 x절편(기울기), y절편을 각각 slope와 intercept 변수에 저장합니다.
#### y = 0 일 때의 x 좌표 즉, (x`, 0)의 값(x_coord)으로 직선의 시작점을 기준으로 좌측 직선과 우측 직선을 구분합니다.
#### 구분한 직선을 미리 만들어놓은 알맞은 배열에 넣습니다.

```
     x_coord = -((intercept-640) / slope)

     if x_coord < 400:
         left_fit.append((slope, intercept))

     elif x_coord > 400:
         right_fit.append((slope, intercept))
```
#### 400 전, 후로 비교한 이유는 표시될 윈도우의 너비가 800 이기 때문에 중간지점인 400을 기준으로 왼쪽 오른쪽으로 나누었습니다.
#### 입력된 배열의 모든 요소의 평균을 저장합니다.
```
    left_fit_average = np.mean(left_fit, 0)
    right_fit_average = np.mean(right_fit, 0)
```
#### 지금 계산한 직선의 평균들은 `slope`와 'intercept'의 값을 가지는 데이터입니다.
#### 이제 이 데이터를 이용하여 직선을 만들기 위해 다시 x1, y1, x2, y2 값을 추출합니다.
```
    slope, intercept = line_parameters

    y1 = img.shape[0]
    y2 = int(y1 * (1/2))

    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
```
#### line_parameters 는 각각 left_fit_average, right_fit_average가 들어가고 img는 출력할 이미지에 해당합니다. img.shape[0] 대신 640을 넣어도 됩니다.
###### img.shape[0]는 해당 이미지의 높이값입니다.
#### 여기서 y1, y2의 값은 화면상에 표시하기 위해 고정된 값으로 두었고 x1, x2는 y1, y2에 해당하는 위치의 x좌표를 구한 결과값입니다.
#### 코드 생성물
```
  # 좌측, 우측 선분의 평균값을 가진 데이터를 이용해 좌표 추출
  def make_coordinates(img, line_parameters):
      slope, intercept = line_parameters

      y1 = img.shape[0]
      y2 = int(y1 * (1/2))

      x1 = int((y1 - intercept)/slope)
      x2 = int((y2 - intercept)/slope)

      return np.array([x1, y1, x2, y2])
  # 추출한 선분들을 좌측, 우측으로 나누고 좌표 추출
  def average_slope_intercept(img, lines):

      left_fit = []
      right_fit = []

      for line in lines:
          x1, y1, x2, y2 = line.reshape(4)

          x = np.array([x1, x2])
          y = np.array([y1, y2])
          A = np.vstack([x, np.ones(len(x))]).T

          slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

          x_coord = -((intercept-640) / slope)

          if x_coord < 400:
              left_fit.append((slope, intercept))

          elif x_coord > 400:
              right_fit.append((slope, intercept))
              
      # 평균 선분 한개 도출
      left_fit_average = np.mean(left_fit, 0)
      right_fit_average = np.mean(right_fit, 0)
      
      # 좌표 추출
      left_line = make_coordinates(img, left_fit_average)
      right_line = make_coordinates(img, right_fit_average)

      return [left_line], [right_line]
```
#### 이 함수에 img2와 lines를 대입하여 return값을 받습니다.
```
  l1, l2 = average_slope_intercept(img2, lines)
```
### 2-5. line 그리기
#### OpenCV 에는 화면에 다양한 도형 및 직선을 그릴 수 있는 기본 함수를 제공합니다.
#### `cv2.line(image, (x1, y1), (x2, y2), (B, G, R), line_thickness)` 함수로 쉽게 그릴 수 있습니다.
#### 다만 라인을 그릴 화면은 원본 이미지 위에 바로 덧칠하는게 아니라 비어있는 이미지 위에 그린 뒤 원본 위에 덮어씌울 것입니다.
#### 반투명하게 위에 덮어 원본의 형태를 남기기 위한 작업 입니다.
```
  def display_lines(img, lines):
      line_image = np.zeros_like(img)

      if lines is not None:
          for line in lines:
              x1, y1, x2, y2 = line.reshape(4)
              cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 20)

      return line_image
```
#### 여기에 사용될 lines는 l1, l2 각각 한번씩 적용하고 각각 따로 저장합니다.
```
  left_line_image = display_lines(img2, np.array(l1))
  right_line_image = display_lines(img2, np.array(l2))
```
#### 이제 `cv2.addWeighted()`함수를 이용하여 해당 라인 이미지와 원본 이미지를 합칩니다.
```
  line_image = cv2.addWeighted(left_line_image, 1, right_line_image, 1, 1)

  combo_image = cv2.addWeighted(img2, 1, line_image, 0.6, 1)
```
<div>
  <img width="400" src="https://github.com/Beomgu-Cho/Auto_Lane_Tracing/blob/main/capture/9.line_image.png">
  <img width="400" src="https://github.com/Beomgu-Cho/Auto_Lane_Tracing/blob/main/capture/10.combo_image.png">
</div>


#### line_image 는 각각 좌, 우측 라인이미지를 합한 것,
#### combo_image 는 line_image 와 img2 의 이미지를 합친 것입니다.
#### combo_image 에서 0.6 에 해당하는 부분이 line_image를 반투명하게 설정한 것입니다. 1 일 때 완전 불투명입니다.
### 2-6. 결과 확인하기
#### 눈으로 보이는 결과를 이용하여 차선의 방향, 차량이 가야하는 방향을 출력해보았습니다.
```
  x2_coord_average = (l2[0][2] + l1[0][2]) / 2

  turning_rate = x2_coord_average - 400

  if turning_rate < -10:
      print("left")
  elif turning_rate > 10:
      print("right")
  else:
      print("straight")
```
![screenshots 11.check_lane.png](https://github.com/Beomgu-Cho/Auto_Lane_Tracing/blob/main/capture/11.check_lane.PNG)
#### 좌, 우측의 x2 좌표값의 평균을 기준으로 중앙에서 벗어난 정도를 체크했습니다.
#### 결과 정상적으로 출력이 됩니다.
## 3. 결과
#### 모든 작업이 끝났다면 종료를 시행해야 합니다.
#### 종료되지 않는 무한 루프를 이용했기 때문에 키보드 입력값을 이용해 해당 루프를 종료하는 문장을 만들어줍니다.
```
  if cv2.waitKey(1) & 0xFF == 27:
    break
```
#### 입력된 KeyChar 값이 27 (esc) 이면 루프를 벗어나게 설정했습니다.
#### 루프를 탈출 했을 때 비디오 혹은 카메라의 접속을 해제하고 켜져있는 모든 윈도우를 종료하게 합니다.
```
  cv2.destroyAllWindows()
  cap.release()
```
# 최종 코드
```
import cv2
import numpy as np
import time


def make_coordinates(img, line_parameters):
    slope, intercept = line_parameters

    y1 = img.shape[0]
    y2 = int(y1 * (1/2))

    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)

    return np.array([x1, y1, x2, y2])


def average_slope_intercept(img, lines):

    left_fit = []
    right_fit = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)

        x = np.array([x1, x2])
        y = np.array([y1, y2])
        A = np.vstack([x, np.ones(len(x))]).T

        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

        x_coord = -((intercept-640) / slope)

        if x_coord < 400:
            left_fit.append((slope, intercept))

        elif x_coord > 400:
            right_fit.append((slope, intercept))

    left_fit_average = np.mean(left_fit, 0)
    right_fit_average = np.mean(right_fit, 0)
    
    left_line = make_coordinates(img, left_fit_average)
    right_line = make_coordinates(img, right_fit_average)

    return [left_line], [right_line]


def display_lines(img, lines):
    line_image = np.zeros_like(img)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 20)

    return line_image


def make_canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 197, 255, cv2.THRESH_BINARY)
    binary_gaussian = cv2.adaptiveThreshold(binary, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
    blur = cv2.GaussianBlur(binary_gaussian, (5, 5), 0)
    blur = cv2.bilateralFilter(blur, 9, 75, 75)

    canny_image = cv2.Canny(blur, 80, 120)

    return canny_image


cap = cv2.VideoCapture('project_video.mp4')
last_time = time.time()

h = 640
w = 800
pts1 = np.float32([[300, 650], [580, 460], [720, 460], [1100, 650]])
pts2 = np.float32([[200, 640], [200, 0], [600, 0], [600, 640]])
M = cv2.getPerspectiveTransform(pts1, pts2)

while(True):
    ret, frame = cap.read()
    img2 = cv2.warpPerspective(frame, M, (w, h), borderValue=(255, 255, 255))

    try:
        canny = make_canny(img2)
        lines = cv2.HoughLinesP(canny, 3, np.pi/180, 100, np.array([]), 100, 400)
        
        x2_coord_average = (l2[0][2] + l1[0][2]) / 2

        turning_rate = x2_coord_average - 400

        if turning_rate < -10:
            print("left")
        elif turning_rate > 10:
            print("right")
        else:
            print("straight")

        left_line_image = display_lines(img2, np.array(l1))
        right_line_image = display_lines(img2, np.array(l2))

        line_image = cv2.addWeighted(left_line_image, 1, right_line_image, 1, 1)

        combo_image = cv2.addWeighted(img2, 1, line_image, 0.6, 1)

        print('Frame took{}'.format(time.time() - last_time))
        last_time = time.time()
        cv2.imshow('result', combo_image)

    except:
        pass

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
cap.release()
```

## 감사합니다.

---------
---------
###### 업로드된 파일의 코드는 오차를 수정하기 위해 다소 바뀐 부분이 있으나 지저분한 관계로 내용에서 제거하였습니다.
###### 수정된 내용
#### 1. l1과 l2의 값이 갑작스럽게 크게 바뀌는 경우 에러라고 판단하고 이전의 값을 가지고 그대로 사용하였습니다.
###### 그러기 위해 l1과 l2의 값을 백업할 변수 l1_copy, l2_copy를 만들었습니다.
#### 2. 간혹 좌표값 반환 과정에서 에러가 생겨 try 문으로 묶은 뒤 에러 발생 시 return 값을 0 으로 만들고 해당 값이 0 일 경우 무시하고 진행할 수 있게 수정하였습니다. 
```
  import cv2
  import numpy as np
  import time


  def make_coordinates(img, line_parameters):
      slope, intercept = line_parameters

      y1 = img.shape[0]
      y2 = int(y1 * (1/2))

      x1 = int((y1 - intercept)/slope)
      x2 = int((y2 - intercept)/slope)

      return np.array([x1, y1, x2, y2])


  def average_slope_intercept(img, lines):

      left_fit = []
      right_fit = []

      for line in lines:
          x1, y1, x2, y2 = line.reshape(4)

          x = np.array([x1, x2])
          y = np.array([y1, y2])
          A = np.vstack([x, np.ones(len(x))]).T

          slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

          x_coord = -((intercept-640) / slope)

          if x_coord < 400:
              left_fit.append((slope, intercept))

          elif x_coord > 400:
              right_fit.append((slope, intercept))

      left_fit_average = np.mean(left_fit, 0)
      right_fit_average = np.mean(right_fit, 0)

      try:
          left_line = make_coordinates(img, left_fit_average)
          right_line = make_coordinates(img, right_fit_average)

      except:
          left_line = 0
          right_line = 0
          pass

      return [left_line], [right_line]


  def display_lines(img, lines):
      line_image = np.zeros_like(img)

      if lines is not None:
          for line in lines:
              x1, y1, x2, y2 = line.reshape(4)
              cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 20)

      return line_image


  def make_canny(img):
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      _, binary = cv2.threshold(gray, 197, 255, cv2.THRESH_BINARY)
      binary_gaussian = cv2.adaptiveThreshold(binary, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
      blur = cv2.GaussianBlur(binary_gaussian, (5, 5), 0)
      blur = cv2.bilateralFilter(blur, 9, 75, 75)

      canny_image = cv2.Canny(blur, 80, 120)

      return canny_image


  cap = cv2.VideoCapture('project_video.mp4')
  last_time = time.time()

  h = 640
  w = 800
  pts1 = np.float32([[300, 650], [580, 460], [720, 460], [1100, 650]])
  pts2 = np.float32([[200, 640], [200, 0], [600, 0], [600, 640]])
  M = cv2.getPerspectiveTransform(pts1, pts2)

  l1 = 0
  l2 = 0
  l1_copy = None
  l2_copy = None

  x2_coord_average = 400

  while(True):
      ret, frame = cap.read()
      img2 = cv2.warpPerspective(frame, M, (w, h), borderValue=(255, 255, 255))

      try:
          canny = make_canny(img2)
          lines = cv2.HoughLinesP(canny, 3, np.pi/180, 100, np.array([]), 100, 400)

          print(len(lines))

          if l1 == 0 and l2 == 0:
              l1, l2 = average_slope_intercept(canny, lines)
          else:
              l1_copy, l2_copy = average_slope_intercept(canny, lines)

          if l1_copy is not None:
              try:
                  if l1_copy[0][0] > l1[0][0] + 20 or l1_copy[0][0] < l1[0][0] - 20:
                      l1 = l1
                  else:
                      l1 = l1_copy

                  if l2_copy is not None:
                      if l2_copy[0][0] > l2[0][0] + 20 or l2_copy[0][0] < l2[0][0] - 20:
                          l2 = l2
                      else:
                          l2 = l2_copy

              except:
                  pass

          elif l1_copy is None:
              try:
                  if l2_copy is not None:
                      if l2_copy[0][0] > l2[0][0] + 20 or l2_copy[0][0] < l2[0][0] - 20:
                          l2 = l2
                      else:
                          l2 = l2_copy

              except:
                  pass

          x2_coord_average = (l2[0][2] + l1[0][2]) / 2

          turning_rate = x2_coord_average - 400

          if turning_rate < -10:
              print("left")
          elif turning_rate > 10:
              print("right")
          else:
              print("straight")

          left_line_image = display_lines(img2, np.array(l1))
          right_line_image = display_lines(img2, np.array(l2))

          line_image = cv2.addWeighted(left_line_image, 1, right_line_image, 1, 1)

          combo_image = cv2.addWeighted(img2, 1, line_image, 0.6, 1)

          print('Frame took{}'.format(time.time() - last_time))
          last_time = time.time()
          cv2.imshow('frame', frame)
          cv2.imshow('img2', img2)
          cv2.imshow('result', combo_image)
          cv2.imshow('canny', canny)

      except:
          pass

      if cv2.waitKey(1) & 0xFF == 27:
          break

  cv2.destroyAllWindows()
  cap.release()
```

## 감사합니다.
