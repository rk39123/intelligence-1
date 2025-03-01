import cv2
import requests
from requests.auth import HTTPBasicAuth

# 이미지 파일 경로
IMAGE_FILE_PATH = "/home/seungrok/E-3-5 2025-02-10 181824/pico/100.jpg"

# API 요청을 보내서 객체 탐지 수행
ACCESS_KEY = "vpiu1lDi5T7x2rNqaQiIU9sak1MpCDMV8TixTJZt"
response = requests.post(
    url="https://suite-endpoint-api-apne2.superb-ai.com/endpoints/237bfa87-a3a2-4d7f-88a6-db275c671cd1/inference",
    auth=HTTPBasicAuth("kdt2025_1-33", ACCESS_KEY),
    headers={"Content-Type": "image/jpeg"},
    data=open(IMAGE_FILE_PATH, "rb").read(),
)

# JSON 응답 파싱
try:
    detection_results = response.json()
except requests.exceptions.JSONDecodeError:
    print("응답이 JSON 형식이 아닙니다:", response.text)
    exit()

# 이미지 불러오기
img = cv2.imread(IMAGE_FILE_PATH)

# 관심 있는 객체 리스트 및 색상 지정 (BGR 형식)
CLASS_COLORS = {
    "RASPEBBRY PICO": (0, 255, 0),     # 초록색
    "HOLE": (0, 0, 255),               # 빨간색
    "OSCILATOR": (255, 0, 0),          # 파란색
    "CHIPSET": (255, 165, 0),          # 주황색
    "USB": (128, 0, 128),              # 보라색
    "BOOTSEL": (0, 255, 255)           # 노란색
}

# 박스 및 텍스트 추가
for obj in detection_results.get("objects", []):
    obj_class = obj["class"]
    score = obj["score"]
    box = obj["box"]  # [x_min, y_min, x_max, y_max]

    # 관심 있는 클래스만 처리
    if obj_class in CLASS_COLORS:
        color = CLASS_COLORS[obj_class]  # 클래스별 색상 선택
        start_point = (box[0], box[1])  # (x_min, y_min)
        end_point = (box[2], box[3])  # (x_max, y_max)

        # 박스 그리기
        cv2.rectangle(img, start_point, end_point, color, 2)

        # 텍스트 추가 (객체명 + 신뢰도)
        text = f"{obj_class} ({score:.2f})"
        position = (box[0], max(box[1] - 10, 20))  # 박스 위쪽에 표시
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

# 이미지 출력
cv2.imshow("Detected Objects", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
