import requests
from requests.auth import HTTPBasicAuth

# 이미지 파일 경로 지정
IMAGE_FILE_PATH = '/home/seungrok/E-3-5 2025-02-10 181824/pico/100.jpg' # 실제 이미지 파일 경로로 변경

# 액세스 키 설정
ACCESS_KEY = "vpiu1lDi5T7x2rNqaQiIU9sak1MpCDMV8TixTJZt"  # 실제 액세스 키로 변경

# 이미지 파일을 바이너리로 읽기
with open(IMAGE_FILE_PATH, "rb") as image_file:
    image_data = image_file.read()

# API 요청 보내기
response = requests.post(
    url="https://suite-endpoint-api-apne2.superb-ai.com/endpoints/237bfa87-a3a2-4d7f-88a6-db275c671cd1/inference",  # 실제 엔드포인트 URL로 변경
    auth=HTTPBasicAuth("kdt2025_1-33", "vpiu1lDi5T7x2rNqaQiIU9sak1MpCDMV8TixTJZt"),
    headers={"Content-Type": "image/jpeg"},
    data=image_data,
)

# 응답 출력
try:
    print(response.json())  # JSON 응답 출력
except requests.exceptions.JSONDecodeError:
    print("응답이 JSON 형식이 아닙니다:", response.text)  # JSON 형식이 아닐 경우 출력

