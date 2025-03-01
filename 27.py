import cv2
import os
import numpy as np
import random

# === 원본 및 증강 이미지 저장 경로 ===
HOME_DIR = os.path.expanduser("~")
INPUT_DIR = os.path.join(HOME_DIR, "stand_camera")
OUTPUT_DIR = os.path.join(HOME_DIR, "stand_camera_augument")

# 폴더가 없으면 생성
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 데이터 증강 함수 ===
def augment_image(image):
    """ 화질 개선 & 다양한 회전 방식으로 증강 """
    augmented_images = []

    h, w = image.shape[:2]

    # 1. 임의의 각도로 회전 (1~359도)
    angle = random.randint(1, 359)
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 회전 후 새로운 크기 계산
    cos_val = abs(rotation_matrix[0, 0])
    sin_val = abs(rotation_matrix[0, 1])
    new_w = int((h * sin_val) + (w * cos_val))
    new_h = int((h * cos_val) + (w * sin_val))

    # 변환 행렬 수정 (중앙 정렬)
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE)
    augmented_images.append(rotated)

    # 2. 좌우 반전 (50% 확률)
    if random.random() > 0.5:
        flipped_h = cv2.flip(image, 1)
        augmented_images.append(flipped_h)

    # 3. 밝기 조절 (자연스럽게 변형, 범위 증가)
    alpha = random.uniform(0.9, 1.1)  # 0.9 (어둡게) ~ 1.1 (밝게)
    bright_image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    augmented_images.append(bright_image)

    # 4. 화질 개선 (업스케일링 유지)
    upscale = cv2.resize(image, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    downscale = cv2.resize(upscale, (w, h), interpolation=cv2.INTER_CUBIC)
    augmented_images.append(downscale)

    # 5. 확대 & 크롭 (확대 범위 증가)
    scale_factor = random.uniform(1.1, 1.3)  # 기존 1.1~1.2 → 1.1~1.3
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # 중앙 크롭 (이미지 짤리는 문제 해결)
    start_x = (new_w - w) // 2
    start_y = (new_h - h) // 2
    cropped_img = resized_img[start_y:start_y + h, start_x:start_x + w]
    augmented_images.append(cropped_img)

    return augmented_images

# === 500장 될 때까지 반복 ===
image_files = [f for f in os.listdir(INPUT_DIR) if f.endswith((".png", ".jpg", ".jpeg"))]

if not image_files:
    print("❌ 원본 이미지가 없습니다! ~/original_images 폴더에 이미지를 추가하세요.")
    exit(1)

print(f"\n📊 총 {len(image_files)}장의 이미지를 500장 이상으로 증강합니다.\n")

target_count = 500  # 목표 이미지 개수

existing_count = len(os.listdir(OUTPUT_DIR))
generated_count = 0

while existing_count + generated_count < target_count:
    for idx, file_name in enumerate(image_files):
        if existing_count + generated_count >= target_count:
            break

        img_path = os.path.join(INPUT_DIR, file_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"❌ 이미지 로드 실패: {file_name}")
            continue

        augmented_images = augment_image(image)

        for j, aug_img in enumerate(augmented_images):
            if existing_count + generated_count >= target_count:
                break

            new_filename = f"aug_{idx}_{j}_{generated_count}.jpg"
            save_path = os.path.join(OUTPUT_DIR, new_filename)

            success = cv2.imwrite(save_path, aug_img)
            if success:
                generated_count += 1
                print(f"✅ 저장됨: {save_path}")
            else:
                print(f"❌ 저장 실패: {save_path}")

print(f"\n✅ 데이터 증강 완료! {OUTPUT_DIR} 폴더를 확인하세요. 총 {generated_count}장 추가 생성됨.")
