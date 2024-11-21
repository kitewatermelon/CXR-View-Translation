import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class PreprocessImageTransform:
    def __init__(self, resize_dim=(512, 512)):
        self.resize_dim = resize_dim

    def get_skew_angle(self, image):
        if len(image.shape) == 3:  # 컬러 이미지인지 확인
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        lines = cv2.HoughLinesP(binary, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
        angles = []

        if lines is not None:
            # 라인을 시각화하기 위해 원본 이미지 복사본 생성
            image_with_lines = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # 컬러로 변환하여 선을 그리기 위해

            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 초록색 선으로 그리기

                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)

            # 디버깅을 위해 이미지 출력
            cv2.imshow("Detected Lines", image_with_lines)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            return np.mean(angles)
        else:
            print("No lines detected.")
            return 0  # 기울기를 찾지 못한 경우 기본 각도


    def rotate_image(self, image, angle):
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        return rotated_image

    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)

        skew_angle = self.get_skew_angle(image)
        rotated_image = self.rotate_image(image, skew_angle)
        processed_image = Image.fromarray(rotated_image)

        return processed_image

class RemoveZeroPadding:
    def __call__(self, image):
        image_np = np.array(image)
        if image_np.ndim == 3:
            non_zero_rows = np.any(image_np != 0, axis=(1, 2))
            non_zero_cols = np.any(image_np != 0, axis=(0, 2))
        else:
            non_zero_rows = np.any(image_np != 0, axis=1)
            non_zero_cols = np.any(image_np != 0, axis=0)

        cropped_image_np = image_np[non_zero_rows][:, non_zero_cols]
        return Image.fromarray(cropped_image_np)

# Transform pipeline 정의
transform = [
        transforms.Compose([
        RemoveZeroPadding(),
        # PreprocessImageTransform(resize_dim=(512, 512)),
        # transforms.RandomRotation(degrees=30, fill=0),  # 회전 추가, 30도 범위 내에서 무작위 회전
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),

    transforms.Compose([
        RemoveZeroPadding(),
        transforms.Resize((256, 256)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # RGB 정규화
    ])
]