from PIL import Image
def make_square(image):
    width, height = image.size
    # 작은 쪽 길이에 맞춰 자르기
    new_size = min(width, height)

    # 중앙을 기준으로 자르기
    left = (width - new_size) // 2
    top = (height - new_size) // 2
    right = (width + new_size) // 2
    bottom = (height + new_size) // 2

    return image.crop((left, top, right, bottom))


def crop_image_4x4(image_path):
    # 이미지 열기
    img = Image.open(image_path)

    # 이미지를 정사각형으로 만들기
    square_img = make_square(img)

    # 이미지 크기 얻기
    width, height = square_img.size

    # 자를 크기 계산 (4x4로 자르기)
    crop_width = width // 4
    crop_height = height // 4

    # 자른 이미지 저장
    img_number = 1
    for row in range(4):
        for col in range(4):
            left = col * crop_width
            upper = row * crop_height
            right = left + crop_width
            lower = upper + crop_height
            cropped_img = square_img.crop((left, upper, right, lower))

            # 자른 이미지 저장
            cropped_img.save(f"cropped_{img_number}.png")
            img_number += 1


# 사용 예시
image_path = './asset/cat_dog.jpeg'
crop_image_4x4(image_path)