import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 모델과 프로세서 로드
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 이미지 로드
image = Image.open("horse.jpg")  # horse.jpg 파일은 같은 디렉토리에 있어야 합니다.

# 텍스트와 이미지 처리
inputs = processor(
    text=["a photo of a horse", "a photo of a dog", 
          "a photo of a bear", "a photo of a person"], 
    images=image, return_tensors="pt", padding=True
)

# 모델 추론
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # 유사도 점수
probs = logits_per_image.softmax(dim=1)  # 점수를 확률로 변환

# 결과 시각화
plt.imshow(image)
plt.title(f"Probabilities: horse: {probs[0,0]:.2f}, dog: {probs[0,1]:.2f}, bear: {probs[0,2]:.2f}, person: {probs[0,3]:.2f}")
plt.show()
