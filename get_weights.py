# TODO: Needs debugging

from efficientnet import EfficientNetB0

model = EfficientNetB0(weights='imagenet', include_top=False)
model.save('weights/EfficientNetB0.pth')
print("Successfully loaded EfficientNetB0")

