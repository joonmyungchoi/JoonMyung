# Define your model
from torchcam.utils import overlay_mask

from joonmyung.analysis import JDataset
from torchvision.transforms.functional import to_pil_image
from torchvision.models import resnet18
from torchcam.methods import SmoothGradCAMpp
import matplotlib.pyplot as plt

model = resnet18(pretrained=True).eval()
dataset = JDataset(device="cpu")
inputs_tensor, _, imgs, _ = dataset[[10, 1]]

cam_extractor = SmoothGradCAMpp(model)
# Preprocess your data and feed it to the model
out = model(inputs_tensor)
# Retrieve the CAM by passing the class index and the model output
cams = cam_extractor(out.squeeze(0).argmax().item(), out)

for name, cam in zip(cam_extractor.target_names, cams):
  plt.imshow(cam.squeeze(0).numpy()); plt.axis('off'); plt.title(name); plt.show()
# Overlayed on the image
for name, cam in zip(cam_extractor.target_names, cams):
  result = overlay_mask(to_pil_image(dataset.unNormalize(inputs_tensor)[0]), to_pil_image(cam.squeeze(0), mode='F'), alpha=0.5)
    # img  : (3, 224, 224)
    # mask : (7, 7)
  plt.imshow(result); plt.axis('off'); plt.title(name); plt.show()
