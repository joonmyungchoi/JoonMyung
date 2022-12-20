import torchvision.models as models
import torchvision.transforms as T
import torchvision.datasets as D
import torch
import numpy as np


model = models.resnet18(pretrained=True)

normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])

transform = T.Compose([
                T.Resize(32),
                T.CenterCrop(32),
                T.ToTensor(),
                normalize
            ])
dataset = D.CIFAR100(root="../data", train = True, download=True, transform=transform)


################## 1. 모델 정보 출력 ##################
from torchinfo import summary

def modelStatus():
    summary(model, (250, 3, 32, 32))
    # Estimated Total Size : 예상 GPU 사용량

################## 2. 학습 상태 모니터링 ##################
import torchvision as viz

# 1. visdom server open : python -m visdom.server
# 2. 이미지 출력
def learningStatusMonitoring():
    image_window = viz.image(
        np.random.rand(3,256,256),
        opts=dict(
            title = "random",
            caption = "random noise"
        )
    )


################## 3. 모델 시각화(graphviz) ##################
# 메모리에 올라간 모델 시각화

from torchviz import make_dot
from torch.autograd import Variable

def modelVisualization1():
    # Variable을 통하여 Input 생성
    x = Variable(torch.randn(10,3,32,32))

    # 앞에서 생성한 model에 Input을 x로 입력한 뒤 (model(x))  graph.png 로 이미지를 출력합니다.
    make_dot(model(x), params=dict(model.named_parameters())).render("graph", format="png")

################## 4. 모델 시각화(netron) ##################



def modelVisualization2():
    params = model.state_dict()
    dummy_data = torch.empty(1, 3, 32, 32, dtype = torch.float32)

    # 4. onnx 파일을 export 해줍니다. 함수에는 차례대로 model, data, 저장할 파일명 순서대로 들어가면 됩니다.
    torch.onnx.export(model, dummy_data, "output.onnx")



modelVisualization2()