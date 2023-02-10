from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

def reshape_swin(data, height=14, width=14):
    result = data.reshape(data.size(0), height, width, data.size(2))

    result = result.transpose(2, 3).transpose(1, 2)

    return result

def reshape_vit(data, height=14, width=14):
    result = data[:, 1:  , :].reshape(data.size(0), height, width, data.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


methods_grad = \
    {"gradcam": GradCAM,
     "scorecam": ScoreCAM,
     "gradcam++": GradCAMPlusPlus,
     "ablationcam": AblationCAM,
     "xgradcam": XGradCAM,
     "eigencam": EigenCAM,
     "eigengradcam": EigenGradCAM,
     "layercam": LayerCAM,
     "fullgrad": FullGrad}






