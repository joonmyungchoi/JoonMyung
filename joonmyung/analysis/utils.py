


def get_update_param(model):
    return [k for k, v in model.named_parameters() if v.requires_grad == True]

def shape(param):
    return f"({', '.join(map(str, param.shape))})"

def analysis_mode():
    from joonmyung.draw import drawImgPlot, drawLinePlot, drawHeatmap, drawBarChart
    from joonmyung.utils import to_np
    print(get_update_param)
    print(drawImgPlot)

if __name__ == '__main__':
    analysis_mode()
