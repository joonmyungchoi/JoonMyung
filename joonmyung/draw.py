import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def draw(matrixes, vmin=None, vmax=None, col=1, p=False, title=[], fmt=1):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    row = (len(matrixes) - 1) // col + 1
    if p:
        print("row : {}, col : {}".format(row, col))
        print("height : {}, width : {}".format(row * 8, col * 8))

    title = title + list(range(len(title), len(matrixes) - len(title)))
    fig, axes = plt.subplots(nrows=row, ncols=col, squeeze=False)
    fig.set_size_inches(col * 8, row * 8)

    for e, matrix in enumerate(matrixes):
        if type(matrix) == torch.Tensor:
            matrix = matrix.detach().cpu().numpy()
        ax = axes[e // col][e % col]
        sns.heatmap(pd.DataFrame(matrix), annot=True, fmt=".{}f".format(fmt), cmap='Greys'
                    , yticklabels=False, xticklabels=False, vmin=vmin, vmax=vmax
                    , linewidths=.1, linecolor='black'
                    , ax=ax)
        ax.set(title=title[e])
    plt.show()
