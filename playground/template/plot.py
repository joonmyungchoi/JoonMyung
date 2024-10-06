import numpy as np
import matplotlib.pyplot as plt
class Radar(object):
    def __init__(self, fig, titles, intervals, rotation=0, block_num=4, rect=None, round_num=1):

        if rect is None:
            rect = [0.05, 0.05, 0.95, 0.95]
        self.intervals = intervals
        self.block_num = block_num
        label_TF = [False] + [True] * (self.block_num-1) + [False]

        self.gaps = [(interval[1] - interval[0]) / (self.block_num) for interval in self.intervals]
        self.n = len(titles)
        self.angles = np.arange(0, 360, 360.0/self.n)
        self.axes = [fig.add_axes(rect, projection="polar", label="axes%d" % i) for i in range(self.n)]

        self.ax = self.axes[0]
        self.ax.set_thetagrids(self.angles, labels=titles, fontsize=14, weight='bold')
        for ax in self.axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)

        for ax, angle, interval, gap in zip(self.axes, self.angles, self.intervals, self.gaps):
            label = [round(interval[0] + block_num * gap, round_num) if label_TF[block_num] else ""
                     for block_num in range(0, self.block_num+1)]

            ax.set_rgrids(range(1, self.block_num+2), angle=angle, labels=label, fontsize=14)
            ax.spines["polar"].set_visible(False)
            ax.set_ylim(0, self.block_num+2)
            ax.set_theta_offset(np.deg2rad(rotation))

    def plot(self, values, *args, **kw):
        values_ratio = []
        for val, interval, gap in zip(values, self.intervals, self.gaps):
            values_ratio.append((val - interval[0]) / gap + 1)
            # first, interval = self.intervals[i][0], (self.intervals[i][1] - self.intervals[i][0]) / self.blocks
            # values[i] = (val - first) / interval + 1
        angle = np.deg2rad(np.r_[self.angles, self.angles[0]])
        values_ratio = np.r_[values_ratio, values_ratio[0]]
        self.ax.plot(angle, values_ratio, *args, **kw)
        self.ax.fill(angle, values_ratio, *args, alpha=0.1)

fig = plt.figure(figsize=(6, 6))
titles = ['MSRVTT\n(FT T2V)', 'MSVD\n(FT T2V)', 'ActivityNet\n(FT T2V)', 'DiDeMo\n(FT T2V)',
          'LSMDC\n(FT T2V)', 'SSV2-label\n(FT T2V)', 'SSV2-Template\n(FT T2V)',
          'MSRVTT\n(FT VQA)', 'MSVD\n(FT VQA)']
intervals = [[45, 51], [57, 63], [51, 58], [57, 63], [27, 33], [58, 64], [71, 75], [42, 45], [48, 50]]
radar = Radar(fig, titles, intervals=intervals, block_num = 3)

umt       = [50.0, 62.1, 57.2, 62.1, 32.7, 64.0, 74.6, 44.9, 49.5]
tome      = [47.0, 59.6, 51.7, 57.3, 27.3, 60.2, 71.8, 42.3, 48.5]
ours      = [50.8, 62.7, 56.6, 62.4, 32.4, 63.8, 74.0, 44.8, 49.4]

radar.plot(umt,  "-",  lw=2, color="b", alpha=0.4, label="UMT") # UMT]
radar.plot(tome, "-", lw=2, color="g", alpha=0.4, label="UMT+ToMe") # UMT
radar.plot(ours, "-", lw=2, color="r", alpha=0.4, label="UMT+vid-TLDR (Ours)") # Ours

print(1)