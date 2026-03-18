import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager, rcParams

# =========================
# 0. 解决中文显示为方框的问题
# =========================
available_fonts = {f.name for f in font_manager.fontManager.ttflist}

candidate_fonts = [
    "SimHei",               # 黑体，Windows 常见
    "Microsoft YaHei",      # 微软雅黑，Windows 常见
    "Noto Sans CJK SC",     # Linux/macOS 常见
    "Source Han Sans SC",   # 思源黑体
    "WenQuanYi Zen Hei",    # Linux 常见
    "Arial Unicode MS",     # 部分 macOS
]

chosen_font = None
for f in candidate_fonts:
    if f in available_fonts:
        chosen_font = f
        break

if chosen_font is not None:
    rcParams["font.sans-serif"] = [chosen_font]
else:
    print("未找到可用中文字体，请先安装中文字体，否则中文仍可能显示为方框。")

rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# =========================
# 1. 原始数据
# =========================
noise_levels = [2, 4, 6, 8, 10]
b_values = [0, 1000, 2000]

data = {
    "噪声图像": {
        0: {
            "PSNR":  [33.2913, 28.2431, 24.8505, 23.6728, 21.7530],
            "SSIM":  [0.9673, 0.9244, 0.8818, 0.8543, 0.8272],
            "LPIPS": [0.1023, 0.3112, 0.4709, 0.6173, 0.7251],
        },
        1000: {
            "PSNR":  [34.4805, 28.4159, 24.7338, 23.1914, 20.9725],
            "SSIM":  [0.9430, 0.8615, 0.8043, 0.7788, 0.7529],
            "LPIPS": [0.3009, 0.6221, 0.7956, 0.8570, 0.9018],
        },
        2000: {
            "PSNR":  [34.6626, 28.0992, 24.0966, 22.4555, 20.2404],
            "SSIM":  [0.9327, 0.8392, 0.7799, 0.7566, 0.7350],
            "LPIPS": [0.3792, 0.6609, 0.7632, 0.8175, 0.9024],
        },
        "AE_all": [13.9110, 30.8974, 42.4867, 48.5138, 51.5477]
    },

    "基线模型": {
        0: {
            "PSNR":  [36.6699, 34.9022, 32.6135, 31.5224, 30.7577],
            "SSIM":  [0.9863, 0.9823, 0.9711, 0.9594, 0.9516],
            "LPIPS": [0.0200, 0.0442, 0.0784, 0.0969, 0.1292],
        },
        1000: {
            "PSNR":  [41.2228, 38.2867, 34.7500, 31.2431, 34.6735],
            "SSIM":  [0.9858, 0.9731, 0.9526, 0.9170, 0.9472],
            "LPIPS": [0.0789, 0.1895, 0.2525, 0.3079, 0.2728],
        },
        2000: {
            "PSNR":  [42.3886, 39.8304, 32.7429, 33.6000, 36.0449],
            "SSIM":  [0.9855, 0.9738, 0.8833, 0.9111, 0.9502],
            "LPIPS": [0.0741, 0.1597, 0.3011, 0.2691, 0.2560],
        },
        "AE_all": [10.3219, 20.0160, 28.6013, 36.2018, 37.3664]
    },

    "改进之后": {
        0: {
            "PSNR":  [40.0172, 35.9322, 33.8919, 33.3060, 32.6617],
            "SSIM":  [0.9939, 0.9876, 0.9803, 0.9739, 0.9685],
            "LPIPS": [0.0134, 0.0288, 0.0453, 0.0533, 0.0627],
        },
        1000: {
            "PSNR":  [43.4928, 39.5068, 36.4136, 32.1806, 36.5118],
            "SSIM":  [0.9915, 0.9808, 0.9661, 0.9206, 0.9609],
            "LPIPS": [0.0410, 0.0999, 0.1808, 0.2391, 0.2094],
        },
        2000: {
            "PSNR":  [44.6443, 40.7231, 35.0750, 31.7253, 38.0488],
            "SSIM":  [0.9916, 0.9813, 0.9340, 0.8414, 0.9654],
            "LPIPS": [0.0461, 0.1012, 0.2014, 0.3033, 0.2195],
        },
        "AE_all": [8.5871, 17.2385, 24.5006, 32.4364, 32.5814]
    }
}

# =========================
# 2. 绘图风格
# =========================
method_colors = {
    "噪声图像": "#1f77b4",
    "基线模型": "#ff7f0e",
    "改进之后": "#2ca02c"
}

b_linestyles = {
    0: "-",
    1000: "--",
    2000: ":"
}

metrics = ["PSNR", "SSIM", "LPIPS", "AE"]

# =========================
# 3. 创建 2x2 子图
# =========================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, metric in zip(axes, metrics):
    if metric != "AE":
        for method in ["噪声图像", "基线模型", "改进之后"]:
            for b in b_values:
                y = data[method][b][metric]
                ax.plot(
                    noise_levels, y,
                    color=method_colors[method],
                    linestyle=b_linestyles[b],
                    marker="o",
                    linewidth=2,
                    markersize=6
                )
    else:
        for method in ["噪声图像", "基线模型", "改进之后"]:
            y = data[method]["AE_all"]
            ax.plot(
                noise_levels, y,
                color=method_colors[method],
                linestyle="-",
                marker="s",
                linewidth=2.5,
                markersize=6
            )

    ax.set_title(metric, fontsize=14, fontweight="bold")
    ax.set_xlabel("噪声水平 (%)", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_xticks(noise_levels)
    ax.grid(True, linestyle="--", alpha=0.4)

# =========================
# 4. 自定义图例
# =========================
method_legend = [
    Line2D([0], [0], color=method_colors["噪声图像"], lw=2.5, label="噪声图像"),
    Line2D([0], [0], color=method_colors["基线模型"], lw=2.5, label="基线模型"),
    Line2D([0], [0], color=method_colors["改进之后"], lw=2.5, label="改进之后"),
]

b_legend = [
    Line2D([0], [0], color="black", lw=2, linestyle=b_linestyles[0], label="b = 0"),
    Line2D([0], [0], color="black", lw=2, linestyle=b_linestyles[1000], label="b = 1000"),
    Line2D([0], [0], color="black", lw=2, linestyle=b_linestyles[2000], label="b = 2000"),
]

fig.legend(
    handles=method_legend + b_legend,
    loc="upper center",
    ncol=6,
    fontsize=11,
    frameon=True,
    bbox_to_anchor=(0.5, 1.02)
)

fig.suptitle("不同噪声水平下各方法在不同 b 值上的指标变化", fontsize=16, fontweight="bold", y=1.06)

plt.tight_layout()
plt.savefig("result/result_plot/metrics_comparison_2.5D.png", dpi=600, bbox_inches="tight")
plt.show()


