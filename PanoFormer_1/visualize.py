import matplotlib.pyplot as plt



import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# def depth_read(filename):
#     depth_png = np.array(Image.open(filename), dtype=int)
#     assert np.max(depth_png) > 255, "Depth map should be 16-bit"
#     depth = depth_png.astype(np.float32) / 256.0
#     return depth

def show_depth(groundtruth):
    groundtruth=np.array(groundtruth.detach().to("cpu"))
    normalized_depth = groundtruth.astype(np.float32) / groundtruth.max()
    cmap = plt.get_cmap('viridis')
    color_image = cmap(normalized_depth)

    if color_image.shape[-1] == 4:
        color_image = color_image[..., :3]

    plt.imshow(color_image)
    plt.colorbar()
    plt.title("Depth Map Visualization")
    plt.show()