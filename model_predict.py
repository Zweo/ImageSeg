import os
import cv2
import torch
import numpy as np
from PIL import Image


def cutImage(file_name):
    img = Image.open(file_name)
    oring_size = img.size
    n = img.size[0] // 256
    img = img.resize((n * 256, n * 256))
    res = []
    for i in range(n):
        for j in range(n):
            res.append(
                img.crop((256 * j, 256 * i, 256 * (j + 1), 256 * (i + 1))))
    return n, res, oring_size, img.size


def mergeImage(n, imgs, o_size, n_size):
    img = Image.new('P', n_size)
    for i in range(n):
        for j in range(n):
            img.paste(imgs[i * n + j],
                      (256 * j, 256 * i, 256 * (j + 1), 256 * (i + 1)))
    img = img.resize(o_size)
    return img


def predict(model, input_path, output_dir):
    name, _ = os.path.splitext(input_path)
    name = os.path.split(name)[-1] + ".png"
    img = cv2.imread(input_path).astype(np.float32)

    # ox, oy = img.shape[0], img.shape[1]
    # img = cv2.resize(img, (256, 256))
    # img = torch.from_numpy(img).unsqueeze(0)
    # if torch.cuda.is_available():
    #     img = img.cuda()
    # img = img.permute(0, 3, 1, 2)
    # label = model(img)
    # label = torch.argmax(label, dim=1).cpu().squeeze().numpy().astype(np.uint8)
    # label = cv2.resize(label, (ox, oy))

    cv2.imwrite(os.path.join(output_dir, name), label)


if __name__ == "__main__":
    from model_define import init_model
    mode = init_model()
    predict(mode, '../data/images/1.tif', '')
