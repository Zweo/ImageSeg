import os
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
            pos = (256 * j, 256 * i, 256 * (j + 1), 256 * (i + 1))
            res.append(img.crop(pos))
    return n, res, oring_size, img.size


def mergeImage(n, imgs, o_size, n_size):
    img = Image.new('L', n_size)
    for i in range(n):
        for j in range(n):
            pos = (256 * j, 256 * i, 256 * (j + 1), 256 * (i + 1))
            img.paste(imgs[i * n + j], pos)
    img = img.resize(o_size)
    return img


def predict(model, input_path, output_dir):
    name, _ = os.path.splitext(input_path)
    name = os.path.split(name)[-1] + ".png"
    n, imgs, o_size, n_size = cutImage(input_path)
    res = []
    for img in imgs:
        img = torch.from_numpy(np.array(img)).float().unsqueeze(0)
        img = img / 255
        if torch.cuda.is_available():
            img = img.cuda()
        img = img.permute(0, 3, 1, 2)
        label = model(img)
        label = torch.argmax(label,
                             dim=1).cpu().squeeze().numpy().astype(np.uint8)
        res.append(Image.fromarray(label))
    img = mergeImage(n, res, o_size, n_size)
    img.save(os.path.join(output_dir, name))


if __name__ == "__main__":
    from model_define import init_model
    model = init_model()
    predict(model, '../data/images/10.tif', '')
