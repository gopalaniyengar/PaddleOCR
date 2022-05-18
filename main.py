'''
conda create --name paddle_env python=3.8 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda activate paddle_env
conda deactivate
conda env list
conda list
which python
which paddleocr
'''

'''
Paddleocr supports Chinese, English, French, German, Korean and Japanese.
You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
to switch the language model in order.
'''

from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import numpy as np
import os

img_names_mid = os.listdir('testimgs/')
img_names = ['testimgs/'+img_name for img_name in img_names_mid]
# print(img_names)

ocr = PaddleOCR(use_angle_cls=True, lang='japan') # need to run only once to download and load model into memory

for k in range(len(img_names)):

    '''Perform OCR'''
    img_path = img_names[k]
    result = ocr.ocr(img_path, cls=True)
    print('------------------------------------------------------------------------------------------------------------------------\n')
    # for line in result:
    #     print(line)

    '''Draw Image'''
    image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores, font_path='font2.ttf')
    im_show = Image.fromarray(im_show)
    im_show.save(f'resimgs/{img_path.split("/")[-1]}')