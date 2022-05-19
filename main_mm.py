"""
DISCONTINUED, WORKING ON COLAB
"""

'''
https://mmocr.readthedocs.io/en/latest/install.html

conda create -n mmocr_env python=3.7 -y
conda activate mmocr_env
conda install pytorch==1.6.0 torchvision==0.7.0 -c pytorch
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
pip install mmdet
git clone https://github.com/open-mmlab/mmocr.git
cd mmocr
pip install -r requirements.txt
pip install -v -e .  # or "python setup.py develop"
export PYTHONPATH=$(pwd):$PYTHONPATH
'''

from mmocr.utils.ocr import MMOCR

# Load models into memory
ocr = MMOCR(det='TextSnake', recog=None)

# Inference
img_path = 'testimgs/1.png'
out_img_path = 'mm_resimgs/1.png'

results = ocr.readtext(img_path, output=out_img_path, export='demo/')
print(results)