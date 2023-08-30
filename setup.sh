pip install braceexpand
pip install timm==0.6.13
pip install google-cloud-storage
pip install tensorboardX
pip install ftfy
pip install regex
pip install setuptools==59.5.0
pip install transformers==4.27.2
pip install diffusers==0.20.0
pip install git+https://github.com/openai/CLIP.git
pip install torchmetrics
pip install open_clip_torch
pip install pycocotools
pip uninstall safetensors -y
# pip install google-cloud-storage
# pip install -r requirements.txt
pip install smart_open
pip install opencv-python
pip install scikit-image

pip install xformers
pip install streamlit
pip install flask
pip install -U flask-cors
pip install invisible-watermark>=0.2.0 # for sdxl


if test -f "/root/.bashrc"; then
  echo """PATH="/export/home/google-cloud-sdk/bin:\$PATH"""" >> /root/.bashrc
  source /root/.bashrc
fi


