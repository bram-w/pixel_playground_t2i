import numpy as np
from PIL import Image
from io import BytesIO

import gen_functions

import os

# import torch
from flask import Flask, request, send_file
from flask_cors import CORS
# from lavis.models import load_model_and_preprocess

os.environ['TRANSFORMERS_CACHE'] = "/export/home/huggingface"

app = Flask(__name__)
CORS(app)

from huggingface_hub import login
with open('hf_auth', 'r') as f:
    auth_token = f.readlines()[0].strip()
login(auth_token)

def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    buffered.seek(0)

    return buffered

def decode_image(img_obj):
    img = Image.open(img_obj).convert("RGB")
    return img


def convert_to_bool(value):
    if isinstance(value, bool):
        return value

    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    else:
        raise ValueError(f"Unknown value {value}")


@app.route("/api/gen", methods=["POST"])
def gen():
    """Instructed generation API
    Usage: 
        curl -X POST 127.0.0.1:8080/api/gen
            -F "text=A photo of"
            -F "seed=0"
    """
    r = request

    request_dict = r.form.to_dict()

    # parse request
    text = request_dict["text"]
    seed = int(request_dict.get("seed"))
    dim = int(request_dict.get("dim"))
    resample = request_dict["resample"]
    upsample = request_dict["upsample"]

    output = gen_functions.gen(
                            text,
                            resample, upsample,
                            seed=seed,
                            dim=dim
                            )

    # return (encode_image(output),)
    encoded_im = encode_image(output)
    files = {"image": encoded_im}
    # return files
    # return np.array(output).tolist()
    return send_file(encode_image(output), mimetype='image/jpeg')




if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, port=7777, threaded=False)
