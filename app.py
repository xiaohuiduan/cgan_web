import imageio
from flask import Flask, request, render_template
import uuid
from cgan.ACGan import ACGan
import os

app = Flask(__name__)

basepath = os.path.dirname(__file__)  # 当前文件所在路径
model_file_path = "cgan/7000_GENERATOR.hdf5"


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/random')
def random():
    gan = ACGan(model_file_path)
    batch_size = 32
    imgs = gan.createRandom(batch_size)
    return_imgs = []
    for img in imgs:
        file_name = str(uuid.uuid4()) + ".png"
        upload_path = os.path.join(basepath, 'static/images', file_name)
        imageio.imwrite(upload_path, img)
        return_imgs.append(file_name)

    return render_template('random.html',
                           data=return_imgs)


@app.route('/special')
def special():
    return render_template('special.html')


@app.route('/special_2')
def special_2():
    # 可以通过 request 的 args 属性来获取参数
    hair = int(request.args.get("hair"))
    eye = int(request.args.get("eye"))
    batch_size = 32
    gan = ACGan(model_file_path)
    imgs = gan.create_special(hair, eye, batch_size)
    return_imgs = []
    for img in imgs:
        file_name = str(uuid.uuid4()) + ".png"
        upload_path = os.path.join(basepath, 'static/images', file_name)
        imageio.imwrite(upload_path, img)
        return_imgs.append(file_name)

    return render_template('special_2.html',
                           data=return_imgs, eye=eye, hair=hair)


if __name__ == '__main__':
    app.run(host="127.0.0.1", threaded=False)
