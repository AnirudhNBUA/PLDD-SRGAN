from flask import Flask,render_template,request
import pickle
import numpy as np
from PIL import Image
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
import torch
from torch.utils.data import DataLoader
from dataset import *
from srgan_model import Generator, Discriminator



app = Flask(__name__)

model = load_model('SRCNN_model.h5')
with open('SRCNN_label_transform.pkl', 'rb') as f:
    label_binarizer = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(img_feat = 3, n_feats = 64, kernel_size = 3, num_block = 16)
generator.load_state_dict(torch.load('trained_model_6400.pt', map_location=torch.device('cpu')))
generator = generator.to(device)
generator.eval()

@app.route('/')
def hello_world():
    return render_template('index.html')


safe = ['Pepper__bell___healthy','Potato___healthy','Tomato_healthy']

@app.route('/detect',methods=['POST','GET'])
def detect():

    image_file = request.files['img']
    image = Image.open(image_file)
    image.save('leaf_image/image.jpg', format='JPEG')
    dataset = testOnly_data(LR_path = 'leaf_image', in_memory = False, transform = None)
    loader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = 0)
    
    result = None 
    counter = 1
    with torch.no_grad():
        for i, te_data in enumerate(loader):
            lr = te_data['LR'].to(device)
            output, _ = generator(lr)
            output = output[0].cpu().numpy()
            output = (output + 1.0) / 2.0
            output = output.transpose(1,2,0)
            result = Image.fromarray((output * 255.0).astype(np.uint8))
            result.save('generated_leaf_images/{}.jpg'.format(counter),format='JPEG')
            counter+=1
    img = result
    img = img.resize((256, 256),resample=Image.BICUBIC)
    img = np.array(img) / 255.0
    npp_image = np.expand_dims(img, axis=0)
    result=model.predict(npp_image)
    itemindex = np.where(result==np.max(result))
    max_prob = np.max(result)
    
    if label_binarizer.classes_[itemindex[1][0]] in safe:
        return render_template('index.html',detect = 'Your crop is safe.', 
                               crop = 'Crop: {}'.format(label_binarizer.classes_[itemindex[1][0]]),prob= 'Probability: {}'.format(str("{:.3f}".format(np.max(result)))))
    else:
        return render_template('index.html',detect = 'Your crop is in danger.', 
        crop = 'Disease: {}'.format(label_binarizer.classes_[itemindex[1][0]]),prob= 'Probability: {}'.format(str("{:.3f}".format(np.max(result)))))

    folder_path = "leaf_image"
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

if __name__ == '__main__':
      app.run(debug=True)