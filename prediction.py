import pickle,sys,os,operator
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
from keras.preprocessing import image as image_utils
from keras.preprocessing import image
from keras import backend as K
import cv2

def image_prediction(test_image):
    try:
        K.clear_session()
        data = []
        img_path = test_image
        testing_img = cv2.imread(img_path)
        cv2.imwrite("../fake_currency/static/detection.png", testing_img)

        model_path = 'cnn.model.h5'
        model = load_model(model_path)
        test_image = load_img(img_path, target_size=(128,128))
        test_image = img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image /= 255
        prediction = model.predict(test_image)
        #print(prediction)
        lb = pickle.load(open('label_transform.pkl_cnn', 'rb'))
        #print(lb)
        result = (lb.inverse_transform(prediction)[0])
        print("result", result)
        K.clear_session()

        return result

    except Exception as e:
        print("Error=", e)
        tb = sys.exc_info()[2]
        print("LINE NO: ", tb.tb_lineno)




#image_prediction()