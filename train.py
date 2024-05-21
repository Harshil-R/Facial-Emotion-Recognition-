import numpy as np
import pandas as pd
import os
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import torch


train_fold = r"E:\AI\Ai emotional detection\AI Project\images\train"
test_fold = r"E:\AI\Ai emotional detection\AI Project\images\validation"


def dataframe(dir):
    labels = []
    image_paths = []
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir,label)):
            image_paths.append(os.path.join(dir,label,imagename))
            labels.append(label)
        print(label, "loading completed")
    return image_paths,labels

train = pd.DataFrame()
train['image'], train['label'] = dataframe(train_fold)


test = pd.DataFrame()
test['image'], test['label'] = dataframe(test_fold)


print(train)
print('====================================================================')
print(test)


def feature_extractor(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, grayscale=True)
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features),48,48,1)
    return features

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_features = feature_extractor(train['image']) 


test_features = feature_extractor(test['image']) 

x_train = train_features/255.0
x_test = test_features/255.0


le = LabelEncoder()
le.fit(train['label'])


y_train = le.transform(train['label'])
y_test = le.transform(test['label'])

y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)


model = Sequential()
model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(7, activation='softmax'))



model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = 'accuracy')

if torch.cuda.is_available():
    print("CUDA (GPU) is available")
    device = torch.device('cuda')
else:
    print()

from keras.callbacks import ModelCheckpoint

checkpoint_filepath = 'model_checkpoints/checkpoint.h5'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)

model.fit(x = x_train, y = y_train, batch_size = 128, epochs = 100, validation_data = (x_test, y_test), callbacks = [model_checkpoint_callback])

json_model = model.to_json()
with open("emotiondetector.json",'w') as json_file:
    json_file.write(json_model)
model.save("emotiondetector.h5")