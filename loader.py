import numpy as np
import pandas as pd
import os

train_fold = r"C:\Users\dhair\OneDrive\Documents\Dhairya Bhatt\Subjects\Sem 5\AI\AI Project\images\train"
test_fold = r"C:\Users\dhair\OneDrive\Documents\Dhairya Bhatt\Subjects\Sem 5\AI\AI Project\images\validation"

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
test['label'], test['iamge'] = dataframe(test_fold)

