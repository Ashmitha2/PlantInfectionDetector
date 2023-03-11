#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt


# In[ ]:


IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 50hk


# In[4]:


dataset = tf.keras.preprocessing.image_dataset_from_directory(
"PlantVillage",
    shuffle=True,
    image_size= (IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)


# In[5]:


class_names=dataset.class_names
class_names


# In[6]:


len(dataset) #every element in the data set is now a batch of 32 images. 32*68 ~ 2152


# In[7]:


for image_batch,label_batch in dataset.take(1):
    print(image_batch.shape) #batchsize , pizel size , rgb - 3
    print(label_batch.numpy()) #class 0 - early , 1 - late , 2 - healthy


# In[8]:


for image_batch,label_batch in dataset.take(1):
    plt.imshow(image_batch[0].numpy().astype("uint8"))
    plt.title(class_names[label_batch[0]])
    plt.axis("off")


# In[9]:


plt.figure(figsize=(15,15))
for image_batch,label_batch in dataset.take(1):
    for i in range (15):
        ax = plt.subplot(3,5,i+1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[label_batch[i]])
        plt.axis("off")


# In[10]:


#80% - training , 10% validation - performed at end of every epoch , 10% test - on final model after 50 epochs
def get_dataset_partitions(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    ds_size=len(ds)
    
    if shuffle :
        ds=ds.shuffle(shuffle_size , seed=12) #for predictability
        
    train_size=int(ds_size*train_split)
    val_size=int(ds_size*val_split)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size) #take is like [:n](first n) and skip is like[n:](n onwards)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds , val_ds , test_ds


# In[11]:


train_ds , val_ds , test_ds = get_dataset_partitions(dataset)


# In[12]:


train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE) 
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
#cache to keep the image in memory until next image is read i.e., load once read many times
#prefetch loads next set of images when cpu is busy to increase performance


# In[ ]:





# In[13]:


#for scaling
resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE,IMAGE_SIZE), #takes care of dimension during prediction
    layers.experimental.preprocessing.Rescaling(1.0/255)
])


# In[14]:


#augmentaion for handling contrasts and allignments
data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
])


# In[15]:


input_shape = (BATCH_SIZE , IMAGE_SIZE,IMAGE_SIZE , CHANNELS)
n_classes = 3

model = models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32,(3,3), activation='relu', input_shape = input_shape),     #no of filters , filter size , activation layer - relu as it is faster
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)), #no.of.such convolution+relu + pooling layers  is decided using trial and error
    #now flatten it
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(n_classes,activation='softmax'), #softmax normalizes the probability of classes , n_classes =3
])

model.build(input_shape=input_shape)


# 
# 

# In[16]:


model.summary()


# In[17]:


model.compile(
    optimizer='adam', #famous optimizer
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)


# 

# In[37]:


history = model.fit(
    train_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=val_ds
)


# In[38]:


scores = model.evaluate(test_ds)

scores
# In[39]:


scores


# In[40]:


history.params


# In[41]:


history.history.keys()


# In[42]:


history.history['accuracy']


# In[45]:


#for accuracy
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(range(EPOCHS),history.history['accuracy'],label='Training accuracy')
plt.plot(range(EPOCHS),history.history['val_accuracy'],label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and validation accuracy')

#for loss
plt.subplot(1,2,2)
plt.plot(range(EPOCHS),history.history['loss'],label='Training loss')
plt.plot(range(EPOCHS),history.history['val_loss'],label='Validation loss')
plt.legend(loc='lower right')
plt.title('Training and validation loss')
plt.show()


# In[46]:


for images_batch , labels_batch in test_ds.take(1):
    plt.imshow(images_batch[0].numpy().astype('uint8'))


# In[47]:


import numpy as np
for images_batch , labels_batch in test_ds.take(1):
    first_image = images_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0].numpy()
    
    print('First image to predict')
    plt.imshow(first_image)
    print("First image's actual label:",class_names[first_label])
    
    batch_prediction = model.predict(images_batch)
    print("Predicted label:",class_names[np.argmax(batch_prediction[0])]) #prob of 3 classes since it is softmax


# In[48]:


def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array,0)
    
    predictions = model.predict(img_array)
    
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100*(np.max(predictions[0])),2)
    return predicted_class , confidence


# In[49]:


plt.figure(figsize=(15,15))
for images , labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        
        predicted_class , confidence = predict (model,images[i].numpy())
        actual_class = class_names[labels[i]]
        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}, \nConfidence: {confidence}%")
        plt.axis("off")


# In[52]:


model_version=1
model.save("test", save_format='h5')


# In[ ]:




