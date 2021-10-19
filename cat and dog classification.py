#download cats and dogs classification data from this soure www.kaggle.com/c/dogs-vs-cats/data
import os, shutil
original_dataset_dir = '/home/nouman/Downloads/dogs-vs-cats/train'
base_dir = '/home/nouman/Downloads/cats_and_dogs_small'
#directories to save paths of folders containing cats and dogs images
os.makedirs(base_dir,exist_ok=True)
train_dir = os.path.join(base_dir, 'train')
os.makedirs(train_dir,exist_ok=True)
validation_dir = os.path.join(base_dir, 'validation')
os.makedirs(validation_dir,exist_ok=True)
test_dir = os.path.join(base_dir, 'test')
os.makedirs(test_dir,exist_ok=True)
#directory containg training images for cats class
train_cats_dir = os.path.join(train_dir, 'cats')
os.makedirs(train_cats_dir,exist_ok=True)
#directory containg training images for dogs class
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.makedirs(train_dogs_dir,exist_ok=True)
#directory containg validation images for cats class
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.makedirs(validation_cats_dir,exist_ok=True)
#directory containg validation images for dogs class
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.makedirs(validation_dogs_dir,exist_ok=True)
#directory containg testing images for cats class
test_cats_dir = os.path.join(test_dir, 'cats')
os.makedirs(test_cats_dir,exist_ok=True)
#directory containg testing images for dogs class
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.makedirs(test_dogs_dir,exist_ok=True)
#copying 1000 cats images in training directory
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)
#copying 500 cats images in validation directory
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)
#copying 500 cats images in test directory
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)
#copying 1000 dogs images in train directory
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)
#copying 500 dogs images in validation directory
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)
#copying 500 dogs images in test directory
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)
#building the model
from keras import layers
from keras import models
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
from keras import optimizers
model.compile(loss='binary_crossentropy', 
              optimizer=optimizers.RMSprop(lr=1e-4), 
              metrics=['acc'])
#generating data from directory
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255) 
train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150, 150),
                                                    batch_size=20,class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir,target_size=(150, 150),
                                                        batch_size=20,class_mode='binary')
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break
#training the model
history = model.fit(train_generator,steps_per_epoch=100,
                    epochs=10,
                    validation_data=validation_generator,
                    validation_steps=50)
model.save('cats_and_dogs_small_1.h5')
#plotting the results
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
