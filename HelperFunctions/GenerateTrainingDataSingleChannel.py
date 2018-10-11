import sys
sys.path.insert(0, "HelperFunctions")
import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
from tifffile import imread
from skimage import transform
from Normalize import normalizeFloat, normalizeMinMax, Path, save_tiff_imagej_compatible
from keras.preprocessing.image import ImageDataGenerator
from skimage import transform


def AugmentImages(ImageDirectory, MaskDirectory):
    X = sorted(glob(ImageDirectory + '*.tif'))
    Y = sorted(glob(MaskDirectory + '*.tif'))

    targetdirX = ImageDirectory
    targetdirY = MaskDirectory

    assert all(Path(x).name==Path(y).name for x,y in zip(X,Y))
    Names = []

    axes = 'XY'

    listX = list(map(imread,X))
    listY = list(map(imread,Y))

    WIDTH = 0
    HEIGHT = 0
    for i in range(len(listX)):
      if listX[i].shape[0] > WIDTH:
        WIDTH = listX[i].shape[0]
      if listX[i].shape[1] > HEIGHT:
        HEIGHT = listX[i].shape[1]
    print('Zero padding all images to the max size found: ',WIDTH, HEIGHT)



    images=[]
    for fn in listX:

      blankX = np.zeros([WIDTH, HEIGHT], dtype = float)

      blankX[:fn.shape[0], :fn.shape[1]] = fn
    
    
      images.append(blankX)
    images = np.array(images)

    masks=[]
    for fn in listY:
      blankY = np.zeros([WIDTH, HEIGHT], dtype = int)
      blankY[:fn.shape[0], :fn.shape[1]] = fn
    
    
      masks.append(blankY)
     
    masks = np.array(masks)

    rankfourX = np.expand_dims(images, axis = -1)
    rankfourY = np.expand_dims(masks, axis = -1)



    # traning data is augmented
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.5,
        zoom_range=(0.9, 1.1),
        horizontal_flip=False,
        vertical_flip=False, 
        fill_mode='constant',
        cval=0)




    train_generatorX = train_datagen.flow(rankfourX, batch_size= rankfourX.shape[0], seed=1337)
    train_generatorY=  train_datagen.flow(rankfourY,  batch_size= rankfourY.shape[0], seed=1337)

    newX = train_generatorX.next()
    newY = train_generatorY.next()


    for i in range(rankfourX.shape[0]):
      resultX = rankfourX[i,:,:,0]
       
     
      base = os.path.split(X[i])[-1]
      
      Filename =  base
      
      save_tiff_imagej_compatible((targetdirX + Filename ) , resultX, axes)
        
    for i in range(rankfourY.shape[0]):
      resultY = rankfourY[i,:,:,0]
      base = os.path.split(Y[i])[-1]

      Filename = base
     
      save_tiff_imagej_compatible((targetdirY + Filename ) , resultY, axes)  

    for i in range(newX.shape[0]):
      resultX = newX[i,:,:,0]
       
     
      base = os.path.split(X[i])[-1]
      
      Filename = "new" + base
      
      save_tiff_imagej_compatible((targetdirX + Filename ) , resultX, axes)
        
    for i in range(newY.shape[0]):
      resultY = newY[i,:,:,0]
      base = os.path.split(Y[i])[-1]

      Filename = "new" + base
     
      save_tiff_imagej_compatible((targetdirY + Filename ) , resultY, axes)    


def LoadTrainingData(ImageDirectory, MaskDirectory): 
    
    X = sorted(glob(ImageDirectory + '*.tif'))
    Y = sorted(glob(MaskDirectory + '*.tif'))
    listX = list(map(imread,X))
    listY = list(map(imread,Y))
    WIDTH = 0
    HEIGHT = 0
    for i in range(len(listX)):
      if listX[i].shape[0] > WIDTH:
        WIDTH = listX[i].shape[0]
      if listX[i].shape[1] > HEIGHT:
        HEIGHT = listX[i].shape[1]
    images=[]
    for fn in listX:

           blankX = np.zeros([WIDTH, HEIGHT], dtype = float)

           blankX[:fn.shape[0], :fn.shape[1]] = fn
    
    
           images.append(blankX)
    images = np.array(images)

    masks=[]
    for fn in listY:
          blankY = np.zeros([WIDTH, HEIGHT], dtype = int)
          blankY[:fn.shape[0], :fn.shape[1]] = fn
    
    
          masks.append(blankY)
    
    masks = np.array(masks)

    rankfourX = np.expand_dims(images, axis = -1)
    rankfourY = np.expand_dims(masks, axis = -1) 
    
    print(rankfourX.shape)
    return rankfourX, rankfourY


def sample_2Dtiles(image, annotation, tile_shape=(32,32), samples=10):
    sample_im=[]
    sample_mask=[]
    for i in range(samples):
        x = np.random.randint(0,image.shape[0]-tile_shape[0]-1)
        y = np.random.randint(0,image.shape[1]-tile_shape[1]-1)
        sample_im.append(   image[x:x+tile_shape[0],y:y+tile_shape[1]])
        sample_mask.append( annotation[x:x+tile_shape[0],y:y+tile_shape[1]])
    return np.array(sample_im), np.array(sample_mask)


def getTrainingTiles(images, masks,PATCH_HEIGHT, PATCH_WIDTH, num_samples = 10):
    Xin = []
    Yin = []
    for im, mask in zip(images, masks):
      x,y = sample_2Dtiles(im, mask, tile_shape=(PATCH_HEIGHT, PATCH_WIDTH), samples=num_samples)
      
      Xin.append(x)
      Yin.append(y)
      
    Xin = np.array(Xin)
    Xin = Xin.reshape((Xin.shape[0]*Xin.shape[1], Xin.shape[2], Xin.shape[3], Xin.shape[4]))
    Yin = np.array(Yin)
    Yin = Yin.reshape((Yin.shape[0]*Yin.shape[1], Yin.shape[2], Yin.shape[3], Yin.shape[4]))
    return Xin, Yin   
    
    
    