from scipy import fftpack
import numpy as np
from PIL import Image, ImageDraw , ImageOps
import matplotlib.pyplot as plt

inputImage = Image.open('input.jpg')
inputImage = ImageOps.grayscale(inputImage)

#convert image to numpy array
inputImage_np = np.array(inputImage)

#fft of image
inputImage_fft = fftpack.fftshift(fftpack.fft2(inputImage))

#Create a band reject filter image
x , y = inputImage_np.shape[0] , inputImage_np.shape[1]      # same x , y of image to the filter

#size of circles
c1_x , c1_y = 100 , 100
c2_x , c2_y = 45 , 45
#create a 2 boxes
box1 = ((x/2)-(c1_x/2),(y/2)-(c1_y/2),(x/2)+(c1_x/2),(y/2)+(c1_y/2))
box2 = ((x/2)-(c2_x/2),(y/2)-(c2_y/2),(x/2)+(c2_x/2),(y/2)+(c2_y/2))
print(box1)
print(box2)
band_pass = Image.new("L" , (inputImage_np.shape[0] , inputImage_np.shape[1]) , color=1)
print(np.shape(band_pass))
drawF = ImageDraw.Draw(band_pass)
drawF.ellipse(box1, fill=0)
drawF.ellipse(box2, fill=1)
band_pass_np = np.array(band_pass)

plt.imshow(band_pass)
plt.show()

#multiply inputImage & Filter
filterdImage = np.multiply(inputImage_fft , band_pass_np)

#inverse fft
ifft = fftpack.ifft2(fftpack.ifftshift(filterdImage))
finalImg = Image.new("L" , (inputImage_np.shape[0] , inputImage_np.shape[1]))
draw = ImageDraw.Draw(finalImg)

for x in range(inputImage_np.shape[0]):
    for y in range(inputImage_np.shape[1]):
        draw.point((x , y) , (int(ifft[y][x])))

#save the image
finalImg.save("outputImage.jpg")
