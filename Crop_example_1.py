from PIL import Image, ImageOps, ImageEnhance, ImageFilter

rotation=0 ; #CCW

#[(max left, max top, max right, max bottom)] pixels
crop_box=(493,281,744,328) 

#[(max left, max top, max right, max bottom)] pixels
crop_digits=[(5,6,30,43),(60,6,85,83),(83,6,120,43),(123,6,160,43),(163,6,195,43)] 

digit_names_root="digit_"
image_mode=".jpg"
Digits=[]

# gets the image
image = Image.open("waterm.jpg")

# corrects rotation
im=image.rotate(rotation, resample=0, expand=0, center=None, translate=None, fillcolor=None)

# crops the image to the defined crop area
region = im.crop(crop_box)

#smoothen the picture
region = region.filter(ImageFilter.SMOOTH)

# show the image
region.show()

# print(region.format, region.size, region.mode)

# crops now each digit
for each in crop_digits:
    Digits.append(region.crop(each))

# #optional edge enhancement
# for index in range(0,len(Digits)):
#      Digits[index]= Digits[index].filter(ImageFilter.EDGE_ENHANCE)

# #let's try transformations...
# for index in range(0,len(Digits)):
#     enhancer = ImageEnhance.Contrast(Digits[index])
#     Digits[index]= enhancer.enhance(70)

# saves each image
for index in range(0,len(Digits)):
    name=digit_names_root+str(index)+image_mode ;
    Digits[index].save(name)
    
# shows the image
for each in Digits:
    each.show()