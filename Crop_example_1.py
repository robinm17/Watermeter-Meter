from PIL import Image, ImageOps, ImageEnhance, ImageFilter

rotation=0 ; #CCW
crop_box=(493,281,744,328) 
crop_digits=[(5,6,30,43),(60,6,85,83),(83,6,120,43),(123,6,160,43),(163,6,195,43)] 
digit_names_root="digit_"
image_mode=".jpg"
Digits=[]

# gets the image
image = Image.open("waterm.jpg")
# print(image.mode)
# image = image1.convert("L")
# print(image.mode)
# for x in range(image.width):
#     for y in range(image.height):
#         # print(image.getpixel((x,y)))
#         if image.getpixel((x,y)) < 80:
#             image.putpixel((x,y),0)
#         else:
#             image.putpixel((x,y),255)

            
# image = image.convert("1")
# im = ImageOps.grayscale(image)
# enhancer = ImageEnhance.Contrast(im)

# factor = 100
# im = enhancer.enhance(factor)
#image data
#print(im.format, im.size, im.mode)

# corrects rotation
im=image.rotate(rotation, resample=0, expand=0, center=None, translate=None, fillcolor=None)



# im.show()
# im.save("test.jpg")
# crops ROI
# im = im.filter(ImageFilter.SHARPEN)
# # im = im.filter(ImageFilter.FIND_EDGES)

region = im.crop(crop_box)
region = region.filter(ImageFilter.SMOOTH)

# image = region.convert("L")
# print(image.mode)
# for x in range(image.width):
#     for y in range(image.height):
#         # print(image.getpixel((x,y)))
#         if image.getpixel((x,y)) < 85:
#             image.putpixel((x,y),0)
#         else:
#             image.putpixel((x,y),255)

# region.convert("1")
image.show()
region.show()
# print(region.format, region.size, region.mode)

# crops now each digit
# for each in crop_digits:
#     Digits.append(region.crop(each))

# for index in range(0,len(Digits)):
#      Digits[index]= Digits[index].filter(ImageFilter.EDGE_ENHANCE)

# #let's try transformations...
# for index in range(0,len(Digits)):
#     enhancer = ImageEnhance.Contrast(Digits[index])
#     Digits[index]= enhancer.enhance(70)

# for index in range(0,len(Digits)):
#      Digits[index]= Digits[index].filter(ImageFilter.SMOOTH)

# for index in range(0,len(Digits)):
#     Digits[index]= Digits[index].filter(ImageFilter.EDGE_ENHANCE)
    

# saves each image
# for index in range(0,len(Digits)):
#     name=digit_names_root+str(index)+image_mode ;
#     Digits[index].save(name)
    
# # shows the image
# #im.show()
# #region.show()
for each in Digits:
    each.show()
# #im.show()