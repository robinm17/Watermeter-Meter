from PIL import Image, ImageOps, ImageEnhance, ImageFilter

rotation=-52 ; #CCW
crop_box=(292,430,474,473) 
crop_digits=[(8,5,43,40),(45,6,80,41),(83,6,120,41),(123,6,160,41),(163,6,195,41)] 
digit_names_root="digit_"
image_mode=".jpg"
Digits=[]

# gets the image
image = Image.open("capture.jpg")
print(image.mode)
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

# region.convert("1")
region.show()
# #print(region.format, region.size, region.mode)

# # crops now each digit
# for each in crop_digits:
#     Digits.append(region.crop(each))

# # for index in range(0,len(Digits)):
# #      Digits[index]= Digits[index].filter(ImageFilter.EDGE_ENHANCE)

# #let's try transformations...
# for index in range(0,len(Digits)):
#     enhancer = ImageEnhance.Contrast(Digits[index])
#     Digits[index]= enhancer.enhance(70)

# for index in range(0,len(Digits)):
#      Digits[index]= Digits[index].filter(ImageFilter.SHARPEN)
    

# # saves each image
# for index in range(0,len(Digits)):
#     name=digit_names_root+str(index)+image_mode ;
#     Digits[index].save(name)
    
# # shows the image
# #im.show()
# #region.show()
# for each in Digits:
#     each.show()
# #im.show()