import re
from tkinter.tix import Tree
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.core.files.base import ContentFile
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import base64
import io
import numpy as np
import json
from io import BytesIO

from .MoveNet_Refined import processImageTensor

# Create your views here.
def cam(request):
    return render(request, "core/camerafeed.html")

@csrf_exempt
def processImage(request):

    data=json.loads(request.body)["imageData"]

    f = open("out.png","w")
    f.write(data)
    f.close()

    format, imgstr = data.split(';base64,')

    if True: 
        ext = format.split('/')[-1] 

        base64_decoded = base64.b64decode(imgstr)


        image = Image.open(io.BytesIO(base64_decoded))

        data = processImageTensor(np.array(image))

        # data = data.decode("utf-8")
        # data = "data:image/jpeg;base64,"+data

        pil_img = Image.fromarray(data)
        buff = BytesIO()
        pil_img.save(buff, format="JPEG")
        new_image_string = "data:image/jpeg;base64,"+base64.b64encode(buff.getvalue()).decode("utf-8")

    
    # image_np = np.array(image)




    # #image = Image.open(io.BytesIO(base64_decoded))
    # imgplot = plt.imshow(image_np)
    # plt.show()

    #img = mpimg.imread()
    #img_plot = plt.imshow(img)
    #plt.show()
    return JsonResponse({"imgData":new_image_string})