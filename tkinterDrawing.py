from PIL import Image,ImageDraw
import PIL
from tkinter import *
import numpy as np
from nnPredict import prepare_NN,Predict

width = 400
height = 400
cor = 8

def Cogness():
    # predict = r.randint(0,9)
    global image1
    global draw
    filename = "./images/mnistPic.png"
    image1.save(filename)
    cv.delete('all')
    cv.create_text(200,200,text = Predict(model,filename),font = "Times 100")
    image1 = PIL.Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image1)

def Clearing():
    cv.delete('all')

def Paint(event):

    x1, y1=event.x-8, event.y-8
    x2,y2 = event.x+8, event.y+8
    cv.create_oval(x1,y1,x2,y2,fill="#000000")
    draw.ellipse((x1,y1,x2,y2),'black','black')

model=prepare_NN()
root = Tk()

cv = Canvas(root,width=width,height=height,bg = 'white')

image1=PIL.Image.new('RGB',(width,height),(255,255,255))
draw = ImageDraw.Draw(image1)
cv.pack(expand = YES,fill = BOTH)
cv.bind("<B1-Motion>",Paint)
button = Button(text = 'Read',command = Cogness)
bClear = Button(text = 'Clear', command = Clearing)
button.pack()
bClear.pack()
root.mainloop()

