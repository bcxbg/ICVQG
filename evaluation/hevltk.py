from tkinter import *
from PIL import Image,ImageTk
import json

def getdata():
    datas = json.load(open('data3.json'))
    res = json.load(open('results.json'))
    data = datas['data']
    qustions = []
    images = []
    # s1 = []
    # s2 = []
    # s3 = []
    # s4 = []
    for d in data:
        qtext = ""
        for q in d['q']:
            qtext = qtext + q + '\n'
        qustions.append(qtext)
        pilImage = Image.open('imgs/'+ d["i"])
        img = pilImage.resize((300, 300), Image.ANTIALIAS)
        images.append(img)
    #     s1.append(0)
    #     s2.append(0)
    #     s3.append(0)
    #     s4.append(0)
    # res = {
    #     "flu":s1,
    #     "inf":s2,
    #     "sal":s3,
    #     "ric":s4
    # }

    return qustions,images,res

def evl(questions,images,scores):
    global now
    now = 0
    window = Tk()
    window.title("First Window")
    window.geometry("1200x600")

    global iimage
    iimage =images[now]
    tkImage = ImageTk.PhotoImage(image =images[now])
    img = Label(window, image=tkImage)
    img.image = tkImage
    img.place(x=1,y=250)

    lbl = Label(window, text=questions[now], justify='left')
    lbl.place(x=305,y=250)

    text1 = Label(window, text="Fluency")
    text1.grid(column=0, row=0)
    selected1 = IntVar()
    rad11 = Radiobutton(window, text="1", value=1, variable=selected1)
    rad12 = Radiobutton(window, text="2", value=2, variable=selected1)
    rad13 = Radiobutton(window, text="3", value=3, variable=selected1)
    rad14 = Radiobutton(window, text="4", value=4, variable=selected1)
    rad15 = Radiobutton(window, text="5", value=5, variable=selected1)
    rad11.grid(column=0, row=1)
    rad12.grid(column=1, row=1)
    rad13.grid(column=2, row=1)
    rad14.grid(column=3, row=1)
    rad15.grid(column=4, row=1)

    text2 = Label(window, text="Inference")
    text2.grid(column=0, row=2)
    selected2 = IntVar()
    rad21 = Radiobutton(window, text="1", value=1, variable=selected2)
    rad22 = Radiobutton(window, text="2", value=2, variable=selected2)
    rad23 = Radiobutton(window, text="3", value=3, variable=selected2)
    rad24 = Radiobutton(window, text="4", value=4, variable=selected2)
    rad25 = Radiobutton(window, text="5", value=5, variable=selected2)
    rad21.grid(column=0, row=3)
    rad22.grid(column=1, row=3)
    rad23.grid(column=2, row=3)
    rad24.grid(column=3, row=3)
    rad25.grid(column=4, row=3)

    text3 = Label(window, text="Saliency")
    text3.grid(column=0, row=4)
    selected3 = IntVar()
    rad31 = Radiobutton(window, text="1", value=1, variable=selected3)
    rad32 = Radiobutton(window, text="2", value=2, variable=selected3)
    rad33 = Radiobutton(window, text="3", value=3, variable=selected3)
    rad34 = Radiobutton(window, text="4", value=4, variable=selected3)
    rad35 = Radiobutton(window, text="5", value=5, variable=selected3)
    rad31.grid(column=0, row=5)
    rad32.grid(column=1, row=5)
    rad33.grid(column=2, row=5)
    rad34.grid(column=3, row=5)
    rad35.grid(column=4, row=5)

    text4 = Label(window, text="Richness")
    text4.grid(column=0, row=6)
    selected4 = IntVar()
    rad41 = Radiobutton(window, text="1", value=1, variable=selected4)
    rad42 = Radiobutton(window, text="2", value=2, variable=selected4)
    rad43 = Radiobutton(window, text="3", value=3, variable=selected4)
    rad44 = Radiobutton(window, text="4", value=4, variable=selected4)
    rad45 = Radiobutton(window, text="5", value=5, variable=selected4)
    rad41.grid(column=0, row=7)
    rad42.grid(column=1, row=7)
    rad43.grid(column=2, row=7)
    rad44.grid(column=3, row=7)
    rad45.grid(column=4, row=7)
    
    selected1.set(scores['flu'][now])
    selected2.set(scores['inf'][now])
    selected3.set(scores['sal'][now])
    selected4.set(scores['ric'][now])
    
    
    def back():   
        # print('back')
        global now
        # print(now)
        scores['flu'][now] = selected1.get()
        scores['inf'][now] = selected2.get()
        scores['sal'][now] = selected3.get()
        scores['ric'][now] = selected4.get()
        # now = 0
        now = now - 1
        if now >= 0:
            lbl.configure(text=questions[now])
            global iimage
            iimage =images[now]
            tkImage = ImageTk.PhotoImage(image =images[now])
            img.image = tkImage
            img.configure(image=tkImage)

            selected1.set(scores['flu'][now])
            selected2.set(scores['inf'][now])
            selected3.set(scores['sal'][now])
            selected4.set(scores['ric'][now])

        else:now = now+1

    def nnext():
        # print('back')
        global now
        # print(now)
        scores['flu'][now] = selected1.get()
        scores['inf'][now] = selected2.get()
        scores['sal'][now] = selected3.get()
        scores['ric'][now] = selected4.get()
        # now = 0
        now = now + 1
        if now < len(questions):
            lbl.configure(text=questions[now])
            global iimage
            iimage =images[now]
            tkImage = ImageTk.PhotoImage(image =images[now])
            img.image = tkImage
            img.configure(image=tkImage)

            selected1.set(scores['flu'][now])
            selected2.set(scores['inf'][now])
            selected3.set(scores['sal'][now])
            selected4.set(scores['ric'][now])

        else:now = now - 1
    
    def save():
        # print('save')
        global now
        scores['flu'][now] = selected1.get()
        scores['inf'][now] = selected2.get()
        scores['sal'][now] = selected3.get()
        scores['ric'][now] = selected4.get()
        with open('results.json', 'w') as outfile: 
            # print('savefile')
            # print(scores)
            json.dump(scores, outfile)
        # print(now)
        # print(selected1.get())
        # print(selected2.get())
        # print(selected3.get())

    btn = Button(window, text="Back", command=back)
    btn.grid(column=0, row=8)
    btn = Button(window, text="Next", command=nnext)
    btn.grid(column=1, row=8)
    btn = Button(window, text="Save", command=save)
    btn.grid(column=2, row=8)

    


    window.mainloop()



q,i,s = getdata()
evl(q,i,s)