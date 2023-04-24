from tkinter import *
from PIL import ImageGrab, ImageOps
import numpy as np
import io
import threading

tf = None

def importTensorflow():
    global tf
    import tensorflow
    tf = tensorflow
    predictBtn.config(state=NORMAL)

threading.Thread(target=importTensorflow).start()

class MyModel:
    def __init__(self) -> None:
        self.IM_H, self.IM_W = 28, 28
        self.CLASS_NAMES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        self.MODEL_PATH = "model.tflite"
    
    def predict(self, image) -> dict:
        image = tf.keras.utils.load_img(image, target_size=(self.IM_H, self.IM_W))
        imageArray = tf.keras.utils.img_to_array(image)
        imageArray = tf.expand_dims(imageArray, 0)
        interpreter = tf.lite.Interpreter(model_path=self.MODEL_PATH)
        interpreter.get_signature_list()
        liteClassifier = interpreter.get_signature_runner("serving_default")
        litePredictions = liteClassifier(sequential_1_input=imageArray)["outputs"]
        liteScore = tf.nn.softmax(litePredictions)
        className = self.CLASS_NAMES[np.argmax(liteScore)]
        confidence = 100 * np.max(liteScore)
        return {"class": className, "confidence": confidence}

app = Tk()

WIN_WIDTH = 600
WIN_HEIGHT = 400

app.resizable(0, 0)
app.geometry(f"{WIN_WIDTH}x{WIN_HEIGHT}")

CANVAS_W = 400
CANVAS_H = 400

canvas = Canvas(app, width=CANVAS_W, height=CANVAS_H, bg="white")
canvas.place(relx=0, rely=0)

penSizeValue = StringVar(value=20)

def drawOnCanvas(event):
    penSize = int(penSizeValue.get())
    mouseLocationX, mouseLocationY = event.x, event.y
    canvas.create_oval(
        mouseLocationX-penSize,
        mouseLocationY-penSize,
        mouseLocationX+penSize,
        mouseLocationY+penSize,
        fill="black"
    )

canvas.bind("<B1-Motion>", drawOnCanvas)

def getCanvasDrawing():
    return ImageGrab.grab(bbox=(
            canvas.winfo_rootx(),
            canvas.winfo_rooty(),
            canvas.winfo_rootx() + canvas.winfo_width(),
            canvas.winfo_rooty() + canvas.winfo_height()
        )).resize((28, 28))

def predictCanvasDrawing():
    canvasDrawing = getCanvasDrawing()
    negativeImage = ImageOps.invert(canvasDrawing)
    imageBytes = io.BytesIO()
    negativeImage.save(imageBytes, format="JPEG")
    imageBytes.seek(0)
    negativeImage.save("last-img.jpg", format="JPEG")
    prediction = MyModel().predict(imageBytes)
    del negativeImage
    imageBytes.close()
    del imageBytes
    className = f"""Class: {prediction.get("class")}"""
    confidencePercent = f"""Confidence: {round(prediction.get("confidence"), 2)}%"""
    classLabel.config(text=className)
    confidenceLabel.config(text=confidencePercent)

def clearCanvas():
    canvas.delete("all")
    classLabel.config(text="Class: None")
    confidenceLabel.config(text="Confidence: None")

DEFAULT_FONT = ("arial", 15)

btnsFrame = LabelFrame(app, bd=0, padx=10, pady=10)
btnsFrame.place(x=CANVAS_W, y=0, height=WIN_HEIGHT,width=WIN_WIDTH-CANVAS_W)

classLabel = Label(
    btnsFrame,
    text="Class: None",
    font=DEFAULT_FONT,
    anchor=W
)
confidenceLabel = Label(
    btnsFrame,
    text="Confidence: None",
    font=DEFAULT_FONT,
    anchor=W
)
predictBtn = Button(
    btnsFrame,
    text="Predict",
    command=predictCanvasDrawing,
    font=DEFAULT_FONT,
    state=DISABLED
)
clearCanvasBtn = Button(
    btnsFrame,
    text="Clear",
    command=clearCanvas,
    font=DEFAULT_FONT,
)
penSizeSpinbox = Spinbox(
    btnsFrame,
    from_=10,
    to=30,
    font=DEFAULT_FONT,
    textvariable=penSizeValue
)

classLabel.place(relx=0.5, rely=0.1, anchor=N, relwidth=1)
confidenceLabel.place(relx=0.5, rely=0.3, anchor=N, relwidth=1)
predictBtn.place(relx=0.5, rely=0.5, anchor=CENTER, relwidth=1)
clearCanvasBtn.place(relx=0.5, rely=0.7, anchor=S, relwidth=1)
penSizeSpinbox.place(relx=0.5, rely=0.9, anchor=S, relwidth=1)

app.mainloop()