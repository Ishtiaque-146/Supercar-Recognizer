from fastai.vision.all import *
import gradio as gr

# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

car_labels = (
    'Aston Martin Valkyrie', 
    'Bugatti Chiron', 
    'Ferrari Daytona SP3', 
    'Ferrari Enzo', 
    'Ferrari LaFerrari', 
    'Ferrari SF90 Stradale', 
    'Ford GT', 
    'Hennessey Venom F5 Roadster', 
    'Koenigsegg Gemera', 
    'Koenigsegg Jesko', 
    'Lamborghini Aventador', 
    'Lamborghini Revuelto', 
    'McLaren P1', 
    'McLaren Senna', 
    'Mercedes-AMG One', 
    'Pagani Zonda', 
    'Porsche 911 GT3 RS', 
    'Porsche 918 Spyder', 
    'Rimac Nevera', 
    'Zenvo Aurora'
)

model = load_learner('car-recognizer-v2.pkl')

def recognize_image(image):
    pred, idx, probs = model.predict(image)
    return dict(zip(car_labels, map(float, probs)))

image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label(num_top_classes=5)
examples = [
    'unknown_00.jpg',
    'unknown_01.jpg',
    'unknown_02.jpg',
    'unknown_03.jpg'
    ]

iface = gr.Interface(fn=recognize_image, inputs=image, outputs=label, examples=examples)
iface.launch(inline=False, share=True)