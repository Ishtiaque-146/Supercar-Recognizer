# Supercar-Recognizer
An Image classification Model from Data collection, cleaning, model training as well as deployment and API integration

The final app can classify 20 different types of Supercars of different regions of the world.

The types of cars are,
1. McLaren F1
2. Ferrari Enzo
3. Ferrari LaFerrari
4. McLaren P1
5. Porsche 918 Spyder
6. Ferrari SF90 Stradale
7. Aston Martin Valkyrie
8. Rimac Nevera
9. Mercedes-AMG One
10. Koenigsegg Jesko
11. Ferrari Daytona SP3
12. Hennessey Venom F5 Roadster
13. Koenigsegg Gemera
14. Lamborghini Revuelto
15. Porsche 911 GT3 RS
16.  Zenvo Aurora
17. Pagani Zonda
18. Ford GT
19. Bugatti Chiron
20. Lamborghini Aventador

[Click Here](https://ishtiaque-146.github.io/Supercar-Recognizer/) to Visit Web aplication

# Dataset Preparation
### Data Collection: 
The data was collected by downloading from DuckDuckGo search using their term name
### DataLoader: 
Used fastai DataBlock API to set up the DataLoader where Resnet34 was used and was fine-tuned multiple times.
### Data Augmentation: 
fastai provides default data augmentation which operates in GPU.
Details can be found in 'notebooks/Data_prep.ipynb'

# Training and Data Cleaning
### Training: 
Fine-tuned a resnet50 model for 5 epochs initially and then with 3 and then 2 epochs and got up to 98.4% accuracy.

### Data Cleaning: 
This part took the highest time. Since the collected data were from the browser, there were many noises. Also, there were images that contained Animations, adds as well as there were garbage data that contains irrelevant subjects. The was cleaned and 
updated by using fastai library `ImageClassifierCleaner`. The data were cleaned each time after training and finetuning till the last time which was the final iteration of the model.

# Model Deployment
The final model was deployed in HuggingFace Spaces using Gradio App. The implementation can be found in `deployment` folder or [here](https://huggingface.co/spaces/Ishtiaque146/supercar-recognizer).

![API](https://github.com/Ishtiaque-146/Supercar-Recognizer/assets/169515556/18ac4b0d-7d5a-4b0a-b69e-aa9ca6c293b4)

# API integration with GitHub Pages
The deployed model API is integrated [here](https://ishtiaque-146.github.io/Supercar-Recognizer/) in GitHub Pages Website. Implementation and other details can be found in `docs` folder.

