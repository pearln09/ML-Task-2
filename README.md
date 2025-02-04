# ML-Task-2
Goal: Create a model capable of predicting the lung tidal volume on the basis of factors such as Age, Gender, Height, and some specific symptoms. Seems easy right? Well, the
fun part is that you need to find a dataset for this on your own.
Hint: You might find it difficult to find datasets or find one which is extremely small. There are several techniques that can be employed to combat such situations. Think of
some of them.
Deliverables:

1. A README file with the following details
   
a. The approach you took to find the dataset and any steps you took to ensure that the dataset is sufficient

b. Explain your approach to the problem, any roadblocks you faced, and measures you took to increase accuracy. Be as elaborate in this section as
possible.

c. Instructions on how to run your code. Please note that these instructions should be complete and actually run your code, with no modifications required from our side.

d. Screenshots of your model in action.

e. (Optional)Flowcharts or diagrams explaining the working of the model.

2. A file containing your actual code (this could be either a standalone Python file, a group of Python files, or a Jupyter Notebook)
   
3. The dataset that you used


The very first thing I did was explore for datsets online provided by various medical institutes or national organisations conducting studies relatedto respiratory problems or spirometry in order to search for tidal volume related data. But as has already been mentioned in the PS, the datset had very less to no entries. So I researched more on how all the factors like age, height, gender and various other important symptoms affect the tidal volume of a given person. Using the Heuristic Formula which links all these factors to the tidal volume.

Heuristic Formula

We can define the tidal volume using a simple linear model combining these factors. Here's a sample formula:

Lung Tidal Volume=Base+Age Factor+Gender Factor+Height Factor+Symptom Factor

Where:

Base: 500 mL (baseline tidal volume)

Age Factor:(Age−30)×5/50 (arbitrary adjustment)

Gender Factor: 50×Gender (0 for Female, 1 for Male)

Height Factor: 2×(Height−170) (linear adjustment)

Symptom Factor: −50×(Symptom1+Symptom2+Symptom3)


Using this formula, I created a synthetic dataset which is medically accurate. This synthetic dataset should provide a plausible basis for creating a predictive model. Note that while the data is realistic, it is not based on actual patient data and should not be used for clinical decision-making. 

For the generation of the synthetic dataset, I used the numpy random library. For the age, I took the data for all ages between 18 and 80, gender with a 50-50 probability of either male or female, height between 150 and 200, 3 extra symptoms (asthma, smoking and cough) with different probablilites of all the symptoms for a better/more accurate model and applied the Heuristic formula on the data to create a synthetic dataset. 

After the data preproceesing steps, I applied 3 different models on my dataset to check the accuracy that I get with each model to choose the best model, the 3 different ones being Random Forest Regression, Simple Linear Model and Decision Tree Regression with Random Forest Regression being the best suited one for the dataset. I have also calculated the accuracy level of the predicted tidal volumes and compared them with the actual test set results. 

While solving the problem, the roadblock I faced was in finding the datset online because there was not a lot of observational data available online and for a more accurate model, I needed more entries in my datset. Hence I decided to create a synthetic datset which is medically accurate but not based on actual patient observation data. While a lot of medical institutions keep a record of the patients, they either have less entries or the datset is not publically accesible. So I thought that creating a dataset using a medically accurate formula was a more feasible approach to the problem. And while there are a lot of symptoms, I chose the 3 major symptoms which affect tidal volume for the simplicity of the dataset. I tried generating the synthetic data using GANs like ysynthetic-data but since I am not very well versed with the working of the same, I took a different approach to solve the problem. Also generating synthetic data using GANs, although real, takes a lot of time in training and is very complex as compared to synthetic data generated using numpy library.

To increase the accuracy, I applied 3 different models to my datset and then compared the predictions to choose the one which is most accurate. 

I have solved the PS in Jupyter Notebook because I am familiar with the working.

<img width="944" alt="Model in action 1" src="https://github.com/pearln09/ML-Task-2/assets/157534726/d795ec66-c34b-45af-b21b-042617ab2659">
<img width="958" alt="Model in action 2" src="https://github.com/pearln09/ML-Task-2/assets/157534726/c02e2998-16d9-4d4d-ba6c-65d578ec514b">
<img width="955" alt="Model in action 3" src="https://github.com/pearln09/ML-Task-2/assets/157534726/d9f7e93c-b38c-4ca6-a83d-f8d229bd925f">
<img width="958" alt="Model in action 4" src="https://github.com/pearln09/ML-Task-2/assets/157534726/df506b2b-28b1-4aa4-a956-ed55590edb40">
<img width="959" alt="Model in action 5" src="https://github.com/pearln09/ML-Task-2/assets/157534726/e77b4e2c-0c8f-402d-b820-cab7ce491bb3">
<img width="956" alt="Model in action 6" src="https://github.com/pearln09/ML-Task-2/assets/157534726/6d32f8cd-2050-407a-b932-4b0c322fce40">
<img width="959" alt="Model in action 7" src="https://github.com/pearln09/ML-Task-2/assets/157534726/8e2f6f0a-0828-4438-9c5d-bcff312e37e1">









