# from django .shortcuts import render
# from django.http import HttpResponse
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn import svm
# from sklearn.metrics import accuracy_score



# def home(request):
#     return render(request, "index.html")

# def predict(request):
#     return render(request, "prediction.html")

# def result(request):
#     # loading the diabetes dataset to a pandas DataFrame
#     diabetes_dataset = pd.read_csv('D:\\archive\\diabetes.csv') 


#     # separating the data and labels
#     X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
#     Y = diabetes_dataset['Outcome']

#     X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)

#     classifier = svm.SVC(kernel='linear')
#     #training the support vector Machine Classifier
#     classifier.fit(X_train, Y_train)

#     try:
#         val1 = float(request.GET['n1'])
#         val2 = float(request.GET['n2'])
#         val3 = float(request.GET['n3'])
#         val4 = float(request.GET['n4'])
#         val5 = float(request.GET['n5'])
#         val6 = float(request.GET['n6'])
#         val7 = float(request.GET['n7'])
#         val8 = float(request.GET['n8'])

#     except (ValueError, KeyError) as e:
#         return(request,"prediction.html",{"result2":"Invalid input. Please enter valid input for all fields."})

#     pred = classifier.predict([[val1,val2,val3,val4,val5,val6,val7,val8]])

#     result1 = ""
#     if pred == [1]:
#         result1 = "The person is diabetic"
#     else:
#         result1 = "The person is not diabetic"

#     return render(request, "prediction.html",{"result2":result1})






# from django.shortcuts import render
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC

# # Function to load the diabetes dataset and train the SVM model
# def load_diabetes_dataset_and_train_svm():
#     # Load the diabetes dataset from CSV (adjust the path as per your setup)
#     diabetes_dataset = pd.read_csv('D:\\archive\\diabetes.csv')

#     # Separate features (X) and target (Y)
#     X = diabetes_dataset.drop(columns='Outcome', axis=1)
#     Y = diabetes_dataset['Outcome']

#     # Split data into training and testing sets
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

#     # Initialize SVM classifier (linear kernel)
#     classifier = SVC(kernel='linear')

#     # Train the SVM classifier
#     classifier.fit(X_train, Y_train)

#     # Return the trained classifier
#     return classifier

# # Global variable to hold trained SVM classifier
# svm_classifier = load_diabetes_dataset_and_train_svm()

# # View function for the homepage
# def home(request):
#     return render(request, "index.html")

# # View function for the prediction page
# def predict(request):
#     return render(request, "prediction.html")

# # View function to process the prediction and display result
# def result(request):
#     global svm_classifier  # Access the global SVM classifier

#     try:
#         # Get input values from the request
#         val1 = float(request.GET['n1'])
#         val2 = float(request.GET['n2'])
#         val3 = float(request.GET['n3'])
#         val4 = float(request.GET['n4'])
#         val5 = float(request.GET['n5'])
#         val6 = float(request.GET['n6'])
#         val7 = float(request.GET['n7'])
#         val8 = float(request.GET['n8'])

#     except (ValueError, KeyError) as e:
#         # Handle invalid input (e.g., non-numeric values)
#         return render(request, "prediction.html", {"result2": "Invalid input. Please enter valid input for all fields."})

#     # Predict using the SVM classifier
#     pred = svm_classifier.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])

#     # Determine the result message based on the prediction
#     if pred == 1:
#         result_message = "The person is diabetic"
#     else:
#         result_message = "The person is not diabetic"

#     # Render the prediction.html template with the result message
#     return render(request, "prediction.html", {"result2": result_message})


from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from django.shortcuts import render

def home(request):
    return render(request, "index.html")

def predict(request):
    return render(request, "prediction.html")

def result(request):
    diabetes_dataset = pd.read_csv(r"C:\Users\aryan\Downloads\diabetes.csv")

    # separating the data and labels
    X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
    Y = diabetes_dataset['Outcome']
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)

    model = LogisticRegression()
    model.fit(X_train, Y_train)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    # pred = model.predict([[val1,val2,val3,val4,val5,val6,val7,val8]])
    
    # result1 = ""
    # if pred == 1:
    #     result1 = "The person is diabetic"
    # else:
    #     result1 = "The person is not diabetic"

    

    # return render(request,  "prediction.html", {"result2":result1})

    try:
        # Collect and validate all 8 inputs
        values = []
        for i in range(1, 9):
            val = request.GET.get(f'n{i}', '')
            if val == '':
                return render(request, "prediction.html", {"result2": f"Please enter all values (missing n{i})"})
            values.append(float(val))

        # Predict
        pred = model.predict([values])
        result1 = "The person is diabetic" if pred[0] == 1 else "The person is not diabetic"
        return render(request, "prediction.html", {"result2": result1})

    except ValueError:
        return render(request, "prediction.html", {"result2": "Invalid input: Please enter numeric values only."})