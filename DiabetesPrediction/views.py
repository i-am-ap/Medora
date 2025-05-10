from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
from django.conf import settings

def home(request):
    return render(request, "index.html")

def predict(request):
    return render(request, "prediction.html")

def result(request):
    csv_path = os.path.join(settings.BASE_DIR,'static', 'data', 'diabetes.csv')
    df = pd.read_csv(csv_path)

    # separating the data and labels
    X = df.drop(columns = 'Outcome', axis=1)
    Y = df['Outcome']
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
