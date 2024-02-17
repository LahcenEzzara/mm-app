from flask import Flask, request, render_template
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

app = Flask(__name__)

# Load the dataset and train the model
iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def result():
    # Get the data from the form
    longueur_sepale = float(request.form['longueur_sepale'])
    largeur_sepale = float(request.form['largeur_sepale'])
    longueur_petale = float(request.form['longueur_petale'])
    largeur_petale = float(request.form['largeur_petale'])

    # Make the prediction
    nouvelles_mesures = np.array([[longueur_sepale, largeur_sepale, longueur_petale, largeur_petale]])
    predictions = knn.predict(nouvelles_mesures)

    return render_template('result.html', prediction=predictions[0])


if __name__ == '__main__':
    app.run(host="localhost", debug=True)
