__author__ = 'Керя'

# from sentiment_classifier import SentimentClassifier
from sklearn.externals import joblib
from flask import Flask, render_template, request
app = Flask(__name__)


class SentimentClassifier(object):
    def __init__(self):
        self.model = joblib.load("./model/pipeline.pkl")
        self.classes_dict = {0: "негативный", 1: "позитивный", -1: "сконцентрируйтесь и повторите"}

    @staticmethod
    def get_probability_words(probability):
        if probability < 0.55:
            return "Нейстральный или не уверенный"
        if probability < 0.7:
            return "Предположительно"
        if probability > 0.95:
            return "Определенно"
        else:
            return ""

    def predict_text(self, text):
        try:
            return self.model.predict(text)[0], self.model.decision_function(text)[0].max()
        except:
            print("prediction1 error")
            return -1, 0.8

    def predict_list(self, list_of_texts):
        try:
            return self.model.predict(list_of_texts), self.model.decision_function(list_of_texts)
        except:
            print('prediction2 error')
            return None

    def get_prediction_message(self, text):
        prediction = self.predict_text([text])
        class_prediction = prediction[0]
        prediction_probability = prediction[1]
        return self.get_probability_words(prediction_probability) + " " + self.classes_dict[class_prediction]


@app.route("/demo", methods=["POST", "GET"])
def index_page(text="", prediction_message=""):
    if request.method == "POST":
        text = request.form["text"]
        prediction_message = classifier.get_prediction_message(text)

    return render_template('hello.html', text=text, prediction_message=prediction_message)

print("Preparing classifier")
classifier = SentimentClassifier()
print("Classifier is ready")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=False)
