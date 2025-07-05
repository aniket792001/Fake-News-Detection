from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

vectorization = joblib.load('models/vectorizer.pkl')
LR = joblib.load('models/lr_model.pkl')
DT = joblib.load('models/dt_model.pkl')
RF = joblib.load('models/rf_model.pkl')


def wordopt(text):
    import re, string
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', b'', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def output_lable(n):
    if n==0:
        return "Fake News"
    elif n==1:
        return "Not A Fake News"

def manual_testing(news):
    import pandas as pd
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test['text'] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)
    return {
        "Logistic Regression": output_lable(pred_LR[0]),
        "Decision Tree": output_lable(pred_DT[0]),
        "Random Forest": output_lable(pred_RF[0])
    }



@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/api/data', methods=['POST'])
def get_data():
    data = request.get_json()
    news = data.get('text', '')
    result = manual_testing(news)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)