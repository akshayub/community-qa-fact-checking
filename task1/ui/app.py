from pickle import load
from flask import Flask, render_template, request
import pandas as pd
from keras.preprocessing import sequence
import tensorflow as tf

max_len=100
model = load(open('electronics_amazon_qa','rb'))
graph = tf.get_default_graph()
tok = load(open('tokenizer','rb'))

app = Flask(__name__)
 
@app.route("/", methods=['GET', 'POST'])
def server():
    if request.method == 'GET':
        return render_template('site.html')
    else:
        global graph
        with graph.as_default():
            q = pd.Series(request.form['question'])
            seq= tok.texts_to_sequences(q)
            seq_matrix = sequence.pad_sequences(seq, maxlen=max_len)
            ans = model.predict(seq_matrix)
            res = ''
            if (int(round(ans[0][0])) == 0):
                res = 'open-ended'
            else:
                res = 'yes-no'
            return render_template('result.html', value=res, question=request.form['question'])
 
if __name__ == "__main__":
    app.run(debug=True)
