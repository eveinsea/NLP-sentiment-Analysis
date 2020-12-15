from . import app
from .forms import FeatureForm

from flask import render_template
from models.review import ReviewModel


@app.route('/')
def index():
    return render_template('welcome.html')


@app.route('/form/', methods=('GET', 'POST'))
def form():
    myform = FeatureForm()

    if myform.is_submitted():
        line = myform.review_text.data
        review_model = ReviewModel()
        sentiment, hilightwords = review_model.predict(line)

        return render_template('result.html',
                               line=line,
                               highlight_words=hilightwords,
                               sentiment=sentiment
                               )

    return render_template('form.html', form=myform)


@app.route('/result/')
def submit():
    return render_template('result.html')


@app.route('/about')
def about():
    return 'Analyze raw data in JSON with NLP and use Naive Bayes model to predict the new reviews'


@app.route('/author')
def author():
    return 'Xingwen.niu@gmail.com'
