import joblib
model_pretrained = joblib.load('Box_Office-LR-20221212.pkl')
import numpy as np

from flask import Flask, request, render_template
app = Flask(__name__)

@app.route("/")
def formPage():
    return render_template('form.html')

@app.route("/submit", methods=['POST'])
def submit():
    if request.method == 'POST':
        form_data = request.form
        result = model_pretrained.predict([[
        int(form_data['budget']),
        int(form_data['original_language']),
        int(form_data['popularity']),
        int(form_data['runtime']),
        int(form_data['has_homepage']),
        int(form_data['collection']),
        int(form_data['num_genres']),
        int(form_data['num_prod_companies']),
        int(form_data['num_prod_countries']),
        int(form_data['cast_count']),
        int(form_data['crew_count']),
        int(form_data['release_day']),
        int(form_data['Keywords_count']),
        int(form_data['isTaglineNA']),
        int(form_data['spoken_count'])
        ]])
        
        print(f'Result:{result}')
        prediction = result

        return render_template('form.html', 
        budget = form_data['budget'],
        original_language = form_data['original_language'],
        popularity = form_data['popularity'],
        runtime = form_data['runtime'],
        has_homepage = form_data['has_homepage'],
        collection = form_data['collection'],
        num_genres = form_data['num_genres'],
        num_prod_companies = form_data['num_prod_companies'],
        num_prod_countries = form_data['num_prod_countries'],
        cast_count = form_data['cast_count'],
        crew_count = form_data['crew_count'],
        release_day = form_data['release_day'],
        Keywords_count = form_data['Keywords_count'],
        isTaglineNA = form_data['isTaglineNA'],
        spoken_count = form_data['spoken_count'],
        prediction = prediction)

if __name__ == "__main__":
    app.run()