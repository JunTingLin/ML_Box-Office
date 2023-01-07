import joblib
model_pretrained = joblib.load('Box_Office-LR-20221212.pkl')
import numpy as np
import pandas as pd

data = pd.read_csv("train.csv")

from flask import Flask, request, render_template
app = Flask(__name__)

@app.route("/")
@app.route("/index")
def formPage():
    return render_template('form.html')

@app.route("/submit", methods=['POST'])
def submit():
    if request.method == 'POST':
        print("收到POST請求...")

        form_data = dict(request.form)
        print(type(form_data))

        # 此為測試試印
        print("form_data['budget']"+form_data['budget'])
        print("form_data['original_language']"+form_data['original_language'])
        print("form_data['popularity']:"+form_data['popularity'])

        # budget (text area)
        if form_data['budget'] == "" :
            form_data['budget']= '0' # default value

        #language （下拉式選單
        language_en = ''
        language_zh = ''
        language_other = ''
        if int(form_data['original_language']) == 0:
            language_en = 'selected'
        elif int(form_data['original_language']) == 1:
            language_zh = 'selected'
        elif int(form_data['original_language']) == 2:
            language_other = 'selected'
        else: #default value
            language_other = 'selected'
            form_data['original_language']='0'


        # popularity (text area)
        if form_data['popularity'] == "" :
            form_data['popularity']= '0' # default value

        # runtime (text area)
        if form_data['runtime'] == "" :
            form_data['runtime']= str(data['runtime'].mean()) # default value

        #homepage (radio)
        has_homepage_yes = ''
        has_homepage_no = ''

        if int(form_data['has_homepage'])== 1:
            has_homepage_yes = 'checked'
        elif int(form_data['has_homepage'])== 0:
            has_homepage_no = 'checked'


        #collection (radio)
        collection_yes = ''
        collection_no = ''

        if int(form_data['collection'])== 1:
            collection_yes = 'checked'
        elif int(form_data['collection'])== 0:
            collection_no = 'checked'

        # num_prod_companies (text area)
        if form_data['num_prod_companies'] == "" :
            form_data['num_prod_companies']= '1' # default value

        # num_genres (text area)
        if form_data['num_genres'] == "" :
            form_data['num_genres']= '0' # default value
        else:
            form_data['num_genres']=str(len(set(form_data['num_genres'].split(" "))))

        # num_prod_countries (text area)
        if form_data['num_prod_countries'] == "" :
            form_data['num_prod_countries']= '1' # default value

        # cast_count (text area)
        if form_data['cast_count'] == "" :
            form_data['cast_count']= '1' # default value

        # crew_count (text area)
        if form_data['crew_count'] == "" :
            form_data['crew_count']= '1' # default value

        #release_day （下拉式選單
        release_day_sun = ''
        release_day_mon = ''
        release_day_tue = ''
        release_day_wed = ''
        release_day_thr = ''
        release_day_fri = ''
        release_day_sat = ''

        if int(form_data['release_day']) == 0:
            release_day_sun = 'selected'
        elif int(form_data['release_day']) == 1:
            release_day_mon = 'selected'
        elif int(form_data['release_day']) == 2:
            release_day_tue = 'selected'
        elif int(form_data['release_day']) == 3:
            release_day_wed = 'selected'
        elif int(form_data['release_day']) == 4:
            release_day_thr = 'selected'
        elif int(form_data['release_day']) == 5:
            release_day_fri = 'selected'
        elif int(form_data['release_day']) == 6:
            release_day_sat = 'selected'
        else: #default value
            release_day_sun = 'selected'
            form_data['release_day']='0'

        # Keywords_count (text area)
        if form_data['Keywords_count'] == "" :
            form_data['Keywords_count']= '0' # default value

        #isTaglineNA (radio)
        isTaglineNA_yes = ''
        isTaglineNA_no = ''

        if int(form_data['isTaglineNA'])== 1:
            isTaglineNA_yes = 'checked'
        elif int(form_data['isTaglineNA'])== 0:
            isTaglineNA_no = 'checked'

        # spoken_count (text area)
        if form_data['spoken_count'] == "" :
            form_data['spoken_count']= '1' # default value

        result = model_pretrained.predict([[
        np.log1p(float(form_data['budget'])),
        int(form_data['original_language']),
        float(form_data['popularity']),
        float(form_data['runtime']),
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

        result = np.expm1(result)
        print(f'Result:{result}')
        prediction = round(result[0],1)

        return render_template('form.html',
        # budget = budget,
        language_en = language_en,
        language_zh = language_zh,
        language_other = language_other,
        # popularity = popularity,
        # runtime = runtime,
        has_homepage_yes = has_homepage_yes,
        has_homepage_no = has_homepage_no,
        collection_yes = collection_yes,
        collection_no = collection_no,
        # num_genres = num_genres,
        # num_prod_companies = num_prod_companies,
        # num_prod_countries = num_prod_countries,
        # cast_count = cast_count,
        # crew_count = crew_count,
        # Keywords_count = Keywords_count,
        # spoken_count = spoken_count,
        release_day_sun = release_day_sun,
        release_day_mon = release_day_mon,
        release_day_tue = release_day_tue,
        release_day_wed = release_day_wed,
        release_day_thr = release_day_thr,
        release_day_fri = release_day_fri,
        release_day_sat = release_day_sat,
        isTaglineNA_yes = isTaglineNA_yes,
        isTaglineNA_no = isTaglineNA_no,
        budget = form_data['budget'],
        # original_language = form_data['original_language'],
        popularity = form_data['popularity'],
        runtime = form_data['runtime'],
        # has_homepage = form_data['has_homepage'],
        # collection = form_data['collection'],
        num_genres = form_data['num_genres'],
        num_prod_companies = form_data['num_prod_companies'],
        num_prod_countries = form_data['num_prod_countries'],
        cast_count = form_data['cast_count'],
        crew_count = form_data['crew_count'],
        # release_day = form_data['release_day'],
        Keywords_count = form_data['Keywords_count'],
        # isTaglineNA = form_data['isTaglineNA'],
        spoken_count = form_data['spoken_count'],
        prediction = prediction)


@app.errorhandler(404)  # 傳入錯誤碼作為參數狀態
def error_date(error):  # 接受錯誤作為參數
    return render_template("404.html"), 404
    #返回對應的HTTP狀態碼，和返回404錯誤的html檔

@app.errorhandler(405)
def error_date(error):
    return render_template("405.html"), 405



if __name__ == "__main__":
    app.run()
