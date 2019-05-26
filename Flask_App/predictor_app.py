import flask
from flask import request,render_template
from predictor_api import generate_quran, generate_bible, load_doc, seed_text_generator, generate_torah
import tensorflow as tf
import keras

# Initialize the app

app = flask.Flask(__name__)



## An example of routing:
## If they go to the page "/" (this means a GET request
## to the page http://127.0.0.1:5000/), return a simple
## page that says the site is up!
#@app.route("/")
#def hello():
#    return flask.send_file("static/html/index.html")

     
@app.route('/')
def home_page():
    return render_template('home_page.html')

@app.route('/quran', methods=['GET','POST'])
def get_quran():

#     if request.method == 'GET':
        
        # Loading seed corpus
        clean_quran = load_doc('/Users/tcbon/Desktop/Coding/Metis/Bootcamp/Project_4/Flask_App/static/models/clean_quran.txt').split('\n')

        #Generating text
        seed_text = seed_text_generator(clean_quran)
        generated = generate_quran(seed_text)
        return render_template("generated_quran.html",generated = generated)
    
    
@app.route('/bible', methods=['GET','POST'])
def get_bible():

#     if request.method == 'GET':
        
        # Loading seed corpus
        clean_new_test = load_doc('/Users/tcbon/Desktop/Coding/Metis/Bootcamp/Project_4/Flask_App/static/models/clean_new_testament.txt').split('\n')

        #Generating text
        seed_text = seed_text_generator(clean_new_test)
        generated = generate_bible(seed_text)
        return render_template("generated_bible.html",generated = generated)
    
@app.route('/torah', methods=['GET','POST'])
def get_torah():

#     if request.method == 'GET':
        
        # Loading seed corpus
        clean_torah = load_doc('/Users/tcbon/Desktop/Coding/Metis/Bootcamp/Project_4/Flask_App/static/models/clean_torah.txt').split('\n')

        #Generating text
        seed_text = seed_text_generator(clean_torah)
        generated = generate_torah(seed_text)
        return render_template("generated_torah.html",generated = generated)
    

#     else:
#         return 'loading'

# Start the server, continuously listen to requests.
# We'll have a running web app!

if __name__=="__main__":
#     For local development:
    app.run(debug=False,threaded = False)
#     For public web serving:
#    app.run(host='0.0.0.0')
#     app.run()
