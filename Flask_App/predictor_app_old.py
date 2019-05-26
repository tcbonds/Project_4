import flask
from flask import request
from predictor_api import generate_seq
import tensorflow as tf
import keras



# Initialize the app

app = flask.Flask(__name__)
#init = tf.initialize_all_variables()

init =  tf.global_variables_initializer()




#sess = tf.Session()
graph = tf.get_default_graph()  
keras.backend.get_session().run(tf.global_variables_initializer())
#
## An example of routing:
## If they go to the page "/" (this means a GET request
## to the page http://127.0.0.1:5000/), return a simple
## page that says the site is up!
#@app.route("/")
#def hello():
#    return flask.send_file("static/html/index.html")



@app.route("/", methods=["POST", "GET"])
def predict():
    # request.args contains all the arguments passed by our form
    # comes built in with flask. It is a dictionary of the form
    # "form name (as set in template)" (key): "string in the textbox" (value)

    #sess.run(init)
    with graph.as_default():
        print(request.args)
        if request.args:
            seed_text, generated = generate_seq(request.args['input'])
            return flask.render_template('predictor.html', generated=generated,x_input=seed_text)
        else:
            seed_text, generated = generate_seq('test')
            return flask.render_template('predictor.html', generated=generated,x_input=seed_text)


# Start the server, continuously listen to requests.
# We'll have a running web app!

if __name__=="__main__":
#     For local development:
    app.run(debug=True)
#     For public web serving:
#    app.run(host='0.0.0.0')
    app.run()
