from flask import Flask, render_template, request, redirect, flash, session, Markup,jsonify
# from flask import  markupsafe.Markup
from pymongo import MongoClient
import requests
import pickle
import io
# import torch
# from torchvision import transforms
from PIL import Image
# from prediction import disease_dic
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing import image


app = Flask(__name__)
app.secret_key = "Miniproject@123"

# MongoDB connection details
mongo_uri = "mongodb+srv://MiniProject2024:Mini123@miniproject.1s62rfp.mongodb.net/?retryWrites=true&w=majority"  
client = MongoClient(mongo_uri)
db = client["farmer_details"]  
users_collection = db["users"]



@app.route("/")
def main():
    return render_template("main.html")
@app.route("/home")
def home():
   return render_template("home.html")



#SIGNUP ROUTE...


@app.route("/signup", methods=["GET", "POST"])
# @app.route("/signup", methods=["POST"])
def signup():
    # return render_template("signup.html")
 if request.method == "POST":
    first_name = request.form["first_name"]
    last_name = request.form["last_name"]
    phone_No = request.form["phone_No"]
    password = request.form["password"]
    Location = request.form["Location"]
    Crop = request.form["Crop"]

    # Create a new user document
    user = {
        "first_name": first_name,
        "last_name": last_name,
        "phone_No":phone_No,
        "password": password,
        "Location": Location,
        "Crop": Crop
    }

    session['Location'] = Location

    users_collection.insert_one(user)

    flash("User signed up successfully. Please log in.")
    return redirect("/login")
 return render_template("signup.html")


#LOGIN ROUTE....


@app.route("/login", methods=["GET", "POST"])  
def login():
    if request.method == "POST":
        phone_No = request.form["phone_No"]
        password = request.form["password"]

        user = users_collection.find_one({"phone_No": phone_No, "password": password})

        if user:
            session["username"] = user["phone_No"]
            return redirect("/services")

        flash("Invalid username or password")
        return render_template("login.html")

    return render_template("login.html")


#SERVICES ROUTE...

@app.route("/services")
def services():
   return render_template("services.html")


# WEATHER MODEL...
#The user can select any city from Telangana

def fetch_weather_data(city):
    api_key = "b1f9041f1812f89b3ac75f2a9ac9d0a8"
    # city= "Adilabad"
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}"
    response = requests.get(url)
    data = response.json()
    return data

@app.route("/weather")
def weather():
    city = session.get('Location')
    # city = "Adilabad"
    weather_data = fetch_weather_data(city)
    forecast_entries = weather_data.get('list', [])
    
    # Extract data for the next 4 days
    next_4_days_forecast = []
    current_date = None

    for entry in forecast_entries:
        date, time = entry['dt_txt'].split()
        if date != current_date:
            current_date = date
            temperature = entry['main']['temp']
            weather_description = entry['weather'][0]['description']
            next_4_days_forecast.append({
                'date': date,
                'temperature': temperature,
                'weather_description': weather_description
            })

        if len(next_4_days_forecast) >= 4:
            break

    return render_template('weather.html', forecast_entries=next_4_days_forecast,Location=city)



#DISEASE PREDICTION.....


# disease_classes = ['Apple___Apple_scab',
#                    'Apple___Black_rot',
#                    'Apple___Cedar_apple_rust',
#                    'Apple___healthy',
#                    'Blueberry___healthy',
#                    'Cherry_(including_sour)___Powdery_mildew',
#                    'Cherry_(including_sour)___healthy',
#                    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
#                    'Corn_(maize)___Common_rust_',
#                    'Corn_(maize)___Northern_Leaf_Blight',
#                    'Corn_(maize)___healthy',
#                    'Grape___Black_rot',
#                    'Grape___Esca_(Black_Measles)',
#                    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
#                    'Grape___healthy',
#                    'Orange___Haunglongbing_(Citrus_greening)',
#                    'Peach___Bacterial_spot',
#                    'Peach___healthy',
#                    'Pepper,_bell___Bacterial_spot',
#                    'Pepper,_bell___healthy',
#                    'Potato___Early_blight',
#                    'Potato___Late_blight',
#                    'Potato___healthy',
#                    'Raspberry___healthy',
#                    'Soybean___healthy',
#                    'Squash___Powdery_mildew',
#                    'Strawberry___Leaf_scorch',
#                    'Strawberry___healthy',
#                    'Tomato___Bacterial_spot',
#                    'Tomato___Early_blight',
#                    'Tomato___Late_blight',
#                    'Tomato___Leaf_Mold',
#                    'Tomato___Septoria_leaf_spot',
#                    'Tomato___Spider_mites Two-spotted_spider_mite',
#                    'Tomato___Target_Spot',
#                    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
#                    'Tomato___Tomato_mosaic_virus',
#                    'Tomato___healthy']

# disease_model_path = r'C:\Users\NHI643\Desktop\mini_project\app\plant_disease.pth'
# disease_model = ResNet9(3, len(disease_classes))
# disease_model.load_state_dict(torch.load(
#     disease_model_path, map_location=torch.device('cpu')))

# def predict_image(img, model=disease_model):
#     transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.ToTensor(),
#     ])
#     image = Image.open(io.BytesIO(img))
#     img_t = transform(image)
#     img_u = torch.unsqueeze(img_t, 0)

#     # Get predictions from model
#     yb = model(img_u)
#     # Pick index with highest probability
#     _, preds = torch.max(yb, dim=1)
#     prediction = disease_classes[preds[0].item()]
#     # Retrieve the class label
#     return prediction




# @app.route('/cropdisease' ,methods=["GET", "POST"])
# def disease():

#     if request.method == 'POST':
#         if 'file' not in request.files:
#          return redirect(request.url)
#         file = request.files.get('file')
#         if not file:
#             return render_template('cropdisease.html')
#         try:
#             img = file.read()

#             prediction = predict_image(img)

#             prediction = Markup(str(disease_dic[prediction]))
#             print(prediction)
#             return render_template('cropdisease.html', prediction=prediction)
        
#         except:
#             pass
#     return render_template('cropdisease.html')



#CROP RECOMMENDATION.....

crop_recommendation_model_path = r'C:\Users\NHI643\Desktop\mini_project\app\RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))

def init_session():
    session['N'] = None
    session['P'] = None
    session['K'] = None
    session['temperature'] = None
    session['humidity'] = None
    session['pH'] = None
    session['rainfall'] = None



@app.route('/crop_recommendation', methods=['GET', 'POST'])
def recommendation():
    if request.method == 'POST':
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        pH = float(request.form['pH'])
        rainfall = float(request.form['rainfall'])

        # Store the input values in the session
        session['N'] = N
        session['P'] = P
        session['K'] = K
        session['temperature'] = temperature
        session['humidity'] = humidity
        session['pH'] = pH
        session['rainfall'] = rainfall

        # Make predictions using your crop recommendation model
        input_data = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
        prediction = crop_recommendation_model.predict(input_data)[0]

        return render_template('crop_recommendation.html', prediction=prediction)

    if session.get('N') is None:
        init_session()

    return render_template('crop_recommendation.html', prediction=None)


#FERTILIZER RECOMMENDATION....
#The values are stored in session and getting render directly from crop recommendation model

model1 = joblib.load(r'C:\Users\NHI643\Desktop\mini_project\app\Fertilizer.sav')

@app.route('/fertilizer_recommendation', methods=['GET', 'POST'])
def fertilizer():
    if request.method == 'POST':
        N = session['N']
        P = session['P']
        K = session['K']
        temperature = session['temperature']
        humidity = session['humidity']
        # pH = session['pH']
        # rainfall = session['rainfall']
        Moisture = float(request.form['Moisture'])
        soil_type = request.form['soil_type']
        crop_type = request.form['crop_type']
        input_data = np.array([[N, P, K, temperature, humidity, Moisture, soil_type, crop_type]])
        prediction = model1.predict(input_data)[0]


        return render_template('fertilizer_recommendation.html', prediction=prediction)

    return render_template('fertilizer_recommendation.html', prediction=None)


#Disease Prediction....

model2 = tf.keras.models.load_model(r'C:\Users\NHI643\Desktop\mini_project\app\rice.hdf5')

# def preprocess_image(image_path):
#     img = image.load_img(image_path, target_size=(180, 180))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_preprocessed = img_array / 255.0
#     return img_preprocessed


def preprocess_image(image_data1):
    img1 = Image.open(io.BytesIO(image_data1))
    img1 = img1.resize((180, 180))  
    img_array1 = image.img_to_array(img1)
    img_array1 = np.expand_dims(img_array1, axis=0)
    img_preprocessed1 = img_array1 / 255.0
    return img_preprocessed1


# def predict(image_data):
#     try:
#         img = preprocess_image(image_data)
#         prediction = model.predict(img)
#         # return {'prediction': 'Replace this with the actual prediction'}
#         return prediction

#     except Exception as e:
#         return {'error': str(e)}
    
# @app.route('/cropdisease', methods=['GET','POST'])
# def predict_endpoint():
#     try:
#         image_data = request.files['image'].read()
#         prediction = predict(image_data)

#         return jsonify(prediction)

#     except Exception as e:
#         return jsonify({'error': str(e)})




# def predict(image_data):
#     try:
#         img = preprocess_image(image_data)
#         prediction = model.predict(img)
#         predicted_class_index = np.argmax(prediction)
#         # return prediction.tolist()  # Convert to a list for JSON serialization
#         predicted_class_label = disease_labels[predicted_class_index]
#         # return  predicted_class_index
#         return predicted_class_label

disease_labels = ["Bacterial disease", "Smut", "Brown Spot"]

# Your image preprocessing and prediction code here
def predict(image_data1):
    try:
        img1 = preprocess_image(image_data1)
        prediction1 = model2.predict(img1)
        predicted_class_index1 = np.argmax(prediction1)
        predicted_class_label1 = disease_labels[predicted_class_index1]

        return predicted_class_label1

    except Exception as e:
        return {'error': str(e)}

# @app.route('/cropdisease', methods=['GET','POST'])
# def predict_endpoint():
#     try:
#         # Ensure that the request contains a file named 'image'
#         if 'image' not in request.files:
#             return jsonify({'error': 'No image file provided'})

#         image_file = request.files['image']

#         # Check if the file has a valid extension (you can modify this as needed)
#         allowed_extensions = {'jpg', 'jpeg', 'png', 'gif'}
#         if '.' not in image_file.filename or image_file.filename.split('.')[-1].lower() not in allowed_extensions:
#             return jsonify({'error': 'Invalid file format. Supported formats: jpg, jpeg, png, gif'})

#         image_data = image_file.read()
#         prediction = predict(image_data)

#         return jsonify({'prediction': prediction})

#     except Exception as e:
#         # Print the error to the console for debugging
#         print(f"An error occurred: {str(e)}")
#         return jsonify({'error': 'An error occurred while processing the request'})


# @app.route('/cropdisease', methods=['GET', 'POST'])
# def predict_endpoint():
#     if request.method == 'POST':
#         try:
#             if 'image' not in request.files:
#                 return jsonify({'error': 'No image file provided'})

#             image_file = request.files['image']
#             allowed_extensions = {'jpg', 'jpeg', 'png', 'gif'}
            
#             if ('.' not in image_file.filename or 
#                 image_file.filename.split('.')[-1].lower() not in allowed_extensions):
#                 return jsonify({'error': 'Invalid file format. Supported formats: jpg, jpeg, png, gif'})

#             image_data = image_file.read()
#             prediction = predict(image_data)

#             # Render the HTML template with the prediction result
#             return render_template('cropdisease.html', prediction=prediction)

#         except Exception as e:
#             # Print the error to the console for debugging
#             print(f"An error occurred: {str(e)}")
#             return jsonify({'error': 'An error occurred while processing the request'})

#     # If it's a GET request or no image has been uploaded yet, render the form
#     return render_template('cropdisease.html', prediction=None)


@app.route('/cropdisease', methods=['GET', 'POST'])
def predict_endpoint():
    if request.method == 'POST':
        try:
            if 'image' not in request.files:
                return jsonify({'error': 'No image file provided'})

            image_file = request.files['image']
            allowed_extensions = {'jpg', 'jpeg', 'png', 'gif'}
            
            if ('.' not in image_file.filename or 
                image_file.filename.split('.')[-1].lower() not in allowed_extensions):
                return jsonify({'error': 'Invalid file format. Supported formats: jpg, jpeg, png, gif'})

            image_data1 = image_file.read() 
            prediction1 = predict(image_data1)

   
            return render_template('cropdisease.html', prediction1=prediction1)

        except Exception as e:

            print(f"An error occurred: {str(e)}")
            return jsonify({'error': 'An error occurred while processing the request'})


    return render_template('cropdisease.html', prediction1=None)




#PEST DETECTION....
#The image should be in size 244/244 and only jpg, png is accepted
#The pest and disease model got overrided due to the same image processing functions are used for both...
model3 = tf.keras.models.load_model(r'C:\Users\NHI643\Desktop\mini_project\app\pest.hdf5')


def preprocess_image1(image_path):
    img = Image.open(io.BytesIO(image_path))
    # img = image.load_img(image_path, target_size=(224, 224))
    img = img.resize((224, 224)) 
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_preprocessed = img_array / 255.0
    return img_preprocessed

pest_labels = ["aphids", "armyworm", "beetle", "bollworm", "grasshopper","mites","mosquito", "sawfly","stem_borer"]

def predict1(image_data):
    try:
        img = preprocess_image1(image_data)
        prediction = model3.predict(img)
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = pest_labels[predicted_class_index]

        return predicted_class_label
        return predicted_class_index

    except Exception as e:
        return {'error': str(e)}

@app.route('/pest_detection', methods=['GET', 'POST'])
def predict_pest():
    if request.method == 'POST':
        try:
            if 'image' not in request.files:
                return jsonify({'error': 'No image file provided'})

            image_file = request.files['image']
            allowed_extensions = {'jpg', 'jpeg', 'png', 'gif'}
            
            if ('.' not in image_file.filename or 
                image_file.filename.split('.')[-1].lower() not in allowed_extensions):
                return jsonify({'error': 'Invalid file format. Supported formats: jpg, jpeg, png, gif'})

            image_data = image_file.read() 
            prediction = predict1(image_data)

   
            return render_template('pest_detection.html', prediction=prediction)

        except Exception as e:

            print(f"An error occurred: {str(e)}")
            return jsonify({'error': 'An error occurred while processing the request'})


    return render_template('pest_detection.html', prediction=None)








# @app.route('/pest_detection', methods=['POST'])
# def predict_pest():
#     try:
#         # Get the image data from the frontend
#         image_data = request.files['image'].read()

#         # Use your pest detection model to make predictions on the image_data
#         img = preprocess_image(io.BytesIO(image_data))
#         prediction = model.predict(img)

#         # You can format the prediction result as needed
#         # In this example, we return the predicted class index as an integer
#         predicted_class_index = np.argmax(prediction)

#         return render_template('pest_detection.html', predicted_class_index=predicted_class_index)

#     except Exception as e:
#         return render_template('error.html', error_message=str(e))




#LOGOUT ROUTE....

@app.route("/logout")
def logout():
    return render_template("home.html")


@app.route("/suggest")
def suggest():
    return render_template("suggest.html")

if __name__ == "__main__":
    app.run(debug=True)
   
