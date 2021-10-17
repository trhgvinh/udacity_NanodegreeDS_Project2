# Disaster Response Pipeline Project

### This app runs with Python 3.6.3, Scikit-learn 0.19.1

### Instructions to run:
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Run the following command in the app's directory to run your web app. Pls note that due to the limitation of file size uploaded to Github the pickle model file in ./models directory is zipped, so if you run the web app directly from this step 2 please unzip the classifier.zip to classifier.pkl before starting the web app.
    `python run.py`
3. Go to http://0.0.0.0:3001/
