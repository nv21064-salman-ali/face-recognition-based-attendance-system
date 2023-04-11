import cv2
import os
from flask import Flask,request,render_template
from datetime import date
from datetime import datetime
from datetime import timedelta
import mysql.connector
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

#### Defining Flask App
app = Flask(__name__)


### Connection to the MySQL Database
def connect_to_database():
    connection = mysql.connector.connect(
        host="localhost",
        user="python_user",
        password="NVTC@1234",
        database="attendance_db"
    )
    return connection 

#### Saving Date today in 2 different formats
date_today = datetime.now().strftime("%Y-%m-%d")
datetoday2 = date.today().strftime("%d-%B-%Y")


#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
try:
    cap = cv2.VideoCapture(1)
except:
    cap = cv2.VideoCapture(0)


#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{date_today}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{date_today}.csv','w') as f:
        f.write('Name,Roll,Time')


#### get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


#### extract the face from an image
def extract_faces(img):
    if np.any(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    else:
        return []

#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl')


#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{date_today}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names,rolls,times,l


#### Add Attendance of a specific user

def create_attendance_table_if_not_exists(connection):
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            Name VARCHAR(255),
            Roll INT,
            Time TIME,
            Date DATE
        )
    """)
    connection.commit()

def load_attendance_data(date_today, connection):
    df = pd.read_sql_query(f"SELECT * FROM attendance WHERE Date = '{date_today}'", connection)
    return df

def save_attendance_data_to_database(username, userid, current_time, date_today, connection):
    cursor = connection.cursor()
    query = "INSERT INTO attendance (Name, Roll, Time, Date) VALUES (%s, %s, %s, %s)"
    values = (username, userid, current_time, date_today)
    cursor.execute(query, values)
    connection.commit()

def save_attendance_data_to_csv(date_today, connection):
    df = pd.read_sql_query(f"SELECT * FROM attendance WHERE Date = '{date_today}'", connection)
    df.to_csv(f'Attendance/Attendance-{date_today}.csv', index=False)

def was_present_within_duration(df, userid, current_time, duration_minutes):
    current_datetime = datetime.strptime(current_time, "%H:%M:%S")
    time_threshold = current_datetime - timedelta(minutes=duration_minutes)

    user_rows = df[df['Roll'] == userid]

    if user_rows.empty:
        return False

    last_present_time = user_rows['Time'].max()
    
    # Convert Timedelta object to datetime.time object
    last_present_time = (datetime.min + last_present_time).time()

    last_present_datetime = datetime.combine(datetime.today(), last_present_time)
    current_datetime = datetime.combine(datetime.today(), current_datetime.time())

    return last_present_datetime > time_threshold



def add_attendance(name):
    username, userid = name.split('_')
    userid = int(userid)
    current_time = datetime.now().strftime("%H:%M:%S")
    date_today = datetime.now().strftime("%Y-%m-%d")

    connection = connect_to_database()
    create_attendance_table_if_not_exists(connection)
    df = load_attendance_data(date_today, connection)

    if not was_present_within_duration(df, userid, current_time, 10):
        save_attendance_data_to_database(username, userid, current_time, date_today, connection)
        save_attendance_data_to_csv(date_today, connection)

    connection.close()

################## ROUTING FUNCTIONS #########################

#### Our main page
@app.route('/')
def home():
    names,rolls,times,l = extract_attendance()    
    return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2) 


@app.route('/start', methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    cap = cv2.VideoCapture(0)
    ret = True
    while ret:
        ret, frame = cap.read()
        if ret:
            faces = extract_faces(frame)
            if faces is not None and len(faces) > 0:
                (x, y, w, h) = faces[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
                face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
                identified_person = identify_face(face.reshape(1, -1))[0]
                add_attendance(identified_person)
                cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            cv2.imshow('Attendance', frame)
            if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == 27:  # quit on 'q' or Esc key
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()    
    return render_template('takeAttendance.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


#### This function will run when we add a new user
@app.route('/add',methods=['GET','POST'])
def add():
    if request.method == 'GET':
        return render_template('addUser.html', totalreg=totalreg(), datetoday2=datetoday2)
    
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i,j = 0,0
    while 1:
        _,frame = cap.read()
        faces = extract_faces(frame)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame,f'Images Captured: {i}/50',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
            if j%10==0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name,frame[y:y+h,x:x+w])
                i+=1
            j+=1
        if j==500:
            break
        cv2.imshow('Adding new User',frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names,rolls,times,l = extract_attendance()    
    return render_template('addUser.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2) 


#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)
