from flask import  Flask , request , render_template
import sqlite3 
import numpy as np
import joblib

Predictor=joblib.load('covid_predict.pkl') 

app=Flask(__name__)


@app.route("/")  
def index():  
    return render_template("start.html")


@app.route("/upload/")
def insert():
    return  render_template("upload.html")


@app.route("/About")
def About():
    return  render_template("About.html")


@app.route("/uploaded",methods = ["POST","GET"])  
def inserted(): 
    
    msg = "msg"  
    flag=1;
    if request.method == "POST":  
        try :
        
            name = request.form["name"]  
            email = request.form["email"]  
            profession = request.form["profession"]  
            age = request.form["age"] 
            gender = request.form["gender"] 
            blood_group = request.form["blood_group"] 
            hypertension = request.form["hypertension"] 
            avg_glucose = request.form["avg_glucose"] 
            bmi = request.form["bmi"]
            Smoking_status = request.form["Smoking_status"] 
            PCR = request.form["PCR"] 
            IFT= request.form["IFT"] 
            asthma = request.form["asthma"] 
            body_temp = request.form["body_temp"] 
            heart = request.form["heart"] 
            new_set=np.array([[age,gender,blood_group,hypertension,avg_glucose,bmi,Smoking_status,PCR,IFT,asthma,body_temp,heart]])
            Predicted_Value=int(Predictor.predict(new_set)[0])
            with sqlite3.connect("Entry_data.db") as con:   
             
                cur = con.cursor()  
                cur.execute("INSERT into Medical_Information(name,email,profession,age,gender,blood_group,hypertension,avg_glucose,bmi,Smoking_status,PCR,IFT,asthma,body_temp,heart_diseases,Covid_status) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(name,email,profession,age,gender,blood_group,hypertension,avg_glucose,bmi,Smoking_status,PCR,IFT,asthma,body_temp,heart,Predicted_Value))  
                
                con.commit() 
                msg = " Your Data  is Successfully Saved"
                #t=cur.execute("Select * from Medical_Info where email="+"'"+email+"'") 
                #t=cur.execute("Select * from Medical_Info where email = Medical_Info.email") 
               
                
                #result =t.fetchall()
                #print(t)
                #print(t.fetchall())
                 
        except:  
            sqlite3.connect("Entry_data.db").rollback()  
            msg = "Sorry !! We Can't Load Your  data "  
            flag=0
        finally: 

            sqlite3.connect("Entry_data.db").close()
            if flag==1 :
                return render_template("success1.html" , msg = msg , data= Predicted_Value)
            else :
                return render_template("success2.html", msg= msg ) 

@app.route("/view")
def view():
    con = sqlite3.connect("Entry_data.db")  
    con.row_factory = sqlite3.Row  
    cur = con.cursor()  
    cur.execute("select * from  Medical_Information")  
    rows = cur.fetchall()  
    return render_template("view.html",rows= rows)  
          


if __name__ == "__main__" :
    app.run(debug=True)






    
