import sqlite3

from numpy import unique  
  
con = sqlite3.connect("Entry_data.db")  
print("Database opened successfully")  
  
con.execute("create table Medical_Information(Sino INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,email TEXT UNIQUE NOT NULL, profession  TEXT NOT NULL,age integer NOt NULL , gender integer NOT NULL ,blood_group integer  NOT NULL,hypertension integer NOT NULL,avg_glucose DECIMAL(3,2) NOt NULL,bmi DECIMAL(2,2) NOT NULL,Smoking_status integer  NOT NULL,PCR integer  NOT NULL,IFT integer  NOT NULL,asthma integer  NOT NULL,body_temp DECIMAL(2,2) NOT  NULL,heart_diseases integer  NOT NULL,Covid_status integer  NOT NULL  )") 
print("Table created successfully")  
  
con.close()  