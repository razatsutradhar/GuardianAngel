import mysql.connector
import time

while True:
    amberdb = mysql.connector.connect(host='sql5.freesqldatabase.com', user='sql5473936', passwd='43NA67P5Aw', database='sql5473936')
    mycurser = amberdb.cursor()
    mycurser.execute('SELECT License FROM `Amber Alerts`')


    file = open("./cache.txt", "w")
    for i in mycurser:
        file.write(i[0]+'\n')
    file.close()
    print("updated " + str(time.time()))
    time.sleep(60)