import uuid
import os
import datetime

class AbsModel():
    def __init__(self):
        self.uid = str(uuid.uuid1())
        os.mkdir("./runs/" + self.uid)
        logfile = open("./runs/" + self.uid + "/run_log.txt", "w+")
        logfile.close()

    def log(self, msg):
        logfile = open("./runs/" + self.uid + "/run_log.txt", "w+")
        logfile.write(str(datetime.datetime.now()) + ": " + msg + " \n")
        print(str(datetime.datetime.now()) + ": " + msg + " \n")
        logfile.close()

