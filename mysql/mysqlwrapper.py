import mysql.connector 

class database : 
    def __init__ (self, namedatabase ):
        self.mydb = mysql.connector.connect(
                    host="localhost",
                    user="twittertrends",
                    password="twittertrends")
    
        self.mycursor = self.mydb.cursor()
        self.namedatabase=namedatabase
        self.__create_database(namedatabase)
    def __create_database (self, database):
        command = "CREATE DATABASE IF NOT EXISTS "+database
        self.__execute_command(command)
    def create_table (self, table, definition): 
        self.mydb = mysql.connector.connect(
                    host="localhost",
                    user="twittertrends",
                    password="twittertrends",
                    database=self.namedatabase)
        self.table = table
        self.definition = definition
        command = "CREATE TABLE IF NOT EXISTS "
        command = command + table + "("
        for name  in definition.keys() : 
            command = command + name + " "+definition.get(name)+", "
        command=command[0:len(command)-2]+")"
    
        self.mycursor = self.mydb.cursor()

        self.__execute_command (command)
    def insert_row (self, values):
        command = "INSERT INTO "
        command = command + self.table + "("
        for name in self.definition.keys() : 
            command = command + name +", "
        command = command [0:len(command)-2]+")"
        command = command + " VALUES ("
        for i in range(len(self.definition.keys())):
            command = command + "%s ,"
        command = command [0:len(command)-2]+")"
        print ("COMMAND ", command)
        self.mycursor.execute(command, values)
        self.mydb.commit()
    def __execute_command (self, command):
        self.mycursor.execute(command)
