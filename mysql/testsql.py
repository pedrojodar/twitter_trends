import mysqlwrapper
c= mysqlwrapper.database("DatabaseTest")

c.create_table("tweets", {"timestamp": "VARCHAR(256)","text": "VARCHAR(256)","user": "VARCHAR(256)","sentiment": "VARCHAR(256)"})

c.insert_row(("1000", "text", "@user" , "0"))
