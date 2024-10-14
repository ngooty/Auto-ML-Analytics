from mareana_ml_code.mareana_machine_learning.utils.utils import PostgresDB

p_host = 'localhost'
p_port = 5432
db = 'mdh'
ssh = True
ssh_user = 'ubuntu'
ssh_host = '10.2.194.192'
ssh_passwd = 'Mzreana@190$'
psql_user= 'postgres'
psql_password= 'postgrespass'
psql_host='localhost'
params = {
             'database': 'mdh',
             'user': 'postgres',
             'password': 'postgrespass',
             'host': 'localhost',
             'port': ''
             }


db = PostgresDB(pgres_host=p_host, pgres_port=p_port, db=db, 
                            ssh=ssh, ssh_user=ssh_user, ssh_host=ssh_host, 
                            ssh_passwd=ssh_passwd,psql_user=psql_user,psql_pass=psql_password,params=params)
#initiates a connection to the PostgreSQL database. In this instance we use ssh and must specify our ssh credentials.
#Below query will return a pandas Data Frame
#result=pgres.query('mdh','select * from work_view_master;')
#print(result)
pgres=db.Connect()
cur=pgres.cursor()
table_create='''
create table IF NOT EXISTS public.work_sklearn_library(module char(50), submodule char(50),
Parameter1  char(100),
Parameter2 char(100),
Parameter3 char(100),
Parameter4 char(100),
Parameter5 char(100),
Parameter6 char(100),
Parameter7 char(100),
Parameter8 char(100),
Parameter9 char(100),
Parameter10 char(100),
Parameter11 char(100),
Parameter12 char(100),
Parameter13 char(100),
Parameter14 char(100),
Parameter15 char(100),
Parameter16 char(100),
Parameter17 char(100),
Parameter18 char(100),
Parameter19 char(100),
Parameter20 char(100),
Parameter21 char(100),
Parameter22 char(100),
Parameter23 char(100),
Parameter24 char(100),
Parameter25 char(100),
Parameter26 char(100));
'''
#cur.execute(table_create)
cur.execute('select count(*) from public.plat_sklearn_library;')
result=cur.fetchall()
print(result)
pgres.commit()
#open the csv file using python standard file I/O
#copy file into the table just created 
#f = open(r'/Users/Nag/Downloads/Data.csv','r')

#cur.copy_from(f, 'work_sklearn_library', sep=',')
#f.close()
# df=pd.read_excel(r'/Users/Nag/Downloads/Data.xlsx')
# df.to_sql('work_sklearn_library','public')
#pgres.commit()
