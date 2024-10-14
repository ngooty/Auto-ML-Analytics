from mareana_ml_code.mareana_machine_learning.utils.database import PostGres

db=PostGres()

cur=db.connect()

print(db.params['host'])

cur.execute('select * from public.work_view_master')
result=cur.fetchall()
#print(result)