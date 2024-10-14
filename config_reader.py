import os


def get_url():
    str_url =os.getenv('AIRFLOW__WEBSERVER__BASE_URL')
    str_url = str_url.rstrip('airflow')
    return str_url

def get_neo_url():
    str_url =os.getenv('AIRFLOW__NEO4J__BASE_URL')
    return str_url

def get_neo_username():
    str_url =os.getenv('AIRFLOW__NEO4J__USER')
    return str_url

def get_neo_password():
    str_url =os.getenv('AIRFLOW__NEO4J__PASSWD')
    return str_url

def get_db_conn():
    db_str =os.getenv('AIRFLOW__CORE__SQL_ALCHEMY_CONN')
    db_str = db_str.replace('/','|').replace(':','|').replace('@','|').split('|')
    return db_str

COMMON_URL = get_url()
db_conn_str = get_db_conn()
neo_4j_url = get_neo_url()
neo_4j_username = get_neo_username()
neo_4j_password = get_neo_password()
