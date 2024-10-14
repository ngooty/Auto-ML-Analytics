import setuptools
setuptools.setup(
    name='mareana_machine_learning',
    version='0.1',
    description='Mareana Machine Learning Package',
    url='#',
    author='Nagaraju Gooty',
    install_requires=['scikit-learn', 'pandas', 'psycopg2-binary','urllib','requests','mlflow','warnings','os','json','numpy','dotenv','datatime','tempfile','lightgbm','xgboost','catboost','plotly','plotly.express','plotly.graph_objects','pandas-profiling','logging','sqlalchemy','sshtunnel'],
    author_email='nagaraju@mareana.com',
    packages=setuptools.find_packages(),
    zip_safe=False
)