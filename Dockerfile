FROM python:3

RUN mkdir -p /application
COPY 'All_Test_Data_With_Descriptions.xlsx' /application/

ADD main_file.py /application

RUN pip install pandas
RUN pip install xlrd
RUN pip install numpy
RUN pip install matplotlib
RUN pip install seaborn
RUN pip install sklearn
RUN pip install imblearn
RUN pip install statsmodels

CMD [ "python", "./application/main_file.py" ]