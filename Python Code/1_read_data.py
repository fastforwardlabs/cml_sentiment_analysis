from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import csv

url = 'http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip'

with urlopen(url) as response:
  with ZipFile(BytesIO(response.read())) as zfile:
    zfile.extractall('temp_data/sentiment140_unzipped')

with open('temp_data/sentiment140_unzipped/training.1600000.processed.noemoticon.csv', encoding='latin-1') as f_in:
  with open('temp_data/sentiment140_unzipped/clean_data.csv', 'w', encoding='utf-8') as f_out:
    f_out.write(f_in.read())    


count_positive = 0
count_neutral = 0
count_negative = 0

with open("temp_data/sentiment140_unzipped/clean_data.csv") as csvfile:
  reader = csv.reader(csvfile, delimiter=',')
  for row in reader:
    label_only=row[0]
    if label_only=='0':
      count_negative = count_negative + 1
    elif label_only=='2':
      count_neutral = count_neutral + 1
    else:
      count_positive = count_positive + 1

print("Positive Labels Count: " + str(count_positive))
print("Negative Labels Count: " + str(count_negative))
print("Neutral Labels Count: " + str(count_neutral))
