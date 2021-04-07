## How to use

First, change the two lines in the code to the desired time:
```
start = int(datetime.datetime(2021,2,26,0,0,0).timestamp())
end = int(datetime.datetime(2021,3,1,0,0,0).timestamp())
```
Then, run the script and pipe it to a file:
```
python3 get_data_from_participants.py > new_file.csv
```
The file new_file.csv should contain data from each participants. The participant list is taken from the website, and is filtered through a hard-coded list.
