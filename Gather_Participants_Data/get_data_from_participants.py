import requests
import datetime

start = int(datetime.datetime(2021,2,26,0,0,0).timestamp())
end = int(datetime.datetime(2021,3,1,0,0,0).timestamp())

user_dict = {}

headers = {
    'Authorization': 'WyI2MDBlNzZkZTk3MDU1Mzc2NjU2NmZmMTEiLCIkNSRyb3VuZHM9NTM1MDAwJGp4Q3BrMWhrbVczdTV5TXYkOC96aDVDZ3I4a0NsdTBoaVhYeUNxQTRZSVk1YTBBYzlQa1lOeVkwb0xlNiJd.YCR7ww.kIjPQp-0DrQxCg0PLnbPntP3tnY'
}

req = requests.get('https://unite.healthscitech.org/api/group/5e56c2b4cb55babfe186b80a/users?page=0&per_page=25',
    headers=headers)

for user in req.json():
    user_dict[user['email']] = user['_id']['$oid']

question_type = {
    'EMA': '5e56d1decb55babfe186b80d',
    'Weekly Survey': '5f7644c6c646f0e52a54ab95',
    'Morning Daily Survey': '600763d4d878cdd3d7f48606',
    'Evening Daily Survey': '600765c6d878cdd3d7f48607'
}


valid_user_list = ['uniterct148','uniterct160','uniterct197','uniterct265','uniterct446','uniterct470','uniterct533','uniterct552','uniterct656','uniterct729','uniterct749','uniterct774']
valid_user_list.sort()

print('UserID', end=',')
for q_type in question_type:
    print(q_type, end=',')
print('Total Watch Data')

for user in valid_user_list:
    if user not in user_dict:
        continue
    print(user, end=',')
    user_id = user_dict[user]

    for q_type in question_type:
        type_id = question_type[q_type]

        req = requests.get(f'https://unite.healthscitech.org/api/prompt/{type_id}/submission/count?users={user_id}&start={start}&end={end}',
            headers=headers)

        print(f'{req.json()["count"]}', end=',')

    req = requests.get(f'https://unite.healthscitech.org/api/sensing/data/group/5e56c2b4cb55babfe186b80a/field/samsung_raw_last_upload?users={user_id}&start={start}&end={end}',
        headers=headers)
    total_watch_data = len(req.json())
    print(total_watch_data)




