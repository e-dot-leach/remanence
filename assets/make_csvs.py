import requests
import json
from internetarchive import search_items, get_item
import cv2
import csv
import numpy as np
import re
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import time
import pandas as pd
from pathlib import Path
import random

def getNearestWebSafeColor(r, g, b):
    r = int(round( ( r / 255.0 ) * 5 ) * 51)
    g = int(round( ( g / 255.0 ) * 5 ) * 51)
    b = int(round( ( b / 255.0 ) * 5 ) * 51)
    return rgb2hex(r, g, b)

def rgb2hex(r,g,b):
    return f'#{int(round(r)):02x}{int(round(g)):02x}{int(round(b)):02x}'

def parse_title_for_date_string(t):
    left_ixs = [i for i, c in enumerate(t) if c == '(']
    right_ixs = [i for i, c in enumerate(t) if c == ')']
    if len(left_ixs) == 1:
        dtxt = t[left_ixs[0] + 1:right_ixs[0]]
    else:
        dtxt = t[left_ixs[-1] + 1:right_ixs[-1]]
    return dtxt

def parse_date_string_for_ia_search(s):
    ### first lets find a year
    pattern = r"\d{4}"
    ns = re.findall(pattern, s)
    if len(ns) > 1:
        raise ValueError('Too many years...')
    else:
        year = int(ns[0])
        s = s.replace(ns[0],'')
    ### next let's look for a month
    months = {'Jan':1,'January':1,'Feb':2,'February':2,'Mar':3,'March':3,
              'Apr':4,'April':4,'May':5,'Jun':6,'June':6,'Jul':7,'July':7,
              'Aug':8,'August':8,'Sep':9,'September':9,'Oct':10,'October':10,
              'Nov':11,'November':11,'Dec':12,'December':12}
    included_months = dict()
    for m in months.keys():
        if m in s:
            included_months[months[m]] = s.index(m)+len(m)
    month_ixs=dict()
    for i,(k,v) in enumerate(included_months.items()):
        try:
            month_ixs[k] = [v, list(included_months.values())[i+1]]
        except IndexError:
            month_ixs[k] = [v, len(s)-1]
    ### then, the days of the month
    days = dict()
    for k,v in month_ixs.items():
        ds = re.findall(r'\d+',s[v[0]:v[1]])
        if len(ds) == 0:
            days[k] = [-1]
        else:
            days[k] = [int(x) for x in ds]
    # finally, we'll put it all together
    end_of_months = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        end_of_months[2] = 29
    dates = list()
    for k,v in days.items():
        for d in v:
            if d == -1:
                dates.append(f'{year}-{k:02d}-01 TO {year}-{k:02d}-{end_of_months[k]:02d}')
            else:
                dates.append(f'{year}-{k:02d}-{d:02d}')
    dates = 'date:['+' AND '.join(dates)+']'
    return dates

def get_views(identifier):
    resp = requests.get('https://be-api.us.archive.org/views/v1/short/' + identifier)
    a = json.loads(resp.content)
    return a[identifier]['all_time']

def get_color_and_per_text(url,plotit=False):
    resp = requests.get(url)
    if resp.status_code != 200:
        time.sleep(10)
        resp = requests.get(url)
        if resp.status_code != 200:
            raise ValueError(f"can't get image at {url}")
    img = np.frombuffer(resp.content, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pix_vals = np.float32(img.reshape((-1, 3)))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 1e-5)
    k = 2
    retval, label, center = cv2.kmeans(pix_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    if plotit:
        seg_im = center[label.flatten()].reshape(img.shape).astype(np.uint8)
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(img)
        ax[1].imshow(seg_im)
        plt.tight_layout()
        plt.show()

    centers = np.uint8(center)
    labels = label.flatten()
    label_counts = np.bincount(labels)
    ix_sort = np.argsort(label_counts)[::-1]
    letter_color = centers[ix_sort[0]]
    pen_per = label_counts[ix_sort[1]]/(label_counts[ix_sort[0]]+label_counts[ix_sort[1]])

    return letter_color, 100*pen_per

def get_thumb_url(identifier):
    return f'https://archive.org/download/{identifier}/__ia_thumb.jpg'

def make_marion_journal_csv(savename='marion_journal.csv'):
    search = search_items('collection:marionstokesjournal', fields=['title', 'identifier', 'imagecount', 'identifier-access'])
    keys = ['title','identifier','url','date','pages','thumb_url','views','color','per_text','pen_color']

    with open(savename, 'w', newline="") as csvfile:
        writer = csv.DictWriter(csvfile, keys)
        writer.writeheader()

        for item in tqdm(search,total=424):
            data = dict()
            data['title'] = item['title']
            data['identifier'] = item['identifier']
            data['url'] = item['identifier-access']
            data['date'] = parse_date_string_for_ia_search(parse_title_for_date_string(item['title']))
            try:
                data['pages'] = item['imagecount']
            except Exception as e:
                data['pages'] = 0
            data['thumb_url'] = get_thumb_url(item['identifier'])#item['identifier-access'].replace('details','download')+'/__ia_thumb.jpg'
            data['views'] = get_views(item['identifier'])
            c, pt = get_color_and_per_text(data['thumb_url'])
            tqdm.write(f'{c}')
            data['color'] = c
            data['per_text'] = f'{pt:.1f}'
            data['pen_color'] = 'undetermined'
            writer.writerow(data)

def get_cia_first_image(item):
    base = item['identifier'].replace('cia-readingroom-document-', '')
    return item['identifier-access'].replace('details','download') + '/' + base + '_jp2.zip/' + base + '_jp2%2F' + base + '_0000.jp2&ext=jpg'


def make_cia_csv(savename='cia_journal_links.csv',journal_csv='marion_journal.csv'):
    journal_data = dict()
    with open(journal_csv, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            journal_data[row['identifier']] = row['date']

    keys = ['name','cia_number','identifier','url','date','linked_journal_id','pages','views','first_page_url','thumb_url']
    with open(savename, 'w', newline="") as csvfile:
        writer = csv.DictWriter(csvfile, keys)
        writer.writeheader()

        for journal_id,date_search in tqdm(journal_data.items(), total=424):
            query_string = f'collection:(ciareadingroom) {date_search}'
            search = search_items(query_string, fields=['title', 'identifier', 'imagecount', 'identifier-access', 'date'])
            if len(list(search)) > 100:
                search = list(search)
                search = random.sample(search, 100)
            for item in tqdm(search,total=len(list(search)),leave=False):
                data = dict()
                try:
                    data['linked_journal_id'] = journal_id
                    data['identifier'] = item['identifier']
                    data['views'] = get_views(item['identifier'])
                    if 'date' in item.keys():
                        data['date'] = item['date'].split('T')[0]
                    else:
                        data['date'] = ''
                    if 'title' in item.keys():
                        data['name'] = ''.join(item['title'].split(':')[1::])[1::]
                        data['cia_number'] = item['title'].split(':')[0].replace('CIA READING ROOM ','')
                    else:
                        data['name'] = ''
                        data['cia_number'] = ''
                    if 'imagecount' in item.keys():
                        data['pages'] = item['imagecount']
                    else:
                        data['pages'] = -1

                    if 'identifier-access' in item.keys():
                        data['url'] = item['identifier-access']
                        data['first_page_url'] = get_cia_first_image(item)
                        data['thumb_url'] = get_thumb_url(item['identifier'])
                    else:
                        data['url'] = ''
                        data['first_page_url'] = ''
                    writer.writerow(data)
                except Exception as e:
                    tqdm.write(f'Found error {e}. Skipping {item}')
                    
if __name__ == "__main__":
    #make_marion_journal_csv('marion_journal.csv')
    #make_cia_csv()
    import json
    with open('cia_journal_links.json') as f:
        data = json.load(f)
    
    new_data = dict()
    for k,v in data.items():
        new_list = list()
        for i in v:
            new_item = dict()
            for nk in ['name','cia_number','identifier','url','date','linked_journal_id','pages','views','first_page_url','thumb_url']:
                new_item[nk] = i[nk]
            new_item['pdf_url'] = new_item['thumb_url'].replace('__ia_thumb.jpg',new_item['identifier'].replace('cia-readingroom-document-','')+'.pdf')
            new_list.append(new_item)
        new_data[k] = new_list
    
    with open('cia_journal_links_new.json','w') as f:
        json.dump(new_data, f, indent=2)