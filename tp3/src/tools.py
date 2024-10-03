import json

def import_json(file_name):

    f= open(file_name, 'r')
    j=json.load(f)  
    f.close()
    return { atr: j[atr] for atr in j}
