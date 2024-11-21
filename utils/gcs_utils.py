import json
import gcsfs
from utils.private.private import TOKEN, GCLOUD_PROJECT

def get_gcs_info():
    with open(TOKEN, 'r', encoding='utf-8') as f:
        token = json.load(f)
    fs = gcsfs.GCSFileSystem(project=GCLOUD_PROJECT,
                        token=token)    
    return fs, token