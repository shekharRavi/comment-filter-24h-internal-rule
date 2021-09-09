import shutil
import psutil

from flask import redirect

def health():
    disk_total, disk_used, disk_free = shutil.disk_usage("/")
    memory = psutil.virtual_memory()

    mlp_health = {
        'service': 'comment_api',
        'version': '1.0',

        'disk': {
            'free': disk_free / (2 ** 30),
            'total': disk_total / (2 ** 30),
            'used': disk_used / (2 ** 30),
            'unit': 'GB'
        },
        'memory': {
            'free': memory.available / (2 ** 30),
            'total': memory.total / (2 ** 30),
            'used': memory.used / (2 ** 30),
            'unit': 'GB'
        },
        'cpu': {
            'percent': psutil.cpu_percent()
        },

    }


    return mlp_health

def documentation():
    return redirect('/swagger.json')
