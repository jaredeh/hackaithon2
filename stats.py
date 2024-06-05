from model import SimpleFNN
import requests
from datetime import datetime, timedelta

class StorageStats():

    def __init__(self, services={"superman": 1, "batman": 7}, threshold=0.5):
        self.model = SimpleFNN(6,1)
        self.model.load('models/20240605094642.safe')
        self.stats = {} #basic key info
        self.accesses = [] #access log
        self.migrations = [] #migration log
        self.services = services
        self.threshold = threshold
        self.load_keys()
    
    def load_keys(self):
        service_keys = [{"key": "lego-batman.jpeg", "platform": 0}, 
        {"key": "lego-superman.jpeg", "platform": 0}]
        for object in service_keys:
            self.add_object(key=object['key'], platform=object['platform'])
    
    def add_object(self, key=None, filetype=0, platform=None, ppi_check=0, source=0, permissions=0, last_service=0):
        if key is None or platform is None:
            raise Exception("Missing 'key' or 'platform'")
        self.stats[key] = {'filetype': filetype, 'platform': platform, 'ppi_check': ppi_check,'source': source, 'permissions': permissions, 'last_service': last_service}

    def platform(self, key):
        if key not in self.stats:
            raise Exception(f"Key {key} not found")
        return self.stats[key]['platform']
 
    def log_get(self, key, service_str):
        if key not in self.stats:
            raise Exception(f"Key {key} not found")
        old_platform = self.stats[key]['platform']
        service = self.services.get(service_str, 0)
        self.stats[key]['last_service'] = service
        self.accesses.append({"key": key, "ts": datetime.now(), "service": service})
        print(f"Running model on key {key}")
        pred = self.model.run(self.stats[key])
        if pred >= self.threshold and old_platform == 0:
            print(f"Key {key} will be migrated")
            self.stats[key]['platform'] = 1
            response = requests.post("http://127.0.0.1:5000/migrate",json={"key": key, "platform": 1})
            response.raise_for_status()
        else:
            print(f"Key {key} will NOT be migrated")
        

    def log_migration(self, key):
        if key not in self.stats:
            raise Exception(f"Key {key} not found")
        self.migrations.append({"key": key, "ts": datetime.now()})

    def filter_by_migrations(self,service_str, timestamp):
        service = self.services.get(service_str, 0)
        migrated_keys = [migration['key'] for migration in self.migrations if migration['ts'] < timestamp]
        accessed_keys = [access['key'] for access in self.accesses if access['service'] == service and access['ts'] < timestamp]
        filtered = [key for key in migrated_keys if key in accessed_keys]
        return filtered
if __name__ == "__main__":
    ss = StorageStats()
    key = list(ss.stats.keys())[0]
    ss.log_get(list(ss.stats.keys())[1], "superman")
    a = ss.filter_by_migrations("batman", datetime.now())
    print(a)
    ss.log_get(key, "batman")
    ss.log_get(key, "batman")
    a = ss.filter_by_migrations("batman", datetime.now())
    print(a)
    a = ss.filter_by_migrations("batman", datetime.now() - timedelta(minutes=10))
    print(a)
    a = ss.filter_by_migrations("superman", datetime.now())
    print(a)