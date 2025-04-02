# Gunicorn configuration
workers = 4
threads = 2
timeout = 300  # 5 minutes
worker_class = 'sync'
max_requests = 1000
max_requests_jitter = 50
