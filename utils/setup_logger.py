import logging

log_level=logging.INFO    
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=log_level,
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='runlog.log'
)
logger = logging.getLogger('eval')