import logging
import coloredlogs

coloredlogs.install(level='INFO')
# coloredlogs.install(milliseconds=True)
coloredlogs.install(fmt='%(asctime)s - %(name)s[%(process)d] %(levelname)s %(message)s')
logger = logging.getLogger('LOG')

# coloredlogs.install(level='INFO', logger=logger)
# logger.setLevel(logging.INFO)
# ch = logging.StreamHandler()
# ch.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# ch.setFormatter(formatter)
# logger.addHandler(ch)
