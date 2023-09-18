# -*- coding: utf-8 -*-

import sys
import logging
from IPython import get_ipython
    
class Logger():
    ''' Logger class for logging console outputs during model training.'''
    
    def __init__(self, project_directory, project_name, log_file_name="nb.log"):
        # Path to log file
        self.log_file = project_directory + "/" + project_name + \
            "/" + log_file_name 
        
    def remove_updated_lines_from_logs(self):
        '''Deletes intermediate outputs during a running epoch from the log file'''
        with open(self.log_file, "w+") as f:
            lines = f.readlines()
            for line in lines:
                if not '\x08' in line:
                    f.write(line)
        
    def activate_logging(self):
        '''Activates the logger'''
        print("Logging to " + self.log_file)
        nblog = open(self.log_file, "a+")
        sys.stdout.echo = nblog
        sys.stderr.echo = nblog
    
        get_ipython().log.handlers[0].stream = nblog
        get_ipython().log.setLevel(logging.INFO)
        
        self.remove_updated_lines_from_logs()