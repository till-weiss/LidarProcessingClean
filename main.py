import config.config as configuration

config = configuration.Configuration().validate()

from core.report import write_report
write_report(config)   

import preprocessing
import processing
import validation

if __name__ == '__main__':
    #
    preprocessing.preprocess_all(config)
    
    processing.process_all(config)

    validation.validate_all(config)
