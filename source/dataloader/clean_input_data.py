import numpy as np
import pandas as pd
import regex as re

def strip_text_source(text):
    # remove leading attribution :%s:,"[^)]*\(([^)]*)\)*[^)]*) - :,"\1 :cg
    subtext = re.sub('^[^\)]*(\([^\)]*\))*[^\)]*\) - ', '', text)
    # remove disclaimers
    return re.sub('^[^:]*\: - ', '', subtext)

def clean_csv(input_filename, output_filename):
    df = pd.read_csv(input_filename,
            converters={'text': strip_text_source},
            nrows=100,
            skip_blank_lines=True)
    df.to_csv(output_filename)

if __name__ == '__main__':
    clean_csv('True.csv', 'Cleaned_True.csv')
    clean_csv('Fake.csv', 'Cleaned_Fake.csv')
