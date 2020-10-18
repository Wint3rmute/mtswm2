import re

def get_readme_features_and_diagnoses(max_features = 59):
    with open("readme.md") as f:
        content = f.readlines()
        
        # Number followed with dot
        regex = re.compile(r'^(\d+)\.\s')
        
        # get only lines passing regexp
        features_lines = list(filter(regex.search, content))
        
        # remove number
        features_lines = [l.split(' ', 1)[1] for l in features_lines]
        features_lines = [l.replace('\n', '').strip() for l in features_lines]
        
        #      features             diagnoses
        return features_lines[:59], features_lines[-5:]