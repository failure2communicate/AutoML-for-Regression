from collections import defaultdict
import sklearn
import joblib 
import pandas as pd
import numpy as np
import json
import os

class Main(object): 
    def __init__(self): 
        """ Initializes the model and all ancilliary data (e.g. word embeddings) """
        self.cur_dir = os.path.dirname(os.path.realpath(__file__))
        self.model = joblib.load(os.path.join(self.cur_dir, 'model.sav'))
    
    def predict(self, mlskill_input): 
        """ Once an ML Package is deployed as an ML Skill, this function will 
        be the endpoint callable by outside clients. If calling this ML Skill 
        through UiPath Studio, if the ML Skill was created with input type set 
        to file, the client will send a file in serialized bytes. 
        
        An example implementation for deserializing an image file: 
        ... 
        def predict(self, mlskill_input):
            from PIL import image 
            import io 
            image = Image.open(io.BytesIO(mlskill_input)) 
            
        :param str mlskill_input: input coming from a client. """

        import pandas as pd
        data = pd.read_json(mlskill_input)
        predictions = self.model.predict(data.values)
        return json.dumps({'predictions': predictions.tolist()})
            
if __name__ == '__main__':
    main = Main()
    df = pd.read_csv('data/test/evaluate.csv', header=0).head(20)
    df = df.drop('shares', axis=1)
    json_df = df.to_json(orient='records')
    print(main.predict(json_df))