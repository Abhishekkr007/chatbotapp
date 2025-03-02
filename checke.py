

import google.generativeai as genai

genai.configure(api_key="AIzaSyAZCRVDycvT7J3W2l7P9QEuFGvAwwLAOlI")

models = list(genai.list_models())  
print(models)
