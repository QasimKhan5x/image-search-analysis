
# Reverse Image Search Web App

Install Requirements by using
    pip install -r requirements.txt

Put your "images" folder inside static folder

Edit the `get_nn_neighbors` method in main.py to return whatever neighbors you want from wherever you want (a ML model or random etc).
The result be a list of string names of image files (in .png format) in your new static/images folder

In root, open terminal and run
`uvicorn main:app --reload`
to run the app in development mode.
