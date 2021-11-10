Digital Image Processing Online

Install Requirements by using
      pip install -r requirements.txt

put your "images" folder inside static folder

edit the get_neighbors method in main to return whatever neighbors you want.
The result be a list of string names of image files (in png format) in your new static/images folder

In root, open terminal and run
      uvicorn main:app --reload
to run the app in development mode.