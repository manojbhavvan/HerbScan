# HerbScan
Identification of Different Medicinal Plants/Raw materials through Image Processing Using Machine Learning Algorithms

## Download Dataset
https://www.kaggle.com/datasets/riteshranjansaroj/segmented-medicinal-leaf-images

## Zip in the Repository and extract it
1. Check for flask version else download it with this command
     $ pip install -U Flask.
2. Check for all the other packages. If any packages are missing using this command
     $ pip install -U <package-name>
3. Run the web application 
    $ python app.py
4. Select the file which is to be tested.
5. The application will display the image predicted with confidence level of the MobileNetV2 model prediction.

## Work to be done.
1. Develop a database with all the information of the dataset classes.
2. Upload the database into a REST API.(So far plan for local API).
3. Fetch the details and display.

Additional works.
1. Design for the frontend of the application.
2. Discussion about more features.
