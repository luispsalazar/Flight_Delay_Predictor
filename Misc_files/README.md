# Final project: 

Travel delays are inevitable, our team wanted to look at airline travel data to see how weather and other factors influence airline delays (a flight delay is defined when it departs more than 15 minutes after the scheduled time and excludes cancellations). Predicting when delays might occur was done using a trained machine learning model in Google Colab and then visualized using Tableau. 



### Final visualization:

Data visualization and final presentation: https://public.tableau.com/views/Team3finalpresentation/StoryBoard?:language=en-US&publish=yes&:display_count=n&:origin=viz_share_link 

Full detailed presentation: https://docs.google.com/presentation/d/1qctDydEv7GRGtydRg5J1TvznyKwuBSe0/edit?usp=sharing&ouid=100563763732377884738&rtpof=true&sd=true 

### Topic and data details:

Data source: https://www.kaggle.com/datasets/threnjen/2019-airline-delays-and-cancellations


#### Reason for selected topic:

Travel delays are inevitable our team wants to look at airline travel data to see how weather influences airline delays greater than 15 minutes to help predict when delays might occur.

#### Description of their source of data: 

This dataset contains detailed airline, weather, airport and employment information for major airports in the US in 2019. 

#### General project workflow:

![Workflow Diagram](https://github.com/ethomas33/Team_3_Final_Project/blob/873b695064ac2036b640e8dfe104f6786ecb917e/Supporting%20resources%20and%20presentation/Workflow%20diagram.png)


#### File explained:

<br /> **Supporting resources and presentation** - this folder contains images feeding the read me, presentations related to this project, and exported results as excel files
<br /> **Misc_files** - this folder contains old version of files, test files, and files that were not successful
<br /> **SavedModels** - this folder contains the models used for this project
<br /> **.gitignore** - this is being used to stop the upload of the source data file as it is 1.4 gb csv file
<br /> **CSV_To_JSON.ipynb** - this notebook was used to convert the original csv data to a json format
<br /> **ConvertCSVtoLIBSV.ipynb** - this notebook converts csv to spark vector file
<br /> **ConvertCSVtoLIBSV.py** - this is the notebook converted to a python file
<br /> **Models_in_SPARK_R4.ipynb** - this notebook trains the machine learning models
<br /> **PredWithTrainedModel.ipynb** - this notebook uses new data to predict based on trained model
<br /> **README.md** - read me
<br /> **cleandelay_writeto_db.ipynb** - writing cleaned, encoded data to table "clean_delaytable" in pgAdmin database "Flightdelay_data-1." 
<br /> **create_writeto_db.ipynb** - writing tables "delaytable", "airports", "fltdeltabnolatlng", and "mergefiles" to pgAdmin database "Flightdelay_data-11." Also used for writing from "mergefiles" table to df in "create_writeto_db.ipynb" script.
<br /> **db_join_script.txt** - used to join pgAdmin tables "airports" and "fltdeltabnolatlng" to create table "mergefiles" in database "Flightdelay_data-1."
<br /> **explo_clean_data_3hot.ipynb** - this notebook cleans and encodes the raw data to be used in model
<br /> **explo_clean_data_3hot.py** - this is the notebook converted to a python file
<br /> **models_in_spark_r4.py** - this is the notebook converted to a python file
<br /> **predwithtrainedmodel.py** - this is the notebook converted to a python file
<br /> **requirements.txt** - this contains the required list of libraries for python environment (please note this currently contains more libraries than needed) 


#### Original data source:

| Column         | Description            |
|----------------|------------------------|
|MONTH |Month number (1-12 -- Jan. to Dec.)
|DAY_OF_WEEK | Day of the week (1-7 -- Mon. to Fri.)  
|DEP_DEL15 | Delayed over 15 min (binary - 1 = delay 0 = ontime) 
|DEP_TIME_BLK | Departure time block (59 min segment -- 0001 to 2359) 
|DISTANCE_GROUP | Flight distance flow groups 
|SEGMENT_NUMBER | The segment that this tail number is on for the day 
|CONCURRENT_FLIGHTS | Concurrent flights leaving from the airport in the same departure block 
|NUMBER_OF_SEATS | Number of seats on aircraft 
|CARRIER_NAME | Carrier name 
|AIRPORT_FLIGHTS_MONTH | Airport flights per month 
|AIRLINE_FLIGHTS_MONTH | Airline flights per month 
|AIRLINE_AIRPORT_FLIGHTS_MONTH | Airline specific airport flights per month 
|AVG_MONTHLY_PASS_AIRPORT | Airport average monthly passangers 
|AVG_MONTHLY_PASS_AIRLINE | Airline average monthly passangers
|FLT_ATTENDANTS_PER_PASS | Flight attendants per passanger  
|GROUND_SERV_PER_PASS | Ground service personal per passanger 
|PLANE_AGE | Plane age 
|DEPARTING_AIRPORT | Departing airport 
|LATITUDE | Latitude for airport 
|LONGITUDE | Longitude for airport 
|PREVIOUS_AIRPORT | Previous airpoirt ("NONE" if there wasn't one) 
|PRCP | Inches of precipitation for day 
|SNOW | Inches of snowfall for day
|SNWD | Inches of snow on ground for day
|TMAX | Max temperature for day 
|AWND | Max wind speed for day 


The original dataset has 25 feature (X) columns and 1 target (y) column. After calculating/adding the principal component column and dropping the Lat/Lng columns, the dataset contains 24 feature columns (21 numerical variables and 3 categorical variables) and 1 target column, before encoding the categorical variables. After encoding the DEP_TIME_BLK, CARRIER_NAME, and DEPARTING_AIRPORT columns (categorical variables) the input dataset contain 153 feature columns and 1 target column.

The cleaned, encoded dataset was then saved to table "clean_delaytable" in pgAdmin database "Flightdelay_data-1."
The "clean_delaytable" contains 153 feature columns and 1 target column for each of the 6,489,062 rows. 

This cleaned, encoded dataset is ready to be split into separate (X) and (y) datasets, with the (X) dataset being a compatible input for the Scaler() function. Saving the cleaned, encoded scaled (X) dataset to pgAdmin is highly recommended.

### Analysis details

When our model ran the following pattern of feature importance:

#### Random forest Model

![RandomForest](https://github.com/ethomas33/Team_3_Final_Project/blob/f12a9c3a575051f336628d2ef1730468d8cce06f/Supporting%20resources%20and%20presentation/Feature_Importance_%20Random_Forest.png)

#### Gradient-boosted algorithm

![RandomForest](https://github.com/ethomas33/Team_3_Final_Project/blob/ecf1bc2589599bfed36c4da24a25163c6c1e0c8a/Images/Feature_Importance_Gradient-boosted_algorithm.png)


### Team details

Team members: Elizabeth Thomas, Stephen Levy, Luis Salazar
