# Final project: 

### Final visualization:
Insert link to dashboard here.
https://public.tableau.com/views/Team3finalpresentation/StoryBoard?:language=en-US&publish=yes&:display_count=n&:origin=viz_share_link 

### Topic and data details:

Data source: https://www.kaggle.com/datasets/threnjen/2019-airline-delays-and-cancellations

Dashboard: https://public.tableau.com/shared/WGCPFJ4GC?:display_count=n&:origin=viz_share_link

#### Reason for selected topic:

Travel delays are inevitable our team wants to look at airline travel data to see how weather influences airline delays greater than 15 minutes to help predict when delays might occur.

#### Description of their source of data: 

This dataset contains detailed airline, weather, airport and employment information for major airports in the US in 2019. 

#### File explained:
.gitignore - this is being used to stop the upload of the source data file as it is 1.4 gb csv file
README.md - this is feeding this readme
create_raw_dbases.ipynb - 
create_writeto_db.ipynb - 
db_join_script.txt - 
delay_model.ipynb - 
delay_model_fewrows.ipynb - 
flight.txt - 
flight_delay.ipynb - 



#### Original data source columns:

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


### Team details

Team members: Elizabeth Thomas, Stephen Levy, Luis Salazar

Detailed workflow presentation:  <br />
Liz notes ---- this is a viewable link only, will move final power point to inside repo when done --- https://docs.google.com/presentation/d/1qctDydEv7GRGtydRg5J1TvznyKwuBSe0/edit?usp=sharing&ouid=100563763732377884738&rtpof=true&sd=true 
