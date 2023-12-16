from flask import Flask, render_template, request, jsonify
from pyspark.ml.feature import VectorAssembler  # Import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.ml import PipelineModel
from datetime import datetime
import logging
import urllib.request, json


# Initialize Flask
app = Flask(__name__)
logging.basicConfig(level = logging.INFO)
# Initialize SparkSession
spark = SparkSession.builder.appName("NYTaxiPredictionService").getOrCreate()

# Load your pre-trained Linear Regression model
loaded_lr_model = RandomForestRegressionModel.load("/Users/pavankumarkotapally/Taxi Project/models/rf_hyper")
loaded_dt_model = PipelineModel.load("/Users/pavankumarkotapally/Taxi Project/models/decision_tree_for_demand")

def getWeather(dt):
    date = dt.strftime('%Y-%m-%d')
    weather_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/new%20york%20city/"+date+"/"+date+"?unitGroup=metric&include=hours&key=<ENTER-YOUR-KEY-HERE>&contentType=json"
    response = urllib.request.urlopen(weather_url)
    data = response.read()
    weather_data = json.loads(data)
    temperature = weather_data['days'][0]['temp']
    temperature=(temperature*(9/5))+32
    condition = weather_data['days'][0]['conditions']
    return [temperature, condition]

def modify_address(x):
    x = x.split(' ')
    final_add = str(x[0])
    for i in range(1, len(x)):
        final_add = final_add + '%20' + str(x[i])
    return final_add

def getDistanceandDuration(pickup, dropoff):  
    distance_url = "https://maps.googleapis.com/maps/api/distancematrix/json?destinations="+dropoff+"&origins="+pickup+"&units=imperial&key=<ENTER-YOUR-KEY-HERE>"
    response = urllib.request.urlopen(distance_url)
    data = response.read()
    distance_data = json.loads(data)
    distance = distance_data['rows'][0]['elements'][0]['distance']['text']
    duration = distance_data['rows'][0]['elements'][0]['duration']['text']
    distance = float(distance.split(' ')[0])
    duration = float(duration.split(' ')[0])
    res = [distance, duration]
    return res

def getCoordinates(pickup):
    coordinates_url = "https://maps.googleapis.com/maps/api/geocode/json?key=ENTER-YOUR-KEY-HERE>&address=" + pickup
    response = urllib.request.urlopen(coordinates_url)
    data = response.read()
    coordinates_data = json.loads(data)
    coordinates_latitude = coordinates_data['results'][0]['geometry']['location']['lat']
    coordinates_longitude = coordinates_data['results'][0]['geometry']['location']['lng']
    res = [coordinates_latitude, coordinates_longitude]
    return res

def getDemandAndPrice(temp, y, m, d, h, mi, lat, lng, isHoliday, passenger_count, trip_duration, trip_distance, speed):
    input_data_demand = [(temp, y, m, d, h, lat, lng, isHoliday)]
    schema_demand = ["temp", "year", "month", "day", "hour", "Latitude", "Longitude", "is_weekend"]
    input_df_demand = spark.createDataFrame(input_data_demand, schema=schema_demand)
    
    assembler = VectorAssembler(
        inputCols=["temp", "year", "month", "day", "hour", "Latitude", "Longitude", "is_weekend"],
        outputCol="featuresDemand"
    )
    
    input_df_demand = assembler.transform(input_df_demand)
    
    predictions_demand = loaded_dt_model.transform(input_df_demand)
    
    demand_predicted = predictions_demand.select("prediction").collect()[0][0]
    de = demand_predicted ** 2
    
    
    input_data = [(y, m, d, h, mi, temp, isHoliday, passenger_count, trip_duration, lat, lng, trip_distance, speed, demand_predicted,de)]
    schema = ["year","month","day","hour","minute",'temp','isHoliday',"passenger_count" ,'trip_duration', 'Latitude',"Longitude", "trip_distance","speed_mph","demand_category","squared_demand"]
    input_df = spark.createDataFrame(input_data, schema=schema)

    
    assembler = VectorAssembler(
        inputCols=["year","month","day","hour","minute",'temp','isHoliday',"passenger_count" ,'trip_duration', 'Latitude',"Longitude", "trip_distance","speed_mph","demand_category","squared_demand"],
        outputCol="features"
    )

    
    input_df = assembler.transform(input_df)

    
    predictions = loaded_lr_model.transform(input_df)

    
    prediction_result = predictions.select("prediction").collect()[0][0]
    
    return [demand_predicted, prediction_result]
    


@app.route("/", methods=["GET"])
def index():
    return render_template("taxi1.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()
    dt = data['datetime']
    dt = datetime.fromisoformat(dt)
    y = dt.year
    m = dt.month
    d = dt.day
    h = dt.hour
    mi = dt.minute
    [temp, condition] = getWeather(dt)
    isHoliday = 0 # tbd
    passenger_count = int(data['passenger_count'])
    pickup_location = modify_address(data['pickupLocation'])
    drop_location = modify_address(data['dropoffLocation'])
    [trip_distance, trip_duration] = getDistanceandDuration(pickup_location, drop_location)
    speed = trip_distance / (trip_duration / 60)
    [lat, lng] = getCoordinates(pickup_location)
    
    [demand, price] = getDemandAndPrice(temp, y, m, d, h, mi, lat, lng, isHoliday, passenger_count, trip_duration, trip_distance, speed)
    
    if h == 23:
        [demand_1, price_1] = getDemandAndPrice(temp, y, m, d + 1, 0, mi, lat, lng, isHoliday, passenger_count, trip_duration, trip_distance, speed)
    else:
        [demand_1, price_1] = getDemandAndPrice(temp, y, m, d, h + 1, mi, lat, lng, isHoliday, passenger_count, trip_duration, trip_distance, speed)
    demand = int(demand)
    demand_1 = int(demand_1)
    
    if demand_1 > demand:
        statement = "Demand will rise in the next hour."
    elif demand_1 == demand:
        statement = "Oops! The demand won't change in the next hour."
    else:
        statement = "Demand will fall in the next hour."

    if price_1 > price:
        statement = statement + "Fares will increase by ${} ".format(str(abs(round(price_1 - price, 2))))
    elif price_1 < price:
        statement = statement + "Fares will decrease by ${} ".format(str(abs(round(price_1 - price, 2))))
    else:
        statement = statement + "Fares will remain the same."
        
    
    if demand == 1:
        demand = "Low"
    elif demand == 2:
        demand = "Medium"
    else:
        demand = "High"
        
    pred = {'price': round(price, 2), 'temperature': temp, 'condition': condition, 'demand': demand, 'price_1': round(price_1, 2), "demand_1": demand_1, "statement": statement}

    return jsonify(pred)


if __name__ == "__main__":
    app.run(debug=True)