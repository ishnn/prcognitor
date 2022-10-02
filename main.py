from tkinter import * # for GUI designing...
import ctypes
from PIL import ImageTk, Image #python image library
import tkinter.messagebox as tkMessageBox
from keras.models import load_model
import numpy as np
import pandas as pd
import random,os
from datetime import datetime
import pytz,game
import tensorflow as tf
from keras import preprocessing
from sklearn.preprocessing import StandardScaler
from numpy.random import seed
from tensorflow.random import set_seed
from twilio.rest import Client

window = Tk()
img = Image.open("images\\home.png")
img = ImageTk.PhotoImage(img)
panel = Label(window, image=img)
panel.pack(side="top", fill="both", expand="yes")

user32 = ctypes.windll.user32
user32.SetProcessDPIAware()
[w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
lt = [w, h]
a = str(lt[0]//2-430)
b= str(lt[1]//2-329)

window.title("procognitor")
window.geometry("1200x700+"+a+"+"+b)
window.resizable(0,0)
solar_wind = pd.read_csv("./input/solar_wind.csv")
solar_wind.timedelta = pd.to_timedelta(solar_wind.timedelta)
solar_wind.set_index(["period", "timedelta"], inplace=True)

dst = pd.read_csv("./input/labels.csv")
dst.timedelta = pd.to_timedelta(dst.timedelta)
dst.set_index(["period", "timedelta"], inplace=True)

sunspots = pd.read_csv("./input/sunspots.csv")
sunspots.timedelta = pd.to_timedelta(sunspots.timedelta)
sunspots.set_index(["period", "timedelta"], inplace=True)

def get_train_test_val(data, test_per_period, val_per_period):
    """Splits data across periods into train, test, and validation"""
    # assign the last `test_per_period` rows from each period to test
    test = data.groupby("period").tail(test_per_period)
    interim = data[~data.index.isin(test.index)]
    # assign the last `val_per_period` from the remaining rows to validation
    val = data.groupby("period").tail(val_per_period)
    # the remaining rows are assigned to train
    train = interim[~interim.index.isin(val.index)]
    return train, test, val



seed(2020)
set_seed(2021)

# subset of solar wind features to use for modeling
SOLAR_WIND_FEATURES = [
    "bt",
    "temperature",
    "bx_gse",
    "by_gse",
    "bz_gse",
    "speed",
    "density",
]

# all of the features we'll use, including sunspot numbers
XCOLS = (
    [col + "_mean" for col in SOLAR_WIND_FEATURES]
    + [col + "_std" for col in SOLAR_WIND_FEATURES]
    + ["smoothed_ssn"]
)


def impute_features(feature_df):
    """Imputes data using the following methods:
    - `smoothed_ssn`: forward fill
    - `solar_wind`: interpolation
    """
    # forward fill sunspot data for the rest of the month
    feature_df.smoothed_ssn = feature_df.smoothed_ssn.fillna(method="ffill")
    # interpolate between missing solar wind values
    feature_df = feature_df.interpolate()
    return feature_df


def aggregate_hourly(feature_df, aggs=["mean", "std"]):
    """Aggregates features to the floor of each hour using mean and standard deviation.
    e.g. All values from "11:00:00" to "11:59:00" will be aggregated to "11:00:00".
    """
    # group by the floor of each hour use timedelta index
    agged = feature_df.groupby(
        ["period", feature_df.index.get_level_values(1).floor("H")]
    ).agg(aggs)
    # flatten hierachical column index
    agged.columns = ["_".join(x) for x in agged.columns]
    return agged


def preprocess_features(solar_wind, sunspots, scaler=None, subset=None):
    """
    Preprocessing steps:
        - Subset the data
        - Aggregate hourly
        - Join solar wind and sunspot data
        - Scale using standard scaler
        - Impute missing values
    """
    # select features we want to use
    if subset:
        solar_wind = solar_wind[subset]

    # aggregate solar wind data and join with sunspots
    hourly_features = aggregate_hourly(solar_wind).join(sunspots)

    # subtract mean and divide by standard deviation
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(hourly_features)

    normalized = pd.DataFrame(
        scaler.transform(hourly_features),
        index=hourly_features.index,
        columns=hourly_features.columns,
    )

    # impute missing values
    imputed = impute_features(normalized)

    # we want to return the scaler object as well to use later during prediction
    return imputed, scaler

features, scaler = preprocess_features(solar_wind, sunspots, subset=SOLAR_WIND_FEATURES)
assert (features.isna().sum() == 0).all()

data_config = {"timesteps": 12, "batch_size": 500}


YCOLS = ["t0", "t1"]


def process_labels(dst):
    y = dst.copy()
    y["t1"] = y.groupby("period").dst.shift(-1)
    y.columns = YCOLS
    return y


labels = process_labels(dst)
data = labels.join(features)
def timeseries_dataset_from_df(df, batch_size):
    dataset = None
    timesteps = data_config["timesteps"]

    # iterate through periods
    for _, period_df in df.groupby("period"):
        # realign features and labels so that first sequence of 32 is aligned with the 33rd target
        inputs = period_df[XCOLS][:-timesteps]
        outputs = period_df[YCOLS][timesteps:]

        period_ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            inputs,
            outputs,
            timesteps,
            batch_size=batch_size,
        )

        if dataset is None:
            dataset = period_ds
        else:
            dataset = dataset.concatenate(period_ds)

    return dataset


def alertzone():

    place=''
    for timeZone in pytz.all_timezones:
        newYorkTz = pytz.timezone(timeZone) 
        timeInNewYork = datetime.now(newYorkTz)
        currentTimeInNewYork = timeInNewYork.strftime("%H:%M:%S")
        hr = int(currentTimeInNewYork.split(":")[0])
        if hr>=12 or hr<=19:
            place +=timeZone+'\n'
    file = open("Alert Zone.txt","w")
    file.write(place)
    file.close()

def message():
    
    account_sid = "AC68830eda262fe635e7fc39f18dfc6b02"
    auth_token = "633be3aff7476c5df31fb06237cdbb01"
    client = Client(account_sid, auth_token)

    message = client.messages.create(
                                  body='A big geomagnetic storm might occur in next 2-3 hours.',
                                  from_='+14245775280',
                                  to='+918604629998'
                              )
message()
def detect():
    train, test, val = get_train_test_val(data, test_per_period=6_000, val_per_period=3_000)
    
    test_ds = timeseries_dataset_from_df(test, data_config["batch_size"])

    model = load_model("model.h5")
    
    x = model.predict(test_ds)

    y = random.randint(0,len(x)-1)
    x = x[y][x[y].argmax()]

    if x>0:
        alertzone()
        message()
        r = tkMessageBox.askquestion("Procognitor","A big geomagnetic storm might occur in next 2-3 hours.\n\n Alert SMS messege has been sent to all high alert zones \n\n press yes to see the list of all High Alert Zones")
        if r.lower()=="yes":
            os.popen(os.getcwd()+'/'+r"Alert Zone.txt")
            
    else:
        tkMessageBox.showinfo("Procognitor","No chances of any big geomagnetic storm according to current data")

def precaution():
    tkMessageBox.showinfo("Procognitor","* The best way is to shut down the power supply grid of plant to prevent event .\n* Although,Carrington event can be prevented by  having fly wheel and damphering device with high capacity in the universe.\n * Inbuiliting the damphering device in the system so that if event occurs  there will be less damage and communication can be possible.")
    
b = Image.open("Images\\1.png")
b = ImageTk.PhotoImage(b)
b1 = Button(window,image = b,bd=0,highlightthickness=0 ,bg="white",activebackground = "black",command=detect)
b1.place(x=768,y=396)


setting = Image.open("Images\\2.png")
setting = ImageTk.PhotoImage(setting)
b2 = Button(window,image = setting,bd=0,highlightthickness=0 ,bg="white",activebackground = "black",command=precaution)
b2.place(x=768,y=483)

b3 = Image.open("Images\\3.png")
b3 = ImageTk.PhotoImage(b3)
b3b = Button(window,image = b3,bd=0,highlightthickness=0 ,bg="white",activebackground = "black",command=game.game)
b3b.place(x=768,y=570)

window.mainloop()
