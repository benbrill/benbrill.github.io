---
layout: post
title: Interactive Visualizations using Climate Data
image: benbrill.github.io\images\ucla-math.png
---

Climate Change is the most dangerous crisis facing our communities, our country, and our world. **Period.** Not only does letting this crisis remain unsolved jeopordize the existance of humanity in the long run, but it disproportionately affects communities of color in the short term. 

Often times, the numbers surrounding climate change might be hard to digest for some. $CO_2~ppm$? What does that even mean? Visualizations can often get the point of this dire crisis across in a much more susinct and effective manner. Let's take a look at a few.

### Creating a Database of Climate information

Across the world, there are thousands of stations who record climate data. Many have been doing every month for every year, since 1980. If you do the math, you'll figure out that that is a lot of rows to have in a dataframe stored in our memory. We want to have access to all of this data, but we don't want it all to be on our memory at the same time, since we won't need it all at once. **Enter: SQL Databases.**

We can create a database hosted locally on our computer that will store all of our data in a local environment. We then will use `SQL` to query the database to get only the information we need at certain points to then create our visualizations




```python
# import core libraries
import pandas as pd
import numpy as np
import sqlite3
```

We are first going to load our smaller tables into our database, which includes:
- `countries`: a list of all the countries in the world with common identifier codes
- `stations`: basic data about the location of climate measuring stations
- `continents`: an extra database identifying which continent each country is in

We will later use SQL to relate and merge these tables to one another


```python
url = "https://raw.githubusercontent.com/mysociety/gaze/master/data/fips-10-4-to-iso-country-codes.csv"
countries =  pd.read_csv(url)
# rename columns to better names
countries = countries.rename({"FIPS 10-4": "countryID", "ISO 3166": "ISO"}, axis = 1)
countries
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>countryID</th>
      <th>ISO</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AF</td>
      <td>AF</td>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AX</td>
      <td>-</td>
      <td>Akrotiri</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AL</td>
      <td>AL</td>
      <td>Albania</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AG</td>
      <td>DZ</td>
      <td>Algeria</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AQ</td>
      <td>AS</td>
      <td>American Samoa</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>274</th>
      <td>-</td>
      <td>-</td>
      <td>World</td>
    </tr>
    <tr>
      <th>275</th>
      <td>YM</td>
      <td>YE</td>
      <td>Yemen</td>
    </tr>
    <tr>
      <th>276</th>
      <td>-</td>
      <td>-</td>
      <td>Zaire</td>
    </tr>
    <tr>
      <th>277</th>
      <td>ZA</td>
      <td>ZM</td>
      <td>Zambia</td>
    </tr>
    <tr>
      <th>278</th>
      <td>ZI</td>
      <td>ZW</td>
      <td>Zimbabwe</td>
    </tr>
  </tbody>
</table>
<p>279 rows × 3 columns</p>
</div>




```python
stations = pd.read_csv("https://raw.githubusercontent.com/PhilChodrow/
PIC16B/master/datasets/noaa-ghcn/station-metadata.csv")
stations["countryID"] = stations["ID"].str[:2] # create a new column with the country code
stations
```


```python
continents = pd.read_csv("continents.csv")
continents
```

Now we need to prepare to load our biggest dataframe: the collection of temperature readings from each station. The data is formatted in an odd way, with each row representing the readings of one station in one year, with columns representing the readings in each month. Adhearing to tidy data standards, we want each row to represent an indivudal data point, so the measurement of one station during one year during one month. Our `prepare_df()` function will reformat this data frame as to stack it, transforming it into tidy format


```python
def prepare_df(df):
    """
    params: temperatures dataframe
    returns: stacked temperature dataframe with country IDs
    """
    df = df.set_index(keys=["ID", "Year"])
    df = df.stack()
    df = df.reset_index()
    df = df.rename(columns = {"level_2"  : "Month" , 0 : "Temp"})
    df["Month"] = df["Month"].str[5:].astype(int)
    df["Temp"]  = df["Temp"] / 100
    return(df)
```

We are going to now load all of this data into a database called `temps.db`. The first three dataframes are relatively simple, because they aren't as large as the temperatures data frame. The temperatures data is so large that it is hosted on multiple urls. We have to access all of these urls, and then add to our database in chucks, as to not consume to much of our memory in the process.


```python
with sqlite3.connect("temps.db") as conn: 
    # add basic tables to db
    stations.to_sql("stations", conn, if_exists = "replace", index = False)
    countries.to_sql("countries", conn, if_exists = "replace", index = False)
    continents.to_sql("continents", conn, if_exists = "replace", index = False)
    # create a list of possible decades
    decades = [(i*10 + 1901,i*10+1910) for i in range(12)] 
    for start, end in decades: # iterate over each decade
        df_iter = pd.read_csv(f"https://raw.githubusercontent.com/PhilChodrow/
        PIC16B/master/datasets/noaa-ghcn/decades/{start}-{end}.csv", chunksize = 100000)
        for df in df_iter: # iterate over each chunk
            df = prepare_df(df)
            df.to_sql("temperatures", conn, if_exists = "append", index = False)
```

Now that we have all of our data loaded, let's query the database to make sure everything loaded correctly.


```python
with sqlite3.connect("temps.db") as conn: 
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")

for result in cursor.fetchall():
    print(result[0])
```

    CREATE TABLE "stations" (
    "ID" TEXT,
      "LATITUDE" REAL,
      "LONGITUDE" REAL,
      "STNELEV" REAL,
      "NAME" TEXT,
      "countryID" TEXT
    )
    CREATE TABLE "temperatures" (
    "ID" TEXT,
      "Year" INTEGER,
      "Month" INTEGER,
      "Temp" REAL
    )
    CREATE TABLE "continents" (
    "Continent_Name" TEXT,
      "Continent_Code" TEXT,
      "Country_Name" TEXT,
      "Two_Letter_Country_Code" TEXT,
      "Three_Letter_Country_Code" TEXT,
      "Country_Number" REAL
    )
    CREATE TABLE "countries" (
    "countryID" TEXT,
      "ISO" TEXT,
      "Name" TEXT
    )
    

It looks like it did!

### Easily Querying the Database

For many users who might be unfamilar with SQL, navigating queries can be rather difficult. So let's make a function that makes it easier for regular python users to access our data. Our function will return temperatures from a specific county, during a specific month in a given range of years


```python
def query_climate_database(country : str, year_begin : int, year_end : int, month : int) -> pd.DataFrame:
    """
    returns a dataframe with temperatures of a given country during a given month in a specified
    time range
    """
    with sqlite3.connect("temps.db") as conn: # connect to database
        cmd = \
            f"""
            SELECT T.id, T.year, T.temp, S.longitude, S.Latitude, C.name, S.NAME
            FROM temperatures T
            LEFT JOIN stations S ON T.id = S.id
            LEFT JOIN countries C ON S.countryID = C.countryID
            WHERE T.year >= {str(year_begin)} AND T.YEAR <= {str(year_end)} 
            AND T.month = {str(month)} AND C.name = ?
            """
        df = pd.read_sql_query(cmd, conn, params = [country])
    return df
```

Let's see if it works!


```python
query_climate_database("India", 1980, 2020, 1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Year</th>
      <th>Temp</th>
      <th>LONGITUDE</th>
      <th>LATITUDE</th>
      <th>Name</th>
      <th>NAME</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>IN001020700</td>
      <td>1980</td>
      <td>23.48</td>
      <td>77.633</td>
      <td>14.583</td>
      <td>India</td>
      <td>PBO_ANANTAPUR</td>
    </tr>
    <tr>
      <th>1</th>
      <td>IN001020700</td>
      <td>1981</td>
      <td>24.57</td>
      <td>77.633</td>
      <td>14.583</td>
      <td>India</td>
      <td>PBO_ANANTAPUR</td>
    </tr>
    <tr>
      <th>2</th>
      <td>IN001020700</td>
      <td>1982</td>
      <td>24.19</td>
      <td>77.633</td>
      <td>14.583</td>
      <td>India</td>
      <td>PBO_ANANTAPUR</td>
    </tr>
    <tr>
      <th>3</th>
      <td>IN001020700</td>
      <td>1983</td>
      <td>23.51</td>
      <td>77.633</td>
      <td>14.583</td>
      <td>India</td>
      <td>PBO_ANANTAPUR</td>
    </tr>
    <tr>
      <th>4</th>
      <td>IN001020700</td>
      <td>1984</td>
      <td>24.81</td>
      <td>77.633</td>
      <td>14.583</td>
      <td>India</td>
      <td>PBO_ANANTAPUR</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3147</th>
      <td>INXLT811965</td>
      <td>1983</td>
      <td>5.10</td>
      <td>88.270</td>
      <td>27.050</td>
      <td>India</td>
      <td>DARJEELING</td>
    </tr>
    <tr>
      <th>3148</th>
      <td>INXLT811965</td>
      <td>1986</td>
      <td>6.90</td>
      <td>88.270</td>
      <td>27.050</td>
      <td>India</td>
      <td>DARJEELING</td>
    </tr>
    <tr>
      <th>3149</th>
      <td>INXLT811965</td>
      <td>1994</td>
      <td>8.10</td>
      <td>88.270</td>
      <td>27.050</td>
      <td>India</td>
      <td>DARJEELING</td>
    </tr>
    <tr>
      <th>3150</th>
      <td>INXLT811965</td>
      <td>1995</td>
      <td>5.60</td>
      <td>88.270</td>
      <td>27.050</td>
      <td>India</td>
      <td>DARJEELING</td>
    </tr>
    <tr>
      <th>3151</th>
      <td>INXLT811965</td>
      <td>1997</td>
      <td>5.70</td>
      <td>88.270</td>
      <td>27.050</td>
      <td>India</td>
      <td>DARJEELING</td>
    </tr>
  </tbody>
</table>
<p>3152 rows × 7 columns</p>
</div>



### Answering Questions with Visualizations

Now that we have an easy way to access the data, we can start to create visualizations. 

> How does the average yearly change in temperature vary within a given country?

We can define "average yearly change" by the coeficient generated by creating a Linear Regression Model. We want to see change *by station* within a country, so we will need to use the `pd.DataFrame.groupby()` method and apply a function to generate a coeficient from each chunk of the dataframe. 

Once we generate this dataframe, we will then put into a `plotly.express` mapbox to visualize our data


```python
from sklearn.linear_model import LinearRegression
import plotly.express as px

def coef(data_group):
    """
    params: a dataframe based on groupby data
    returns: coefficent generated by Linear Regression
    """
    x = data_group[["Year"]] # 2 brackets because X should be a df
    y = data_group["Temp"] # 1 bracket because y should be a series
    LR = LinearRegression()
    LR.fit(x, y)
    return LR.coef_[0]

def temperature_coefficient_plot(country : str, year_begin : int, year_end : int,
 month : int, min_obs: int, **kwargs):
    """
    generates a map plot of stations within a given country colored by 
    the coefficent of yearly change in average temperature for each station
    """
    # access database and get table
    df = query_climate_database(country, year_begin, year_end, month) 
    # add a column with the number of observations
    df["obs"] = df.groupby("Name")["Temp"].transform(len)  for each row
    df = df[df["obs"] >= min_obs] # filter datatrame to only have minimum observations

    # calculate coefficients
    coefs = df.groupby(["NAME"]).apply(coef)
    coefs = coefs.reset_index()
    coefs = coefs.rename({0:'Estimated Yearly Increase (C)'}, axis = 1)
    df = pd.merge(df, coefs, on = "NAME")
    # create map
    fig = px.scatter_mapbox(df, lat = "LATITUDE", lon = "LONGITUDE", 
              hover_name = "NAME", 
              color = "Estimated Yearly Increase (C)", **kwargs)


    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig
```


```python
color_map = px.colors.diverging.RdGy_r
fig = temperature_coefficient_plot("India", 1980, 2020, 1, 
                                   min_obs = 10,
                                   zoom = 2,
                                   mapbox_style="carto-positron",
                                   color_continuous_scale=color_map)
fig.show()
```
{% include temp_coef.html %}


> How do average temperatures during the year change as we get closer to the present for each country and continent?

Not only is `plotly` inteactive, but it also has an animation feature. So we can make a psudeo 3D plot, with the x and y axis being year and average temperature per country respectively, and the z axis be a given year, which we can animate through.


```python
def animated_tempChange_perYear(year_begin : int, year_end : int, **kwargs):
    """
    returns an animated figure showing the change in average 
    temperature across a given year for each country 
    """
    # query data base to get aggregated average temperatures by column, name, year, and month
    with sqlite3.connect("temps.db") as conn: 
        cmd = \
            f"""
            SELECT T.id, T.year, S.longitude, S.Latitude, C.name, S.NAME,
            T.MONTH,D.Continent_Code, AVG(T.temp) "meanTemp"
            FROM temperatures T
            LEFT JOIN stations S ON T.id = S.id
            LEFT JOIN countries C ON S.countryID = C.countryID
            LEFT JOIN continents D ON C.ISO = D.Two_Letter_Country_Code
            WHERE T.year >= {year_begin} AND T.YEAR <= {year_end}
            GROUP BY D.Continent_Code, C.name, T.Year, T.Month
            """
        df = pd.read_sql_query(cmd, conn)
    df = df.dropna()
    fig = px.scatter(df, x="Month", y ="meanTemp", 
            color = "Continent_Code", 
            animation_frame = "Year",  
            template="plotly_dark", **kwargs)

    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig

```


```python
fig = animated_tempChange_perYear(1990, 2010)
fig.show()
```
{% include animated_temp.html %}


> How much does each country contribute to its continents overall average temperature in a certain range of years?

Some countries might have more climate infrastructure than others, and thus may disproportionately contribute to a continents mean temperature in a given set of years. Within each continent, we want to visualize what percentage of stations are in which country, and the average temperature within each of those countries


```python
def country_contributions(year_begin : int, year_end : int, **kwargs):
    """
    returns tree map of countries in continents sized by total 
    amount of stations and colored by        average temperature 
    """
    with sqlite3.connect("temps.db") as conn: 
        cmd = \
            f"""
            SELECT T.id, T.year, S.longitude, S.Latitude, C.name, S.NAME,
            T.MONTH,D.Continent_Code, COUNT(T.temp) "counts", AVG(T.temp) "meanTemp"
            FROM temperatures T
            LEFT JOIN stations S ON T.id = S.id
            LEFT JOIN countries C ON S.countryID = C.countryID
            LEFT JOIN continents D ON C.ISO = D.Two_Letter_Country_Code
            WHERE T.year >= {year_begin} AND T.YEAR <= {year_end}
            GROUP BY C.name
            """
        df = pd.read_sql_query(cmd, conn)
    df = df.dropna()
    fig = px.treemap(df, path=['Continent_Code', 'Name'], values='counts',
                    color='meanTemp', template="plotly_dark", color_continuous_scale='icefire', 
                    title="Highest Contributions toward Continent Mean Temperature", 
                    hover_data=['meanTemp'])
    
    # fig.update_layout(margin={"r":0,"t":3,"l":0,"b":0})
    return fig
```


```python
fig = country_contributions(2000, 2010)
fig.show()
```
{% include country_contributions.html %}

