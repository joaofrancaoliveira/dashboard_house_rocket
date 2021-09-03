import streamlit            as st
import numpy                as np
import pandas               as pd
import matplotlib.pyplot    as plt
import plotly_express       as px
import folium
import geopandas

from streamlit_folium   import folium_static
from folium.plugins     import MarkerCluster
from datetime           import datetime

st.set_page_config( layout = 'wide' )

@st.cache( allow_output_mutation = True )
def get_data(path):
    pd.set_option('float_format', '{:.3f}'.format)
    data = pd.read_csv(path)
    pd.set_option('float_format', '{:.3f}'.format)
    return data

@st.cache( allow_output_mutation = True )
def get_geofile(url):
    geofile = geopandas.read_file(url)
    return geofile

def data_transform(data):
    # Convert object to date
    data['date'] = pd.to_datetime(data['date'])
    
    # Descriptive statistics
    num_attributes = data.select_dtypes(include=['int64', 'float64'])
     
    # Central tendency - media, mediana
    pd.set_option('display.float_format', lambda x: '%.3f' %x)
    media = pd.DataFrame( num_attributes.apply(np.mean, axis=0))
    mediana = pd.DataFrame( num_attributes.apply( np.median, axis=0))
    
    # Dispersion - Std, Min, Max
    std = pd.DataFrame(num_attributes.apply(np.std, axis=0))
    min_ = pd.DataFrame(num_attributes.apply(np.min, axis=0))
    max_ = pd.DataFrame(num_attributes.apply(np.max, axis=0))
    
    statistics_descriptive = pd.concat([max_, min_, media, mediana, std], axis=1).reset_index()
    statistics_descriptive.columns = ['attributes', 'maximo', 'minimo', 'media', 'mediana', 'std']
    
    data['dormitory_type'] = 'NA'
    
    for i in range(len(data)):
        if data.loc[i, 'bedrooms'] == 1:
            data.loc[i, 'dormitory_type'] = 'studio'
        elif data.loc[i, 'bedrooms'] == 2:
            data.loc[i, 'dormitory_type'] = 'apartment'
        else:
            data.loc[i, 'dormitory_type'] = 'house'
    
    quartis = np.percentile(data.price, [25, 50, 75])
    
    data['level'] = 'NA'
    for i in range(len(data)):
        if data.loc[i, 'price'] <= quartis[0]:
            data.loc[i, 'level'] = 0
        elif (data.loc[i, 'price'] > quartis[0]) & (data.loc[i, 'price'] <= quartis[1]):
            data.loc[i, 'level'] = 1
        elif (data.loc[i, 'price'] > quartis[1]) & (data.loc[i, 'price'] <= quartis[2]):
            data.loc[i, 'level'] = 2
        else:
            data.loc[i, 'level'] = 3
    
    data['m2'] = data['sqft_lot'] * 0.3048
    data['price_m2'] = data['price'] / data['m2']
    
    return data, statistics_descriptive

def overview_data(data):
    
    # ===============
    # Data Overview
    # ===============

    f_attributes = st.sidebar.multiselect( 'Enter Columns', data.columns)
    f_zipcodes = st.sidebar.multiselect( 'Enter Zipcode', data['zipcode'].unique())

    st.title('Data Overview')

    # Attributes (colunas) + Zipcode (linhas)
    if (f_attributes != []) and (f_zipcodes != []):
        data = data.loc[data['zipcode'].isin(f_zipcodes), f_attributes]

    # Attributes (Selecionar Colunas)
    elif (f_attributes != []) and (f_zipcodes == []):
        data = data.loc[:, f_attributes]

    # Zipcode (Selecionar Linhas)
    elif (f_attributes == []) and (f_zipcodes != []):
        data = data.loc[data['zipcode'].isin(f_zipcodes), :]

    # Selecionar Tudo
    else:
        data = data.copy()
        

    # Filtering some information by zipcode
    # Total number of properties
    tnp = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()

    # Avarage Price
    avp = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()

    # Living room size average 
    lrsa = data[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()

    # Avarege price (m²)
    apm2 = data[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()

    # Merge of Filtering
    m1 = pd.merge(tnp, avp, on='zipcode', how='inner')
    m2 = pd.merge(m1, lrsa, on='zipcode', how='inner')
    zipcode_filters = pd.merge(m2, apm2, on='zipcode', how='inner')

    zipcode_filters.columns = ['Zipcode', 'Total Houses', 'Price', 'Sqrt Living', 'Price/M²']


    st.header('Data')
    st.dataframe( data )
    c1, c2 = st.beta_columns((1, 1))
    c1.header('Average Values')
    c1.dataframe(zipcode_filters, height=200)
    c2.header('Descriptive Analysis')
    c2.dataframe(statistic_desc, height=200)
    
    return None

def portfolio_density(data, geofile):
    # =================
    # Portfolio Density 
    # =================

    st.title('Region Overview')

    c1, c2 = st.beta_columns((1,1))
    c1.header('Portfolio Density')

    #df = data.sample(100)
    df = data

    # Base Map - Folium
    density_map = folium.Map(
        location=[data['lat'].mean(), data['long'].mean()],
        default_zoom_start = 15)

    marker_cluster = MarkerCluster().add_to(density_map)
    for name, row in df.iterrows():
        folium.Marker([row['lat'], row['long']],
                    popup='Sold ${0} on: {1}. Features: {2} sqft, {3} bedrooms, {4} bathrooms, year built: {5}'.format(row['price'],
                                                                                                                        row['date'],
                                                                                                                        row['sqft_living'],
                                                                                                                        row['bedrooms'],
                                                                                                                        row['bathrooms'],
                                                                                                                        row['yr_built'])).add_to(marker_cluster)

    with c1:
        folium_static(density_map)
        

    # Region Price Map
    c2.header('Price Density')

    df = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df.columns = ['ZIP', 'PRICE']

    #df = data.sample(100)

    geofile = geofile[geofile['ZIP'].isin(df['ZIP'].tolist())]

    region_price_map = folium.Map(location=[data['lat'].mean(),
                                            data['long'].mean()],
                                default_zoom_start=15 )

    region_price_map.choropleth( data = df,
                                geo_data = geofile,
                                columns=['ZIP', 'PRICE'],
                                key_on='feature.properties.ZIP',
                                fill_color='YlOrRd',
                                fill_opacity= 0.7,
                                line_opacity= 0.2,
                                legend_name='AVG PRICE')

    with c2:
        folium_static(region_price_map)

    return None

def commercial_distribution(data):
    # ==================================================
    # Distribuição dos imóveis por categorias comerciais
    # ==================================================

    st.sidebar.title('Commercial Options')
    st.title( 'Commercial Attributes')

    data['date'] = pd.to_datetime(data['date'], ).dt.strftime('%Y-%m-%d')

    # Filters
    min_yr_built = int(data['yr_built'].min())
    max_yr_built = int(data['yr_built'].max())

    st.sidebar.subheader('Select Max Year Built')
    f_yr_built = st.sidebar.slider('Year Built',
                                min_yr_built,
                                max_yr_built,
                                min_yr_built)

    # Average Price per Year
    st.header('Average Price per Year Built')
    # Data Selection
    df = data.loc[data['yr_built'] < f_yr_built]
    df = df[['yr_built', 'price']].groupby('yr_built').mean().reset_index()
    # Plot
    fig = px.line(df, x='yr_built', y='price')
    st.plotly_chart(fig, use_container_width=True)

    # Average Price per Day
    st.header('Average Price per Day of Acquisition')
    st.sidebar.subheader('Select Max Date')

    # Filters
    min_date = datetime.strptime(data['date'].min(), '%Y-%m-%d')
    max_date = datetime.strptime(data['date'].max(), '%Y-%m-%d')

    # Data Filtering
    data['date'] = pd.to_datetime(data['date'])
    f_date = st.sidebar.slider('Date', min_date, max_date, min_date)
    df = data.loc[data['date'] < f_date]


    # Plot
    df = data[['date', 'price']].groupby('date').mean().reset_index()
    fig = px.line(df, x='date', y='price')
    st.plotly_chart(fig, use_container_width=True)


    # ==========
    # Histograma 
    # ==========

    st.header('Price Distribution')
    st.sidebar.subheader('Select Max Price')

    # Filter
    min_price = int(data['price'].min())
    max_price = int(data['price'].max())
    avg_price = int(data['price'].mean())

    # Data Filter
    f_price = st.sidebar.slider('Price', min_price, max_price, avg_price)
    df = data.loc[data['price'] < f_price]

    # Plot
    fig = px.histogram(df, x='price', nbins=50)
    st.plotly_chart(fig, use_container_width=True)
    
    return None
def attributes_distribution(data):
    # ==============================================
    # Distribuição dos imóveis por categoria físicas
    # ==============================================
    st.sidebar.title('Attributes Options')
    st.title('House Attributes')

    # Filters
    f_bedrooms = st.sidebar.selectbox('Max number of bedrooms', sorted(set(data['bedrooms'].unique())))
    f_bathrooms = st.sidebar.selectbox('Max number of bathrooms', sorted(set( data['bathrooms'].unique())))
    
    # Filters
    f_floors = st.sidebar.selectbox('Max number of Floor',
                                    sorted(set(data['floors'].unique())))
    f_waterview = st.sidebar.checkbox('Only Houses with Water View')
    
    # House per water view
    if f_waterview:
        df = data[data['waterfront'] == 1]
    else:
        df = data.copy()

    c1, c2 = st.beta_columns(2)

    # House per bedrooms
    c1.header('Houses per Bedrooms')
    df = df[data['bedrooms'] < f_bedrooms]
    fig = px.histogram(df, x='bedrooms', nbins=19)
    c1.plotly_chart(fig, use_container_width=True)

    # House per bathrooms
    c2.header('Houses per Bathrooms')
    df = df[data['bathrooms'] < f_bathrooms]
    fig = px.histogram(df, x='bathrooms', nbins=19)
    c2.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.beta_columns(2)

    # House per floors
    c1.header('Houses per Floor')
    df = df[data['floors'] < f_floors]

    # Plot
    fig = px.histogram(df, x='floors', nbins=8)
    c1.plotly_chart(fig, use_container_width=True)

    c2.header('Water View')
    fig = px.histogram(df, x='waterfront')
    c2.plotly_chart(fig, use_container_width=True)
    return None



if __name__ == '__main__':
    #ETL
    # Extraction
    path = 'kc_house_data.csv'
    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
    
    data = get_data(path)
    geofile = get_geofile(url)
    
    
    # Transformation
    data, statistic_desc = data_transform(data)
    overview_data(data)
    portfolio_density(data, geofile)
    commercial_distribution(data)
    attributes_distribution(data)
    # Loading


