import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime 
#import lightgbm as lgb
import altair as alt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# data available from till 2020-02-29
start_date = '2017-01-01'
# get prediction max to 2020-05-01
end_date = '2020-05-01'
# last_hist_trans 
last_hist_trans = '2020-02-29'
# mandatory list of columns
trans_column_list = ['date_id', 'product_id', 'department_id', 'section_id', 'family_id',
       'subfamily_id', 'country_id', 'regular_price', 'promotion_price',
       'item_cost', 'quantity', 'promo_mechanics_id', 'promotion_flag']


def plot_feature_importances(feature_order, model):
    features = feature_order
    importances = model.feature_importances_
    data = pd.DataFrame({'importance': importances,
                     'feature': feature_order})
    c1= alt.Chart(data).mark_bar().encode(
        x=alt.X('importance:Q'),
        y=alt.Y('feature:N',  sort='-x'),
        color=alt.Color('importance:Q', title='Importance by color')
    ).transform_filter(
        alt.FieldGTPredicate(field='importance', gt=0)
    ).properties(
    title='Feature Importances'
)
    st.altair_chart(c1, use_container_width=True)


def r2_rmse(g):
    """Returns R2 score."""
    r2 = r2_score(g['y'], g['lgb'])
    return r2
    
def evaluate_mean_r2(x_test, y_test, y_pred, print_metrics=True):
    """Returns average R^2 score per product."""
    x_test['y'] = y_test
    x_test['lgb'] = y_pred

    r2 = []
    for p in x_test.product_id.unique():
        r2.append(r2_rmse(x_test.loc[x_test['product_id'] == p, :]))
    
    #st.write(r2, x_test.product_id.unique)
    r2 = {'Mean R^2 per product': round(np.mean(r2), 2)}
    if print_metrics:
        print(r2)
    else:
        return r2


def evaluate_prediction(y_test, y_pred, print_metrics=True):
    """Returns/print main metrics."""    
    metrics = {'MAE': round(mean_absolute_error(y_test, y_pred), 2),
               'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
               'R^2': round(r2_score(y_test, y_pred), 2)
    }
    if print_metrics:
        for k, v in metrics.items():
            print(f'{k}: {v:.3f}')
    else:
        return metrics
    
def get_predictions(features, model, encoder):
    
    features_1 = features.copy()
    features_1 = {
        department: features_1[
            features_1['department_id'] == department
            ] for department in features_1['department_id'].unique()
    }
    for department, feature_dt in features_1.items():
        country_key = int(feature_dt['country_id'].iloc[0])

        # predict qty with promo
        features_1[department].loc[:, 'product_id'] = encoder.transform(features_1[department].loc[:, 'product_id'])
        features_1[department]['quantity_with_promo'] = model.predict(features_1[department].loc[:, feature_order]).astype(int)
        # predict qty without promo (regular=promo price, depth = 1, mechanics = 0, promo flag = 0)
        if 'promotion_price' in features.columns:
            features_1[department]['promotion_price_fallback'] = features_1[department]['promotion_price']
        if 'promo_mechanics_id' in features.columns:
            features_1[department]['promo_mechanics_id_fallback'] = features_1[department]['promo_mechanics_id']
            features_1[department]['promo_mechanics_id'] = 0

        if 'promotion_price' in features.columns:
            if 'regular_price' in features.columns:
                features_1[department]['promotion_price'] = features_1[department]['regular_price']

        if 'promotion_depth' in features.columns:
            features_1[department]['promotion_depth'] = 1
        if 'promotion_flag' in features.columns:
            features_1[department]['promotion_flag'] = 0
        features_1[department]['quantity_without_promo'] = model.predict(features_1[department].loc[:, feature_order]).astype(int)
           
        if 'promotion_price' in features.columns:
            features_1[department]['promotion_price'] = features_1[department]['promotion_price_fallback']
        if 'promo_mechanics' in features.columns:  
            features_1[department]['promo_mechanics_id'] = features_1[department]['promo_mechanics_id_fallback']

        # ecnoder inverse
        features_1[department].loc[:, 'product_id'] = encoder.inverse_transform(features_1[department].loc[:, 'product_id'])
        del model

    features_1 = pd.concat(list(features_1.values()))
    return features_1

def aggregate_predictions(features):
    predictions = features[['product_id', 'quantity_with_promo', 'quantity_without_promo']]
    predictions = predictions.groupby('product_id').sum()
    prices = features[['product_id', 'item_cost', 'regular_price', 'promotion_price']].groupby('product_id').mean()
    res = pd.merge(left=predictions, right=prices, on='product_id').reset_index()
    res['quantity_with_promo'] = res['quantity_with_promo']
    #res.loc[res['quantity_with_promo'] <= 0, 'quantity_with_promo'] = 0
    res['quantity_without_promo'] = res['quantity_without_promo']
    #res.loc[res['quantity_without_promo'] <= 0, 'quantity_without_promo'] = 0
    
    return res


def calculate_KPIS(df):
    promo_sales = df['promotion_price'] * df['quantity_with_promo']
    reg_sales = df['regular_price'] * df['quantity_without_promo']
    df['incremental_sales'] = promo_sales - reg_sales
    df['baseline_sales'] = reg_sales
    promo_mrg = (df['promotion_price'] - df['item_cost']) * df['quantity_with_promo']
    reg_mrg = (df['regular_price'] - df['item_cost']) * df['quantity_without_promo']
    df['incremental_margin'] = promo_mrg - reg_mrg
    df['baseline_margin'] = reg_mrg

    return df

def get_final_results(promo_products, features, discount, model, product_encoder, promo):
    # generate features for promo
    if promo:
        features['promo_mechanics_id'] = 1
        features['promotion_price'] = features['regular_price'] * (100 - discount) / 100
        features['promotion_depth'] = features['promotion_price'] / features['regular_price']
        features['promotion_flag'] = (features['promo_mechanics_id'] > 0).astype(int)
    
        pred = get_predictions(features, model, product_encoder)

        aggregated_predictions = aggregate_predictions(pred)
    
        return calculate_KPIS(aggregated_predictions), pred
    
    if not promo:
        pred = get_predictions(features, model, product_encoder)
        return None, pred


@st.cache(suppress_st_warning=True)
def get_model_for_country_dept(features_dt, params, product_encoder):
    # get model for the given dataframe

    features_dt['product_id'] = product_encoder.transform(features_dt['product_id'])

    x_train = features_dt.loc[
        (
            (features_dt['is_historical_data'] == 1)
        ),
        :
    ].drop(columns=['quantity']).copy()
    y_train = features_dt.loc[
        (
            (features_dt['is_historical_data'] == 1)
        ),
        'quantity'
    ].copy()

    lgbm = lgb.LGBMRegressor(
        **params
    ) 
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
    
    lgbm.fit(
        x_train,
        y_train,
        categorical_feature = [val for val in features_dt.columns if val in ['product_id', 'department_id', 'section_id', 'family_id', 'subfamily_id', 'promo_mechanics_id']]
    )

    st.write('Train scores')
    y_pred = lgbm.predict(x_train)
    st.write(evaluate_prediction(y_train, y_pred, False))
    st.write(evaluate_mean_r2(x_train, y_train, y_pred, False))

    st.write('Test scores')
    y_pred_test = lgbm.predict(x_test)
    st.write(evaluate_prediction(y_test, y_pred_test, False))
    st.write(evaluate_mean_r2(x_test, y_test, y_pred_test, False))    
    
    features_dt['product_id'] = product_encoder.inverse_transform(features_dt['product_id'])

    return lgbm


@st.cache
def get_features_products(features_dt, events, start_date, end_date):
    # first, create products and unique attributes data frame for efficient joins
    unique_attributes_for_each_product = [
        'product_id', 
        'department_id', 
        'section_id', 
        'family_id', 
        'subfamily_id', 
        'country_id'
    ]
    # make a copy, so we work with clean data and don't edit by referrence
    features_full = features_dt.loc[:, unique_attributes_for_each_product].drop_duplicates().copy()

    # Current Vertica representation of following attributes is not entirely applicable for trainig, 
    # as dept/section/family/subfamily are hierarchical attributes, and we need to transform them 
    # before feeding to the model to have unique IDs
    features_full['department_id'] = features_full['department_id'].astype('str')
    features_full['section_id'] = features_full['department_id'] + '-' + features_full['section_id'].astype('str') 
    features_full['family_id'] = features_full['section_id'] + '-' + features_full['family_id'].astype('str') 
    features_full['subfamily_id'] = features_full['family_id'] + '-' + features_full['subfamily_id'].astype('str')

    # Do the encoding, as the model consumes only integers, and after previous manipulations we have unique strings
    le = LabelEncoder()
    #le.fit(features_full['department_id'])
    #features_full['department_id'] = le.transform(features_full['department_id'])
    le.fit(features_full['section_id'])
    features_full['section_id'] = le.transform(features_full['section_id'])
    le.fit(features_full['family_id'])
    features_full['family_id'] = le.transform(features_full['family_id'])
    le.fit(features_full['subfamily_id'])
    features_full['subfamily_id'] = le.transform(features_full['subfamily_id'])
    features_full['department_id'] = features_full['department_id'].astype(int)

    # Column jf stands for join field, dummy value to use pandas merge as a cross join
    features_full['jf'] = 0

    # Cross join merge between unique product features and all time periods
    features_full = pd.merge(
        left=pd.DataFrame({'date_id': pd.date_range(start_date, end_date), 'jf': 0}), 
        right=features_full, 
        on='jf'
    ).drop(columns=['jf'])

    # Historical periods only are used in model training
    features_full['is_historical_data'] = 0

    for product_id in features_full.product_id.unique():
        start_historic_date = features_dt[features_dt['product_id']==product_id].date_id.min()
        end_historic_date = features_dt[features_dt['product_id']==product_id].date_id.max()
        features_full.loc[
            ((features_full['date_id'] >= start_historic_date) 
             & (features_full['date_id'] <= end_historic_date) 
             & (features_full['product_id'] == product_id)), 
            'is_historical_data'
        ] = 1

  
    features_full = pd.merge(
        left=features_full, 
        right=features_dt.loc[:, ['date_id', 'product_id', 'promotion_price', 'regular_price', 'item_cost', 'quantity', 'promo_mechanics_id', 'promotion_flag']],
        on=['date_id', 'product_id'],
        how='left'
    )

    # Filling missing values
    # TODO: reivew and documment missing values treatment
    features_full['is_historical_data'] = features_full['is_historical_data'].fillna(0).astype('int')
    features_full['promo_mechanics_id'] = features_full['promo_mechanics_id'].fillna(0).astype('int')
    features_full['quantity'] = features_full['quantity'].fillna(0).astype('int')
    features_full['promotion_price'] = features_full['promotion_price'].fillna(0).astype('int')
    features_full['regular_price'] = features_full.loc[:, ['product_id', 'regular_price']].groupby(
        'product_id').fillna(method='ffill').fillna(method='bfill')['regular_price']
    features_full['item_cost'] = features_full.loc[:, ['product_id', 'item_cost']].groupby(
        'product_id').fillna(method='ffill').fillna(method='bfill')['item_cost']

    # Promotion price 0 is a fallback value in Vertica, actual promoiton price in such cases should match 
    features_full.loc[
        features_full['promotion_price'] == 0, 
        'promotion_price'
    ] = features_full.loc[
        features_full['promotion_price'] == 0, 
        'regular_price'
    ]

    features_full['year'] = features_full['date_id'].dt.year
    features_full['month'] = features_full['date_id'].dt.month
    features_full['week'] = features_full['date_id'].dt.week
    features_full['day'] = features_full['date_id'].dt.day
    features_full['weekday'] = features_full['date_id'].dt.dayofweek

    features_full.sort_values(by=['product_id', 'date_id'], inplace=True)
    # Creating rolling average features
    periods = [14, 28, 61, 92, 182, 336, 365, 395]
    for period in periods:
        for p in features_full.product_id.unique():
            features_full.loc[(features_full.product_id == p) &
                              (features_full.date_id <=last_hist_trans), 
                              'mean_days_sales_{}'.format(period)
                             ] = features_full.loc[(features_full.product_id == p) &
                                                   (features_full.date_id <=last_hist_trans), 
                                                   'quantity'
                                                  ].rolling(period).mean().shift(1).fillna(method='ffill').fillna(method='bfill').values

    features_full.update(features_full.sort_values(["product_id", "date_id"]).groupby("product_id").ffill().bfill())
     

    features_full['promotion_depth'] = features_full['promotion_price'] / features_full['regular_price']
    features_full.loc[features_full['regular_price'] == 0, 'promotion_depth'] = 0
    features_full['promotion_flag'] = (features_full['promo_mechanics_id'] > 0).astype(int)

    # get products 
    products = features_full.product_id.unique()
    country_id = features_full.country_id.unique()[0]
    department_id = features_full.department_id.unique()[0]

    # set event flag
    events = events.loc[events['country_id']==country_id, ['date_id','event_flag']]
    features_full = features_full.merge(events, on ='date_id', how='left')
    features_full['event_flag'] = features_full['event_flag'].fillna(0).astype(int)
    
    le.fit(features_full['product_id'])    
    return features_full, le, products, country_id, department_id


@st.cache(suppress_st_warning=True)
def get_trans_data(uploaded_file):
    try:
        features_dt = pd.read_csv(uploaded_file, parse_dates=['date_id'])
    except:
        st.error('Wrong input file')
        raise

    if len([val for val in trans_column_list if val not in features_dt.columns]) != 0:
        st.error('Some data is missing or in a wrong format, mandatory columns for input: {}'.format(trans_column_list))
        raise
    return features_dt


def get_hyperparams_for_model():
    st.sidebar.subheader('Select hyperparameters for your model:')

    params = {}
    boosting_type = st.sidebar.radio("Select boosting type",
                    ('gbdt', 'dart', 'goss'))
    params['boosting_type'] = boosting_type

    objective = st.sidebar.radio("Select objective function",
                     ('quantile', 'regression_l1', 'regression_l2'))
    params['objective'] = objective

    n_estimators = st.sidebar.slider('Select number of estimators',
                           min_value=100,
                           max_value=1000,
                           value=1000, 
                           step=100, 
                           format='%d')
    params['n_estimators'] = n_estimators

    learning_rate = st.sidebar.number_input('Insert a learning rate',
                                min_value=0.001,
                                value=0.01, 
                                step=0.01,
                                format='%.3f')
    params['learning_rate'] = learning_rate

    if objective=='quantile':
        alpha = st.sidebar.number_input('Insert an alpha',
                            min_value=0.01,
                            max_value=1.0,
                            value=0.50,
                            step=0.05,
                            format='%.2f')
        params['alpha'] = alpha
        
    if objective=='regression_l1':
        reg_alpha = st.sidebar.number_input('Insert a reg_alpha',
                            min_value=0.1,
                            max_value=1.0,
                            value=0.3,
                            step=0.1,
                            format='%.1f')
        params['reg_alpha'] = reg_alpha


    if objective=='regression_l2':
        reg_lambda = st.sidebar.number_input('Insert a reg_lambda',
                             min_value=0.1,
                             max_value=1.0,
                             value=0.3,
                             step=0.1,
                             format='%.1f')
        params['reg_lambda'] = reg_lambda

    monotone_constraints = st.sidebar.checkbox('Add monotone_constraints',
                                  value=True)
    if monotone_constraints:
        monotone_constraints = list()
        for x in feature_order:
            if x in ['promotion_price', 'regular_price', 'promotion_depth']:
                monotone_constraints.append(-1)
            elif x in ['promotion_flag']:
                monotone_constraints.append(1)
            else:
                monotone_constraints.append(0) 
    
        params['monotone_constraints'] = monotone_constraints
        
    return params


def get_country_name(n):
     return 'Country ' + str(n)
    
    
def get_department_name(n):
     return 'Department ' + str(n)
    
    
def get_product_name(n):
    return 'Product ' + str(n)


def get_percentage_name(n):
    return str(n) + '% '



def plot_prediction(df, promoted_products, product_encoder, start_date, end_date, promo=False, discount=0):
    # get prediction
    df['date_id'] = pd.to_datetime(df[['year', 'month', 'day']]).dt.date
    df_pred = df.loc[
        df['product_id'].isin(promoted_products) &
        (df['date_id']>=start_date) &
        (df['date_id']<=end_date)
        , feature_order
    ].copy()
    
    totals, prediction = get_final_results(promoted_products, df_pred, discount, lgbm, product_encoder, promo)
    prediction['date_id'] = pd.to_datetime(prediction[['year', 'month', 'day']]).dt.date

    pred_data = prediction.loc[:, ['date_id', 'product_id', 'quantity_without_promo', 'quantity_with_promo']]
    
    chart_data = df.loc[
        (df['product_id'].isin(promoted_products)) &
        (df['date_id']<start_date) &
        (df['date_id']>=datetime.date(2019, 12, 1)),
        ['date_id', 'product_id', 'quantity']
    ].copy()

    # plot data
    colors={}
    fig, ax = plt.subplots(figsize=(16,8))
        #plot hist data
    for p in promoted_products:
        chart_data[chart_data['product_id']==p].plot(
                x = 'date_id',
                y=['quantity'],
                ax=ax,
                linestyle='dotted',
                grid=True,
                fontsize=16,
                rot=15
        )
    
        colors[p]=plt.gca().lines[-1].get_color()

    # plot qty WITHOUT promo
    for p in promoted_products:
        pred_data[pred_data['product_id']==p].plot(
            x = 'date_id',
            y=['quantity_without_promo'],
            ax=ax,
            grid=True,
            fontsize=16,
            rot=15,
            color = colors[p]
    )

    if promo:
        # plot qty WITH promo
        for p in promoted_products:
                pred_data[pred_data['product_id']==p].plot(
                x = 'date_id',
                y=['quantity_with_promo'],
                ax=ax,
                grid=True,
                fontsize=16,
                linestyle='dashdot',
                rot=15,
                color = colors[p]
                )
    # add legend, title
    ax.legend([get_product_name(val) for val in promoted_products])
    plt.title('Product quantity sold over time', fontsize=20)
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('Quantity', fontsize=16)

    st.pyplot()
        
    # return incr totals for promo simulation only
    if promo:
        return totals


def get_total_incrementals(df, promoted_products, product_encoder, start_date, end_date, discounts_list, promo):
    # get prediction
    df['date_id'] = pd.to_datetime(df[['year', 'month', 'day']]).dt.date
    df_pred = df.loc[
        df['product_id'].isin(promoted_products) &
        (df['date_id']>=start_date) &
        (df['date_id']<=end_date)
        , feature_order
    ].copy()
    
    all_totals = pd.DataFrame()
    for discount in discounts_list:
        totals, _ = get_final_results(promoted_products, df_pred, discount, lgbm, product_encoder, promo)    
        totals['discount'] = discount
        all_totals = pd.concat([all_totals, totals])
    
    return all_totals
    

def plot_total_incrementals(totals):
    points = alt.Chart(totals).mark_circle(size=200).encode(
        x=alt.X('incremental_sales:Q'),
        y=alt.Y('incremental_margin:Q'),
        color=alt.Color('discount:N'),
        tooltip=['product_id:N', 'incremental_sales:Q', 'incremental_margin:Q', 'discount:N']
    ).interactive(
    ).properties(
        title='Impact of the discount value'
    )

    st.altair_chart(points, use_container_width=True)    


@st.cache(suppress_st_warning=True)
def get_event_dates(uploaded_file):
    try:
        df_events = pd.read_csv(uploaded_file, parse_dates = ['date_id'])
        df_events['event_flag'] = 1
    except:
        st.error('wrong input file')
        raise
        
    if ('date_id' not in df_events.columns or 'country_id' not in df_events.columns):
        st.error('Some data is missing or in a wrong format, mandatory columns for input event calendar: date_id and country_id')
        raise
    return df_events

st.title('DEMAND FORECAST DEMO')
st.markdown("""You are going to develop machine learning predictive model in a few clicks!
At the very beginning you should load transaction data.""")

st.markdown("""Mandatory fileds needed as an input: **date_id, country_id, product_id, department_id, section_id, 
family_id, subfamily_id, regular_price, promotion_price, item_cost, promo_mechanics_id, promotion_flag, quantity**. Further instructions you will find below.""")

# get file with data
uploaded_file = st.sidebar.file_uploader("Choose a file with tranactions", type=['csv'])

uploaded_calendar = st.sidebar.file_uploader("Choose a file with event dates", type=['csv'])

if uploaded_file is not None and uploaded_calendar is not None:
    # get events 
    events = get_event_dates(uploaded_calendar)
    # get trans data
    features_dt = get_trans_data(uploaded_file)
    st.subheader('Loaded data:')
    st.write(features_dt)
    # get features, products
    df, product_encoder, products, country_id, department_id = get_features_products(features_dt, events, start_date, end_date)

    # show calculated features
    st.subheader('Take a look on features generated from the input data that can be used for the model training:')
    # select features for model training
    exclude_from_features = ['quantity', 'date_id']
    st.dataframe(df.drop(columns=exclude_from_features))
    
    features_names = [value for value in df.columns if value not in exclude_from_features] 
    selected_features = st.multiselect(
        'Select features:',
        features_names,
        default = features_names
    )

    # construct features list (selected by user + obligatory- such as product id, day , etc.)
    feature_order = []
    feature_all = []
    feature_all = selected_features + ['quantity']
    feature_order = selected_features
    for f in ['product_id', 'year', 'month', 'day', 'country_id', 'department_id', 'is_historical_data']:
        if f not in feature_order:
            feature_order.append(f)
            feature_all.append(f)        
    
    # filter out NOT SELECTED by used features
    df = df.loc[:,feature_all] 
    
    # get params
    params = get_hyperparams_for_model()

    # model training
    st.write("""Model trainded with the following hyperparameters: """, params)
    lgbm = get_model_for_country_dept(df, params, product_encoder)

    # plot feature importances of the model generated above 
    plot_feature_importances(feature_order, lgbm)

    st.subheader('Select products and period for simulation')
    # get input data for simulation
    promoted_products = st.multiselect(
        'Select products',
         products.tolist(),
         default = products[0],
        format_func=get_product_name
    )

    start_date = st.date_input(
         "Get prediction from date",
         datetime.date(2020, 3, 1))

    end_date = st.date_input(
         "Get prediction to date",
         datetime.date(2020, 3, 14))

    duration = end_date - start_date
    if (duration.days > 20):
        st.warning('You have selected a date period with a duration of more than 20 days, to get more accurate prediction please use shorter intervals.')
    
    if (len(promoted_products) != 0 and start_date <= end_date):
        plot_prediction(df, promoted_products, product_encoder, start_date, end_date, promo=False)
    else:
        st.warning('please select at least one product and "start date" smaller than "end date"')
   
    st.title('USE CASE: PROMO CAMPAIGN SIMULATION')
    st.markdown("""We can now simulate a promotion by selecting products, date period and a discount value (same for all selected products). Moreover,  such KPIs as incremental sales and incremental margin can be calculated using the next formulas:""")    

    st.markdown("""<body style='text-align: left;  '>incremental_sales = promo_price * quantity_with_promo - 
            regular_price * quantity_without_promo</body>""", unsafe_allow_html=True)

    st.markdown("""<body style='text-align: left;  type:italic'>incremental_margin = (promo_price - item_cost) * quantity_with_promo -  (regular_price - item_cost) * quantity_without_promo</body>""", unsafe_allow_html=True)

    mandatory_for_promo = ['promotion_flag', 'promotion_price', 'regular_price', 'item_cost', 'promotion_depth', 'promo_mechanics_id']
    missing_promo_features = [val for val in mandatory_for_promo if val not in feature_all]

    if (len(missing_promo_features)!=0):
            st.warning('Feature: {} should be selected at the very beginning to be able to simulate promo and calculate KPIs.'.format(missing_promo_features))
    else:
        promo = True
        st.subheader('Select products, discount and period for promo simulation')

        discount = st.number_input('Insert a discount value in %',
                                min_value=0,
                                max_value= 99,
                                value=10, 
                                step=1,
                                format='%d')

        # get input data for PROMO simulation
        promo_sim_products = st.multiselect(
            'Select products to be promoted',
             products.tolist(),
             default = [products[0], products[1], products[2]],
        format_func=get_product_name
        )


        promo_start_date = st.date_input(
             "Select promotion start date",
             datetime.date(2020, 3, 1))

        promo_end_date = st.date_input(
             "Select promotion end date",
             datetime.date(2020, 3, 14))
        
        # add warning for campaign duration 
        promo_duration = promo_end_date-promo_start_date
        if (promo_duration.days > 20):
            st.warning('You have selected a promo period with a duration of more than 20 days, to get more accurate prediction please use shorter intervals.')
        if (len(promo_sim_products) != 0 and promo_start_date <= promo_end_date):

            totals = plot_prediction(df, promo_sim_products, product_encoder, promo_start_date, promo_end_date, promo, discount)

            st.text('Calculated incremental sales and incremental margin for the selected products:')
            #st.write(totals.T.style.format({4:'{:.2f}'}))
             
            st.write(totals.T)
            st.markdown('Total **incremental sales = {}** and **incremental margin = {}** for the simulated promo campaign with start date = {}, end date = {}, discount = {}% and products selected above.'.format(
                '${:,.2f}'.format(totals.incremental_sales.sum()), 
                '${:,.2f}'.format(totals.incremental_margin.sum()), 
                promo_start_date, 
                promo_end_date, 
                discount
            )
                       )

            st.title('PROMO SIMULATION: DISCOUNT VALUE')
            st.markdown('So far we have discovered how to select products for the promo campaign, let\'s find out the best amount of discount that should be applied to maxime profitability. Products, start date and end date for the promo are taken from the example above.')
            discounts = np.linspace(5, 95, num=19, dtype=int).tolist()
            discounts_list = st.multiselect(
                'Select discount values to compare',
                discounts,
                default = [discounts[0], discounts[2], discounts[4]],
                format_func = get_percentage_name
            )
            
            if (len(discounts_list)!=0):
                all_totasls = get_total_incrementals(df, promo_sim_products, product_encoder, start_date, end_date, discounts_list, promo)
                plot_total_incrementals(all_totasls)
        else:
            st.warning('please select at least one product and "start date" smaller than "end date"')