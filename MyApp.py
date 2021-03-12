from catboost import CatBoostClassifier, Pool
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import shap
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

# Webapp title
st.set_page_config(page_title='SHAP Values Explorer')

# Load CatBoost model
@st.cache
def model_cache(modelname):
    cached_model = CatBoostClassifier().load_model(f'data/{modelname}')
    return cached_model

model = model_cache('catboost_model')


# Load pickle files
@st.cache
def st_cache(filename):
    cached_data = pickle.load(open(f'data/{filename}', 'rb'))
    return cached_data

expected_value = st_cache('expected_value')
X_test = st_cache('X_test')
eval_set_features = st_cache('eval_set_features')
train_set_features = st_cache('train_set_features')
shap_values = st_cache('shap_values')


# Set Streamlit app body title
st.title('Explore the SHAP values of a CatBoost song genre Multiclassifier')

st.markdown(
    " This WebApp allows you to visualize the SHAP values of an inputed instance (left sidebar) as well as the SHAP values of a preloaded evaluation dataset."
    " More details on SHAP (SHapley Additive exPlanations) can be found on [**GitHub**](https://Blaqhsff.com/MarcSkovMadsen/awesome-streamlit/issues)."
)   
st.info(
    " The model used is a **CatBoost Multiclassifier**."
    " More SHAP explanations and details on the model in [**this Notebook**](https://github.com/fauconnier-n/Streamlit-SHAP-Explorer/blob/main/notebook.ipynb)"
)

# Input sidebar subheader
st.sidebar.subheader('Input the feature values of a song here :')

# Set 3 columns layout
col1, col2 = st.sidebar.beta_columns(2)

# Set input fields
track_popularity = col1.number_input(label='track_popularity', min_value=0, max_value=100,format='%i',step=1, value=20)
danceability = col2.number_input(label='danceability', min_value=0.0, max_value=1.0,format= '%.6f',step=0.000001, value=0.5)
energy = col1.number_input(label='energy', min_value=0.0, max_value=1.0,format= '%.6f',step=0.000001, value=0.5)
key = col2.number_input(label='key', min_value=0, max_value=11,format= '%i', value=0)
loudness = col1.number_input(label='loudness', min_value=-50.0, max_value=2.0,format= '%.6f',step=0.000001, value=-6.0)
mode = col2.number_input(label='mode', min_value=0, max_value=1,format= '%i',step=1, value=0)
speechiness = col1.number_input(label='speechiness', min_value=0.0, max_value=1.0,format= '%.6f',step=0.000001, value=0.5)
acousticness = col2.number_input(label='acousticness', min_value=0.0, max_value=1.00,format= '%.6f',step=0.000001, value=0.5)
instrumentalness = col1.number_input(label='instrumentalness', min_value=0.0, max_value=1.0,format= '%.6f',step=0.000001, value=0.5)
liveness = col2.number_input(label='liveness', min_value=0.0, max_value=1.0,format= '%.6f',step=0.000001, value=0.5)
valence = col1.number_input(label='valence', min_value=0.0, max_value=1.0,format= '%.6f',step=0.000001, value=0.5)
tempo = col2.number_input(label='tempo', min_value=0.0, max_value=300.0,format= '%.6f',step=0.000001, value=120.0)
duration_ms = col1.number_input(label='duration_ms', min_value=0, max_value=999999,format= '%i',step=1, value=225000)

st.sidebar.info('*nb. More on those features on the [Spotify API documentation](https://developer.spotify.com/documentation/web-api/reference/#object-audiofeaturesobject)*')


# List of Streamlit inputs to predict on
input_list = [track_popularity, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms]

# Get predicted classes
input_preds_class = model.predict(input_list)

# Get predicted probabilities for each class
input_preds_proba = model.predict_proba(input_list)

# Write predictions on Streamlit app
st.sidebar.write('Class predicted :', input_preds_class[0])
st.sidebar.write(pd.DataFrame({'Genre' : model.classes_, 'Probability' : input_preds_proba}))

# DataFrame to predict on
df_input = pd.DataFrame([input_list], columns=['track_popularity', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'])

# Calculate shap values of inputed instance
explainer = shap.TreeExplainer(model)
input_shap_values = explainer.shap_values(df_input)



# SHAP force plot for inputed instance predicted class
st.subheader('Force plot')

force_plot = shap.force_plot(explainer.expected_value[np.argmax(input_preds_proba)],
                    input_shap_values[np.argmax(input_preds_proba)],
                    eval_set_features,
                    matplotlib=True,
                    show=False)

plt.suptitle(f'Class predicted : {model.classes_[np.argmax(input_preds_proba)]}',
             fontsize=20,
             y=1.35)

st.pyplot(force_plot)

# Force plot expander explanations
with st.beta_expander("More on force plots"):
     st.markdown("""
        The Force plot shows how each feature has contributed in moving away or towards the base value (average class output of the evaluation dataset) in to the predicted value of the specific instance (inputed on the left side bar) for the predicted class.

        Those values are **log odds**: SHAP doesn't support output probabilities for Multiclassification as of now.

        The SHAP values displayed are additive. Once the negative values (blue) are substracted from the positive values (pink), the distance from the base value to the output remains.

     """)



# SHAP decision plot for inputed instance
st.subheader('Decision plot')

def class_labels(row_index):
    return [f'{model.classes_[i]} (pred: {input_preds_proba[i].round(2)})' for i in range(len(expected_value))]

decision_plot, ax = plt.subplots()
ax = shap.multioutput_decision_plot(expected_value,
                               input_shap_values,
                               row_index=0, 
                               feature_names=eval_set_features, 
                               legend_labels=class_labels(0),
                               legend_location='lower right',
#                               link='logit',
                               highlight=np.argmax(input_preds_proba)) # Highlight the predicted class

st.pyplot(decision_plot)

# Decision plot expander explanations
with st.beta_expander("More on decision plots"):
     st.markdown("""
     Just like the force plot, the [**decision plot**](https://slundberg.github.io/shap/notebooks/plots) shows how each feature has contributed in moving away or towards the base value (the grey line, aka. the average model output on the evaluation dataset) to the predicted value of the specific instance (inputed on the left side bar), but allows us to visualize those effects **for each class**.
It also show the impact of less influencial features more clearly.

From SHAP documentation:
- *The x-axis represents the model's output. In this case, the units are log odds. (SHAP doesn't support probability output for multiclass)*
- *The plot is centered on the x-axis at explainer.expected_value (the base value). All SHAP values are relative to the model's expected value like a linear model's effects are relative to the intercept.*
- *The y-axis lists the model's features. By default, the features are ordered by descending importance. The importance is calculated over the observations plotted. _This is usually different than the importance ordering for the entire dataset._ In addition to feature importance ordering, the decision plot also supports hierarchical cluster feature ordering and user-defined feature ordering.*
- *Each observation's prediction is represented by a colored line. At the top of the plot, each line strikes the x-axis at its corresponding observation's predicted value. This value determines the color of the line on a spectrum.*
- *Moving from the bottom of the plot to the top, SHAP values for each feature are added to the model's base value. This shows how each feature contributes to the overall prediction.*
- *At the bottom of the plot, the observations converge at explainer.expected_value (the base value)*""")

# Set up 2 columns to display in the body of the app
st.subheader('Dependence plot: SHAP values of the evaluation dataset')
colbis1, colbis2, colbis3 = st.beta_columns(3)



# Selectors for dependence plot
class_selector = colbis1.selectbox('Genre :', model.classes_, index=0)
feature_selector = colbis2.selectbox('Main feature :', X_test.columns, index=0)
interaction_selector = colbis3.selectbox('Interaction feature :', X_test.columns, index=5)

# SHAP dependence plot
dependence_plot= shap.dependence_plot(feature_selector, 
                        shap_values[model.classes_.tolist().index(class_selector)], 
                        X_test, 
                        interaction_index=interaction_selector, 
                        x_jitter=0.95, 
                        alpha=0.4, 
                        dot_size=6, 
                        show=True)

plt.title(f'Genre : {model.classes_[model.classes_.tolist().index(class_selector)]}', fontsize=10)

st.pyplot(dependence_plot)

# Dependance plots expander explanations
with st.beta_expander("More on decision plots"):
     st.markdown("""
     From the SHAP documentation:

"*A [**dependence plot**](https://slundberg.github.io/shap/notebooks/plots/dependence_plot.html) is a scatter plot that shows the effect a single feature has on the predictions made by the model.* 

- *Each dot is a single prediction (row) from the dataset.*
- *The x-axis is the value of the feature (from the X matrix).*
- *The y-axis is the SHAP value for that feature, which represents how much knowing that feature's value changes the output of the model for that sample's prediction. For this model the units are log-odds of making over 50k annually.*
- *The color corresponds to a second feature that may have an interaction effect with the feature we are plotting (by default this second feature is chosen automatically). If an interaction effect is present between this other feature and the feature we are plotting it will show up as a distinct vertical pattern of coloring. For the example below 20-year-olds with a high level of education are less likely make over 50k than 20-year-olds with a low level of education. This suggests an interaction effect between Education-Num and Age.*"


Here the feature key goes from 0 (C) to 12 (B) by consecutive semi-tone. Mode is minor if = 0, major if = 1.

We can see that the probability of rap tends to be higher if the key is C# relatively to other keys.
D# (3) major has a geater positive impact than minor on the probability of rap. G (10) major has a greater negative impact on the probability of rap compared to minor.""")
