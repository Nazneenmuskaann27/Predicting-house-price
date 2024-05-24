import streamlit as st
import pandas as pd
import numpy as np
import pickle
def set_background_image(image_url):
    # Apply custom CSS to set the background image
    page_bg_img = '''
    <style>
    .stApp {
        background-position: top;
        background-image: url(%s);
        background-size: cover;
    }

    @media (max-width: 768px) {
        /* Adjust background size for mobile devices /
        .stApp {
            background-position: top;
            background-size: contain;
            background-repeat: no-repeat;
        }
    }
    </style>
    ''' % image_url
    st.markdown(page_bg_img, unsafe_allow_html=True)


def main():
    # Set the background image URL
    background_image_url = "https://static.vecteezy.com/system/resources/thumbnails/005/327/845/small/hand-holding-house-model-in-blue-background-for-refinance-plan-and-real-estate-concept-free-photo.jpg"

    # Set the background image
    set_background_image(background_image_url)

    custom_css = """
       <style>
       body {
           background-color: #4699d4;
           color: #ffffff;
           font-family: Arial, sans-serif;
       }
       select {
           background-color: #000000 !important; / Black background for select box /
           color: #ffffff !important; / White text within select box /
       }
       label {
           color: #ffffff !important; / White color for select box label */
       }
       </style>
       """
    st.markdown(custom_css, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
# Load the trained model and other necessary objects from pickle file
def load_model(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

model_data = load_model('model_and_data.pkl')
model = model_data['model']
data_columns = model_data['data_columns']

# Function to predict house price
def predict_price(bedrooms, sqft):
    input_data = pd.DataFrame([[bedrooms, sqft]], columns=['bedrooms', 'sqft'])
    prediction = model.predict(input_data)[0]
    return prediction

# Streamlit app
def main():
    st.title('House Price Prediction')
    st.markdown('Enter the number of bedrooms and square footage to predict the house price.')

    # Input form
    bedrooms = st.slider('Number of bedrooms', min_value=1, max_value=10, value=3)
    sqft = st.slider('Square footage', min_value=500, max_value=5000, value=1500)

    if st.button('Predict'):
        price = predict_price(bedrooms, sqft)
        st.success(f'Predicted Price: ${price:,.2f}')

if __name__ == '__main__':
    main()
