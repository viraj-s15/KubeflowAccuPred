import gradio as gr
import pickle
import pandas as pd
import xgboost as xg
from sklearn.preprocessing import LabelEncoder

with open("model/best_xgb_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)


def predict_frequency(
    age,
    gender,
    item_purchased,
    category,
    purchase_amount,
    location,
    size,
    color,
    season,
    review_rating,
    subscription_status,
    shipping_type,
    discount_applied,
    promo_code_used,
    previous_purchases,
    payment_method,
):
    data = pd.DataFrame(
        {
            "Age": [age],
            "Gender": [gender],
            "Item Purchased": [item_purchased],
            "Category": [category],
            "Purchase Amount (USD)": [purchase_amount],
            "Location": [location],
            "Size": [size],
            "Color": [color],
            "Season": [season],
            "Review Rating": [review_rating],
            "Subscription Status": [subscription_status],
            "Shipping Type": [shipping_type],
            "Discount Applied": [discount_applied],
            "Promo Code Used": [promo_code_used],
            "Previous Purchases": [previous_purchases],
            "Payment Method": [payment_method],
        }
    )
    cols = data.columns
    object_cols = [col for col in cols if data[col].dtype == "object"]

    if object_cols:
        le = LabelEncoder()
        for col in object_cols:
            data[col] = le.fit_transform(data[col])
    data_dmatrix = xg.DMatrix(data=data)
    print(data_dmatrix)
    prediction = loaded_model.predict(data_dmatrix)[0]
    return f"Predicted Frequency: {prediction}"


df = pd.read_csv("./data/data.csv")

inputs = [
    gr.Number(label="Age"),
    gr.Radio(["Male", "Female"], label="Gender"),
    gr.Dropdown(
        [
            "Blouse",
            "Sweater",
            "Jeans",
            "Sandals",
            "Sneakers",
            "Shirt",
            "Shorts",
            "Coat",
            "Handbag",
            "Shoes",
            "Dress",
            "Skirt",
            "Sunglasses",
            "Pants",
            "Jacket",
            "Hoodie",
            "Jewelry",
            "T-shirt",
            "Scarf",
            "Hat",
            "Socks",
            "Backpack",
            "Belt",
            "Boots",
            "Gloves",
        ],
        label="Item Purchased",
    ),
    gr.Dropdown(["Clothing", "Footwear", "Outerwear", "Accessories"], label="Category"),
    gr.Number(label="Purchase Amount (USD)"),
    gr.Dropdown(
        [
            "Kentucky",
            "Maine",
            "Massachusetts",
            "Rhode Island",
            "Oregon",
            "Wyoming",
            "Montana",
            "Louisiana",
            "West Virginia",
            "Missouri",
            "Arkansas",
            "Hawaii",
            "Delaware",
            "New Hampshire",
            "New York",
            "Alabama",
            "Mississippi",
            "North Carolina",
            "California",
            "Oklahoma",
            "Florida",
            "Texas",
            "Nevada",
            "Kansas",
            "Colorado",
            "North Dakota",
            "Illinois",
            "Indiana",
            "Arizona",
            "Alaska",
            "Tennessee",
            "Ohio",
            "New Jersey",
            "Maryland",
            "Vermont",
            "New Mexico",
            "South Carolina",
            "Idaho",
            "Pennsylvania",
            "Connecticut",
            "Utah",
            "Virginia",
            "Georgia",
            "Nebraska",
            "Iowa",
            "South Dakota",
            "Minnesota",
            "Washington",
            "Wisconsin",
            "Michigan",
        ]
    ),
    gr.Dropdown(["L", "S", "M", "XL"], label="Size"),
    gr.Dropdown(
        [
            "Gray",
            "Maroon",
            "Turquoise",
            "White",
            "Charcoal",
            "Silver",
            "Pink",
            "Purple",
            "Olive",
            "Gold",
            "Violet",
            "Teal",
            "Lavender",
            "Black",
            "Green",
            "Peach",
            "Red",
            "Cyan",
            "Brown",
            "Beige",
            "Orange",
            "Indigo",
            "Yellow",
            "Magenta",
            "Blue",
        ],
        label="Colour",
    ),
    gr.Dropdown(['Winter', 'Spring', 'Summer', 'Fall'],label="Season"),
    gr.Number(label="Review Rating"),
    gr.Radio(["Yes", "No"], label="Subscription"),
    gr.Dropdown(['Express', 'Free Shipping', 'Next Day Air', 'Standard', '2-Day Shipping', 'Store Pickup'],label="Shiping Type"),
    gr.Radio(["Yes", "No"], label="Discount Applied"),
    gr.Number(label="Previous Purchases"),
    gr.Dropdown(['Venmo', 'Cash', 'Credit Card', 'PayPal', 'Bank Transfer', 'Debit Card'],label="Payment Method"), 
]

iface = gr.Interface(
    fn=predict_frequency, inputs=inputs, outputs=gr.Textbox(label="Customer Frequency")
)

iface.launch(share=True)
