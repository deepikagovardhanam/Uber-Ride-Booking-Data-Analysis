
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\deepi\Downloads\ncr_ride_bookings.csv")

##A. Dataset Understanding

# 1. Display first 5 records
print(df.head())

# 2. Display last 5 records
print(df.tail())

# 3. Display 10 random records
print(df.sample(10))

# 4. Number of rows and columns
print(df.shape)

# 5. Column names
print(df.columns)

# 6. Data types
print(df.dtypes)

# 7. Dataset information
df.info()

# 8. Statistical summary
print(df.describe())

# 9. Dataset size
print(df.size)

# 10. Number of dimensions
print(df.ndim)

#B. Data Quality Analysis

# 1. Missing values column-wise
print(df.isnull().sum())

# 2. Total missing values
print(df.isnull().sum().sum())

# 3. Missing value percentage
print(df.isnull().mean() * 100)

# 4. Duplicate records
print(df.duplicated().sum())

# 5. Unique customers
print(df["Customer ID"].nunique())

# 6. Unique vehicle types
print(df["Vehicle Type"].unique())

# 7. Unique payment methods
print(df["Payment Method"].unique())

# 8. Unique booking statuses
print(df["Booking Status"].unique())

# 9. Count unique values in each column
print(df.nunique())

# 10. Remove duplicate records
df = df.drop_duplicates()

#Part C — Descriptive Analytics.

# 1. Total bookings
print(df.shape[0])

# 2. Completed bookings
print((df["Booking Status"] == "Completed").sum())

# 3. Bookings by status
print(df["Booking Status"].value_counts())

# 4. Total cancelled bookings
cancelled = df["Booking Status"].isin([
    "Cancelled by Driver",
    "Cancelled by Customer"
])
print(cancelled.sum())

# 5. Most frequently used vehicle
print(df["Vehicle Type"].value_counts().idxmax())

# 6. Average booking value
print(df["Booking Value"].mean())

# 7. Average ride distance
print(df["Ride Distance"].mean())

# 8. Average customer rating
print(df["Customer Rating"].mean())

# 9. Average driver rating
print(df["Driver Ratings"].mean())

# 10. Total booking value
print(df["Booking Value"].sum())

###D. groupby() Analysis — 10 questions ⭐

# 1. Number of bookings by vehicle type
print(df["Vehicle Type"].value_counts())

# 2. Total booking value by vehicle
print(df.groupby("Vehicle Type")["Booking Value"].sum())

# 3. Average booking value by vehicle
print(df.groupby("Vehicle Type")["Booking Value"].mean())

# 4. Average ride distance by vehicle
print(df.groupby("Vehicle Type")["Ride Distance"].mean())

# 5. Average customer rating by vehicle
print(df.groupby("Vehicle Type")["Customer Rating"].mean())

# 6. Bookings by payment method
print(df["Payment Method"].value_counts())

# 7. Total booking value by payment method
print(df.groupby("Payment Method")["Booking Value"].sum())

# 8. Bookings by status
print(df.groupby("Booking Status").size())

# 9. Bookings by pickup location
print(df["Pickup Location"].value_counts())

# 10. Total booking value by pickup location
print(df.groupby("Pickup Location")["Booking Value"].sum())

print(
    df.groupby("Vehicle Type").agg(
        Total_Bookings=("Booking Status", "size"),
        Total_Value=("Booking Value", "sum"),
        Average_Value=("Booking Value", "mean"),
        Average_Distance=("Ride Distance", "mean")
    )
)

##E. Filtering — 5 questions
# 1. Booking value greater than ₹1,000
print(df[df["Booking Value"] > 1000])

# 2. Ride distance greater than 20 km
print(df[df["Ride Distance"] > 20])

# 3. Customer rating greater than 4.5
print(df[df["Customer Rating"] > 4.5])

# 4. UPI payment bookings
print(df[df["Payment Method"] == "UPI"])

# 5. Completed bookings
print(df[df["Booking Status"] == "Completed"])

##F. Multiple Conditions — 2 questions

# 1. Value > ₹1,000 AND distance > 20 km
print(df[
    (df["Booking Value"] > 1000) &
    (df["Ride Distance"] > 20)
])

# 2. Completed bookings with value > ₹1,000
print(df[
    (df["Booking Status"] == "Completed") &
    (df["Booking Value"] > 1000)
])

# 3. UPI payments with completed status
print(df[
    (df["Payment Method"] == "UPI") &
    (df["Booking Status"] == "Completed")
])

# 4. OR condition
print(df[
    (df["Vehicle Type"] == "Auto") |
    (df["Vehicle Type"] == "Go Sedan")
])

# 5. NOT condition
print(df[df["Booking Status"] != "Completed"])

###G. Sorting — 2 questions
# 1. Top 10 highest-value rides
print(df.sort_values(
    "Booking Value", ascending=False
).head(10))

# 2. Five lowest-value rides
print(df.sort_values(
    "Booking Value", ascending=True
).head(5))

# 3. Longest rides
print(df.sort_values(
    "Ride Distance", ascending=False
).head(10))

# 4. Highest customer ratings
print(df.sort_values(
    "Customer Rating", ascending=False
).head(10))

# 5. Sort by vehicle and booking value
print(df.sort_values(
    ["Vehicle Type", "Booking Value"],
    ascending=[True, False]
))

#H. Final Business Insights

# 1. Vehicle with the most bookings
print(df["Vehicle Type"].value_counts().idxmax())

# 2. Vehicle with the highest total booking value
print(
    df.groupby("Vehicle Type")["Booking Value"]
    .sum().idxmax()
)

# 3. Vehicle with the highest average booking value
print(
    df.groupby("Vehicle Type")["Booking Value"]
    .mean().idxmax()
)

# 4. Vehicle with the highest average distance
print(
    df.groupby("Vehicle Type")["Ride Distance"]
    .mean().idxmax()
)

# 5. Most-used payment method
print(df["Payment Method"].value_counts().idxmax())

# 6. Payment method with highest total booking value
print(
    df.groupby("Payment Method")["Booking Value"]
    .sum().idxmax()
)

# 7. Most common booking status
print(df["Booking Status"].value_counts().idxmax())

# 8. Total cancelled bookings
print(df["Booking Status"].isin([
    "Cancelled by Driver",
    "Cancelled by Customer"
]).sum())

# 9. Pickup location with most bookings
print(df["Pickup Location"].value_counts().idxmax())

# 10. Pickup location with highest total booking value
print(
    df.groupby("Pickup Location")["Booking Value"]
    .sum().idxmax()
)

# I. NUMPY ANALYSIS

# 1. Convert columns into NumPy arrays
booking_values = df["Booking Value"].dropna().to_numpy()
distances = df["Ride Distance"].dropna().to_numpy()

# 2. Booking Value statistics
print("Average Booking Value:", np.mean(booking_values))
print("Median Booking Value:", np.median(booking_values))
print("Minimum Booking Value:", np.min(booking_values))
print("Maximum Booking Value:", np.max(booking_values))
print("Total Booking Value:", np.sum(booking_values))
print("Standard Deviation:", np.std(booking_values))

# 3. Ride Distance analysis
print("Average Ride Distance:", np.mean(distances))
print("Median Ride Distance:", np.median(distances))
print("Shortest Ride:", np.min(distances))
print("Longest Ride:", np.max(distances))
print("Total Ride Distance:", np.sum(distances))

# 4. Conditional analysis
print(
    "High Value Rides:",
    np.sum(booking_values > 1000)
)

print(
    "Long Distance Rides:",
    np.sum(distances > 20)
)

# 5. Categorize bookings using np.where()
df["Booking_Category"] = np.where(
    df["Booking Value"] > 1000,
    "High Value",
    "Regular Value"
)

print(df["Booking_Category"].value_counts())

# 6. Customer and Driver Rating
customer_ratings = df["Customer Rating"].dropna().to_numpy()
driver_ratings = df["Driver Ratings"].dropna().to_numpy()

print("Average Customer Rating:", np.mean(customer_ratings))
print("Average Driver Rating:", np.mean(driver_ratings))

#MATPLOTLIB VISUALIZATION 


# 1. Booking Status Analysis
plt.figure(figsize=(8, 5))

df["Booking Status"].value_counts().plot(
    kind="bar",
    color="skyblue"
)

plt.title("Number of Bookings by Status")
plt.xlabel("Booking Status")
plt.ylabel("Number of Bookings")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 2. Booking Value Distribution
plt.figure(figsize=(8, 5))

plt.hist(
    df["Booking Value"].dropna(),
    bins=20,
    color="orange",
    edgecolor="black"
)

plt.title("Booking Value Distribution")
plt.xlabel("Booking Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# 3. Average Booking Value by Vehicle Type
plt.figure(figsize=(9, 5))

df.groupby("Vehicle Type")["Booking Value"].mean().plot(
    kind="bar",
    color="mediumseagreen"
)

plt.title("Average Booking Value by Vehicle Type")
plt.xlabel("Vehicle Type")
plt.ylabel("Average Booking Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 4. Ride Distance Distribution
plt.figure(figsize=(8, 5))

plt.hist(
    df["Ride Distance"].dropna(),
    bins=20,
    color="mediumpurple",
    edgecolor="black"
)

plt.title("Ride Distance Distribution")
plt.xlabel("Ride Distance (km)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# 5. Payment Method Analysis
plt.figure(figsize=(8, 5))

df["Payment Method"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Bookings by Payment Method")
plt.ylabel("")
plt.tight_layout()
plt.show()


# 6. Customer Rating Distribution
plt.figure(figsize=(8, 5))

plt.hist(
    df["Customer Rating"].dropna(),
    bins=10,
    color="lightcoral",
    edgecolor="black"
)

plt.title("Customer Rating Distribution")
plt.xlabel("Customer Rating")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

#SEABORN VISUALIZATION

import seaborn as sns
import matplotlib.pyplot as plt

# Set chart style
sns.set_theme(style="whitegrid")


# 1. Booking Value Distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Booking Value",
    bins=20,
    kde=True,
    color="skyblue"
)

plt.title("Booking Value Distribution")
plt.xlabel("Booking Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# 2. Booking Count by Vehicle Type
plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="Vehicle Type",
    hue="Vehicle Type",
    legend=False,
    palette="viridis"
)

plt.title("Bookings by Vehicle Type")
plt.xlabel("Vehicle Type")
plt.ylabel("Number of Bookings")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 3. Average Booking Value by Payment Method
plt.figure(figsize=(9, 5))

sns.barplot(
    data=df,
    x="Payment Method",
    y="Booking Value",
    hue="Payment Method",
    legend=False,
    estimator="mean",
    errorbar=None,
    palette="magma"
)

plt.title("Average Booking Value by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Average Booking Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 4. Ride Distance vs Booking Value
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Ride Distance",
    y="Booking Value",
    hue="Booking Status",
    alpha=0.6
)

plt.title("Ride Distance vs Booking Value")
plt.xlabel("Ride Distance (km)")
plt.ylabel("Booking Value")
plt.tight_layout()
plt.show()


# 5. Customer Rating by Vehicle Type
plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="Vehicle Type",
    y="Customer Rating",
    color="lightgreen"
)

plt.title("Customer Rating by Vehicle Type")
plt.xlabel("Vehicle Type")
plt.ylabel("Customer Rating")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 6. Correlation Heatmap
plt.figure(figsize=(8, 5))

numeric_df = df[
    ["Booking Value", "Ride Distance",
     "Customer Rating", "Driver Ratings"]
].apply(
    lambda column: pd.to_numeric(column, errors="coerce")
)

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Between Numeric Variables")
plt.tight_layout()
plt.show()

# L. BUSINESS INSIGHTS

print("\n--- BUSINESS INSIGHTS ---")

# 1. Most common booking status
print("Booking Status Counts:")
print(df["Booking Status"].value_counts())

# 2. Most popular vehicle type
print("\nBookings by Vehicle Type:")
print(df["Vehicle Type"].value_counts())

# 3. Most used payment method
print("\nBookings by Payment Method:")
print(df["Payment Method"].value_counts())

# 4. Average booking value by vehicle type
print("\nAverage Booking Value by Vehicle:")
print(df.groupby("Vehicle Type")["Booking Value"].mean().sort_values(ascending=False))

# 5. Average ride distance by vehicle type
print("\nAverage Ride Distance by Vehicle:")
print(df.groupby("Vehicle Type")["Ride Distance"].mean().sort_values(ascending=False))

# 6. Customer ratings by vehicle type
print("\nAverage Customer Rating by Vehicle:")
print(df.groupby("Vehicle Type")["Customer Rating"].mean().sort_values(ascending=False))


