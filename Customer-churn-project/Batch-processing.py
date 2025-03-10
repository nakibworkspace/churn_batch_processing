from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, MinMaxScaler
from pyspark.ml import Pipeline

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("ChurnPreprocessing") \
    .config("spark.ui.port", "4040") \
    .getOrCreate()

# Load dataset
df = spark.read.csv("/root/code/Customer-churn-project/Data/WA_Fn-UseC_-Telco-Customer-Churn.csv", header=True, inferSchema=True)

# Drop Unnecessary Column
df = df.drop("customerID")

# Convert TotalCharges to Numeric & Handle Missing Values
df = df.withColumn("TotalCharges", col("TotalCharges").cast(DoubleType()))
median_value = df.approxQuantile("TotalCharges", [0.5], 0.01)[0]
df = df.fillna({"TotalCharges": median_value})

# Replace "No internet service" & "No phone service" with "No"
replace_cols = ["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
for col_name in replace_cols:
    df = df.withColumn(col_name, when(col(col_name) == "No internet service", "No").otherwise(col(col_name)))

df = df.withColumn("PhoneService", when(col("PhoneService") == "No phone service", "No").otherwise(col("PhoneService")))

# Convert gender to Numeric
df = df.withColumn("gender", when(col("gender") == "Female", 1).otherwise(0))

# One-Hot Encoding for Multi-Class Categorical Features
categorical_cols = ['InternetService', 'Contract', 'PaymentMethod']
indexers = [StringIndexer(inputCol=col, outputCol=col+"_index") for col in categorical_cols]
encoders = [OneHotEncoder(inputCol=col+"_index", outputCol=col+"_encoded") for col in categorical_cols]

pipeline = Pipeline(stages=indexers + encoders)
df = pipeline.fit(df).transform(df)
df = df.drop(*categorical_cols)  # Drop original categorical columns

# Convert Binary Columns to Numeric
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
               'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
               'TechSupport', 'StreamingTV', 'StreamingMovies', 
               'PaperlessBilling', 'Churn']

for col_name in binary_cols:
    df = df.withColumn(col_name, when(col(col_name) == "Yes", 1).otherwise(0))

# Feature Scaling for Numerical Features
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
assembler = VectorAssembler(inputCols=num_cols, outputCol="features_unscaled")
scaler = MinMaxScaler(inputCol="features_unscaled", outputCol="features")

pipeline = Pipeline(stages=[assembler, scaler])
df = pipeline.fit(df).transform(df)

df = df.drop(*num_cols)  # Drop original numerical columns

# Split Data into Train & Test
train_df, test_df = df.randomSplit([0.67, 0.33], seed=42)


train_df_csv= train_df.drop("InternetService_encoded", "Contract_encoded", 
                            "PaymentMethod_encoded", "features_unscaled", "features")

test_df_csv= test_df.drop("InternetService_encoded", "Contract_encoded", 
                           "PaymentMethod_encoded", "features_unscaled", "features")



# Save Preprocessed Data
train_df_csv.write.csv("/root/code/Customer-churn-project/Data/train_data.csv", header=True, mode="overwrite")
test_df_csv.write.csv("/root/code/Customer-churn-project/Data/test_data.csv", header=True, mode="overwrite")

print("✅ Preprocessing complete! Data saved successfully.")

# Keep the Spark session alive to access the Spark UI
input("Press Enter to exit...")  # Keeps Spark UI alive until you manually exit
