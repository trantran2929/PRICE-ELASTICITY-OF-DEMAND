import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import lightgbm as lgbm
import xgboost as xgb
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Hàm đánh giá mô hình
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

# Hàm tính độ co giãn giá cầu (PED) từ mô hình log-log
def calculate_ped(X, y, price_col):
    epsilon = 1e-5
    log_X = np.log(X[price_col] + epsilon)
    log_y = np.log(y + epsilon)
    model = LinearRegression()
    model.fit(log_X.values.reshape(-1, 1), log_y)
    ped = model.coef_[0]
    return ped

# Hàm xử lý ngoại lai bằng Z-score
def remove_outliers(df, columns, threshold=3):
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
            df_clean = df_clean[z_scores < threshold]
    return df_clean

# Hàm tải và xử lý dữ liệu Store Sales (Price = Total_Sale_Value / Qty_Sold)
@st.cache_data
def load_store_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'Store_Sales_Price_Elasticity_Promotions_Data.parquet')
    
    if not os.path.exists(file_path):
        st.error(f"File không tồn tại tại: {file_path}")
        st.info("Vui lòng đặt file 'Store_Sales_Price_Elasticity_Promotions_Data.parquet' trong cùng thư mục với file dashboard.py hoặc trong thư mục 'data'.")
        return None, None, None, None, None, None
    
    try:
        store_data = pd.read_parquet(file_path)
    except Exception as e:
        st.error(f"Không thể tải dữ liệu Store Sales từ {file_path}. Lỗi: {str(e)}")
        return None, None, None, None, None, None
    
    # Tính cột Price
    store_data['Price'] = store_data['Total_Sale_Value'] / store_data['Qty_Sold'].replace(0, np.nan)
    store_data['Price'] = store_data['Price'].fillna(store_data['Price'].median())
    
    store_data = store_data.drop_duplicates()
    num_cols = store_data.select_dtypes(include=['int16', 'int32', 'float32', 'int8', 'int64', 'float64']).columns
    imputer = SimpleImputer(strategy='median')
    store_data[num_cols] = imputer.fit_transform(store_data[num_cols])
    
    store_data = remove_outliers(store_data, ['Qty_Sold', 'Total_Sale_Value', 'Price'])
    
    store_data['Sold_Date'] = pd.to_datetime(store_data['Sold_Date'], errors='coerce')
    
    X = store_data[['Store_Number', 'SKU_Coded', 'Product_Class_Code', 'Price', 'On_Promo']]
    y = store_data['Total_Sale_Value']
    price_col = 'Price'
    
    cat_cols = X.select_dtypes(include=['object']).columns
    X = X.copy()
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return store_data, X_train, X_test, y_train, y_test, price_col

# Hàm tải và xử lý dữ liệu Electronics
@st.cache_data
def load_electronics_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'DatafinitiElectronicsProductsPricingData.csv')
    
    if not os.path.exists(file_path):
        st.error(f"File không tồn tại tại: {file_path}")
        st.info("Vui lòng đặt file 'DatafinitiElectronicsProductsPricingData.csv' trong cùng thư mục với file dashboard.py hoặc trong thư mục 'data'.")
        return None, None, None, None, None, None
    
    try:
        electronics_data = pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Không thể tải dữ liệu Electronics từ {file_path}. Lỗi: {str(e)}")
        return None, None, None, None, None, None
    
    electronics_data = electronics_data.drop_duplicates()
    price_col = 'prices.amountMin'
    selected_cols = ['prices.amountMin', 'prices.amountMax', 'prices.isSale', 'brand', 'categories', 'dateAdded', 'weight']
    if len(selected_cols) != len(set(selected_cols)):
        st.warning("Cột trùng lặp trong selected_cols, loại bỏ trùng lặp.")
        selected_cols = list(dict.fromkeys(selected_cols))
    electronics_data_filtered = electronics_data[selected_cols].copy()
    electronics_data_filtered = electronics_data_filtered[electronics_data_filtered[price_col].notna()]
    
    electronics_data_filtered = remove_outliers(electronics_data_filtered, [price_col, 'prices.amountMax'])
    
    electronics_data_filtered['prices.isSale'] = electronics_data_filtered['prices.isSale'].map({'true': 1, 'false': 0, True: 1, False: 0})
    electronics_data_filtered['brand_encoded'] = electronics_data_filtered['brand'].fillna('Unknown')
    le_brand = LabelEncoder()
    electronics_data_filtered['brand_encoded'] = le_brand.fit_transform(electronics_data_filtered['brand_encoded'])
    
    electronics_data_filtered['category_count'] = electronics_data_filtered['categories'].fillna('').apply(lambda x: len(str(x).split(',')))
    electronics_data_filtered['dateAdded'] = pd.to_datetime(electronics_data_filtered['dateAdded'], errors='coerce')
    electronics_data_filtered['year_added'] = electronics_data_filtered['dateAdded'].dt.year
    electronics_data_filtered['month_added'] = electronics_data_filtered['dateAdded'].dt.month
    
    try:
        electronics_data_filtered['weight_value'] = electronics_data_filtered['weight'].str.extract(r'(\d+\.?\d*)').astype(float)
    except:
        electronics_data_filtered['weight_value'] = np.nan
    
    features = ['prices.amountMax', 'prices.isSale', 'brand_encoded', 'category_count', 'year_added', 'month_added', 'weight_value']
    electronics_data_filtered = electronics_data_filtered.dropna(subset=[price_col])
    
    for feature in features:
        if feature in electronics_data_filtered.columns and electronics_data_filtered[feature].dtype in ['int64', 'float64']:
            imputer = SimpleImputer(strategy='median')
            electronics_data_filtered[feature] = imputer.fit_transform(electronics_data_filtered[feature].values.reshape(-1, 1))
    
    if len(electronics_data_filtered.columns) != len(set(electronics_data_filtered.columns)):
        st.error(f"Phát hiện cột trùng lặp trong electronics_data_filtered: {electronics_data_filtered.columns.tolist()}")
        return None, None, None, None, None, None
    
    X_electronics = electronics_data_filtered[features].copy()
    y_electronics = electronics_data_filtered[price_col]
    
    scaler = MinMaxScaler()
    X_electronics_scaled = scaler.fit_transform(X_electronics)
    
    X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X_electronics_scaled, y_electronics, test_size=0.2, random_state=42)
    return electronics_data_filtered, X_train_e, X_test_e, y_train_e, y_test_e, price_col

# Hàm tối ưu hóa giá
def optimize_price(model, X_sample, price_col, min_price, max_price, feature_names, dataset, current_price, current_quantity, product_id=None):
    def revenue_function(price):
        try:
            if dataset == "Store Sales" and price_col in feature_names:
                X_temp = X_sample.copy()
                price_idx = feature_names.index(price_col)
                X_temp[0, price_idx] = price
                quantity = model.predict(X_temp)[0]
            else:
                quantity = current_quantity * (current_price / price)
            return -price * quantity
        except Exception as e:
            raise ValueError(f"Lỗi trong revenue_function: {str(e)}")
    
    try:
        result = minimize(revenue_function, x0=current_price, bounds=[(min_price, max_price)])
        optimal_price = result.x[0]
        optimal_revenue = -result.fun
        return optimal_price, optimal_revenue
    except Exception as e:
        st.error(f"Lỗi khi tối ưu hóa giá: {str(e)}")
        return None, None

# Hàm huấn luyện và dự đoán
def train_and_predict(X_train, X_test, y_train, y_test, model_type):
    try:
        if np.any(np.isnan(X_train)) or np.any(np.isnan(X_test)) or np.any(np.isnan(y_train)) or np.any(np.isnan(y_test)):
            raise ValueError("Dữ liệu chứa giá trị NaN")
        if np.any(np.isinf(X_train)) or np.any(np.isinf(X_test)) or np.any(np.isinf(y_train)) or np.any(np.isinf(y_test)):
            raise ValueError("Dữ liệu chứa giá trị vô cực")

        if model_type == 'Linear Regression':
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = evaluate_model(y_test, y_pred)
        
        elif model_type == 'Log-Log Model':
            epsilon = 1e-5
            if np.any(y_train <= 0) or np.any(y_test <= 0):
                raise ValueError("Log-Log Model yêu cầu giá trị y > 0")
            log_y_train = np.log(y_train + epsilon)
            log_y_test = np.log(y_test + epsilon)
            model = LinearRegression()
            model.fit(X_train, log_y_train)
            log_y_pred = model.predict(X_test)
            y_pred = np.exp(log_y_pred) - epsilon
            metrics = evaluate_model(y_test, y_pred)
        
        elif model_type == 'LightGBM':
            model = lgbm.LGBMRegressor(
                random_state=42,
                force_row_wise=True,
                num_leaves=31,
                n_estimators=20,
                learning_rate=0.1
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = evaluate_model(y_test, y_pred)
        
        elif model_type == 'XGBoost':
            model = xgb.XGBRegressor(
                random_state=42,
                tree_method='hist',
                n_estimators=20
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = evaluate_model(y_test, y_pred)
        
        elif model_type == 'Random Forest':
            model = RandomForestRegressor(
                random_state=42,
                n_estimators=20,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = evaluate_model(y_test, y_pred)
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        for train_idx, val_idx in kf.split(X_train):
            X_kf_train, X_kf_val = X_train[train_idx], X_train[val_idx]
            y_kf_train, y_kf_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model.fit(X_kf_train, y_kf_train)
            y_kf_pred = model.predict(X_kf_val)
            mse = mean_squared_error(y_kf_val, y_kf_pred)
            rmse = np.sqrt(mse)
            cv_scores.append(rmse)
        cv_rmse = np.mean(cv_scores)
        
        return model, y_pred, metrics, cv_rmse
    
    except Exception as e:
        st.error(f"Lỗi khi huấn luyện mô hình {model_type}: {str(e)}")
        return None, None, None, None

# Streamlit App
st.title("Dashboard Đo lường Độ Co giãn Giá Cầu & Tối ưu Giá")

# Sidebar
st.sidebar.header("Cài đặt")
dataset = st.sidebar.selectbox("Chọn bộ dữ liệu", ["Store Sales", "Electronics"])
model_type = st.sidebar.selectbox("Chọn mô hình", ["Linear Regression", "Log-Log Model", "LightGBM", "XGBoost", "Random Forest"])
show_visualizations = st.sidebar.checkbox("Hiển thị biểu đồ trực quan", value=True)
show_comparison = st.sidebar.checkbox("So sánh tất cả mô hình", value=False)
show_price_optimization = st.sidebar.checkbox("Tối ưu hóa giá", value=False)
show_what_if = st.sidebar.checkbox("Mô phỏng kịch bản giá", value=False)

# Tải dữ liệu
if dataset == "Store Sales":
    data, X_train, X_test, y_train, y_test, price_col = load_store_data()
    target_name = "Total_Sale_Value"
    group_col = "Product_Class_Code"
else:
    data, X_train, X_test, y_train, y_test, price_col = load_electronics_data()
    target_name = "prices.amountMin"
    group_col = "categories"

if data is not None:
    st.header(f"Thông tin bộ dữ liệu: {dataset}")
    st.write(f"Kích thước dữ liệu: {data.shape}")
    st.write("Số lượng giá trị thiếu:")
    st.write(data.isnull().sum())
    
    if show_visualizations:
        st.subheader("Phân tích mối quan hệ giá và nhu cầu")
        
        if dataset == "Store Sales":
            corr = data[[price_col, target_name]].corr().iloc[0, 1]
        else:
            corr = data[[price_col, target_name]].corr().iloc[0, 1]
        st.write(f"Hệ số tương quan Pearson giữa {price_col} và {target_name}: {corr:.4f}")
        
        st.write(f"Biểu đồ phân tán: {price_col} vs {target_name}")
        fig = px.scatter(data, x=price_col, y=target_name, title=f"{price_col} vs {target_name}")
        st.plotly_chart(fig)
        
        st.write(f"Phân tích theo nhóm sản phẩm ({group_col})")
        if data is not None and group_col in data.columns:
            if dataset == "Store Sales":
                group_means = data.groupby(group_col)[target_name].mean().reset_index()
                fig = px.bar(group_means, x=group_col, y=target_name, title=f"{target_name} theo {group_col}")
                st.plotly_chart(fig)
            else:
                group_means = data.groupby(group_col)[target_name].mean().reset_index()
                group_means = group_means.rename(columns={target_name: 'Target'})
                if len(group_means.columns) != len(set(group_means.columns)):
                    st.warning("Phát hiện cột trùng lặp trong group_means, đặt tên cột thủ công.")
                    group_means.columns = ['categories', 'Target']
                fig = px.bar(group_means, x=group_col, y='Target', title=f"{target_name} theo {group_col}")
                st.plotly_chart(fig)
        else:
            st.error(f"Không thể tạo biểu đồ vì dữ liệu không tải được hoặc cột {group_col} không tồn tại.")
        
        st.write(f"Phân phối của {target_name}")
        fig = px.histogram(data, x=target_name, nbins=50, title=f"Phân phối của {target_name}")
        st.plotly_chart(fig)
        
        st.write("Ma trận tương quan")
        corr_data = data.select_dtypes(include=['int16', 'int32', 'float32', 'int8', 'int64', 'float64'])
        corr = corr_data.corr()
        fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='RdBu'))
        fig.update_layout(title="Ma trận tương quan")
        st.plotly_chart(fig)

    st.subheader("Độ co giãn giá cầu (PED)")
    ped = calculate_ped(data, data[target_name], price_col)
    st.write(f"Độ co giãn giá cầu (PED): {ped:.4f}")
    if abs(ped) > 1:
        st.write("Cầu co giãn cao (elastic) - Giảm giá có thể tăng doanh thu.")
    else:
        st.write("Cầu ít co giãn (inelastic) - Tăng giá có thể tăng doanh thu.")

    model, y_pred, metrics, cv_rmse = train_and_predict(X_train, X_test, y_train, y_test, model_type)
    
    st.header(f"Kết quả: {model_type} trên {dataset}")
    st.write(f"R² Score: {metrics['r2']:.4f}")
    st.write(f"RMSE: {metrics['rmse']:.4f}")
    st.write(f"MAE: {metrics['mae']:.4f}")
    st.write(f"Cross-Validation RMSE: {cv_rmse:.4f}")
    
    st.subheader("So sánh giá trị thực tế và dự đoán")
    fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                     title=f"{model_type}: Actual vs Predicted")
    fig.add_scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                    mode='lines', name='Ideal', line=dict(color='red', dash='dash'))
    st.plotly_chart(fig)
    
    # Tối ưu hóa giá
    if show_price_optimization:
        st.subheader("Tối ưu hóa giá")
        if data is not None and price_col in data.columns:
            st.write("Nhập thông tin sản phẩm để tối ưu hóa giá:")
            current_price = st.number_input("Giá hiện tại", min_value=0.0, value=float(data[price_col].mean()), step=0.01)
            current_quantity = st.number_input("Số lượt bán hiện tại (nhu cầu)", min_value=0.0, value=1.0, step=1.0)
            
            if dataset == "Store Sales":
                product_options = data['Product_Class_Code'].unique()
                product_id = st.selectbox("Chọn mã sản phẩm (Product_Class_Code)", options=product_options)
            else:
                product_options = data['categories'].unique()
                product_id = st.selectbox("Chọn danh mục sản phẩm (categories)", options=product_options)
            
            min_price = data[price_col].min()
            max_price = data[price_col].max()
            
            if dataset == "Store Sales":
                feature_names = ['Store_Number', 'SKU_Coded', 'Product_Class_Code', 'Price', 'On_Promo']
            else:
                feature_names = ['prices.amountMax', 'prices.isSale', 'brand_encoded', 'category_count', 'year_added', 'month_added', 'weight_value']
            
            if dataset == "Store Sales":
                sample_data = data[data['Product_Class_Code'] == product_id][feature_names].head(1)
            else:
                sample_data = data[data['categories'] == product_id][feature_names].head(1)
            
            if not sample_data.empty:
                scaler = MinMaxScaler()
                X_sample = scaler.fit_transform(sample_data)
                optimal_price, optimal_revenue = optimize_price(
                    model, X_sample, price_col, min_price, max_price, feature_names, dataset, current_price, current_quantity, product_id
                )
                if optimal_price is not None:
                    st.write(f"Giá tối ưu: {optimal_price:.2f}")
                    st.write(f"Doanh thu dự kiến: {optimal_revenue:.2f}")
                else:
                    st.warning("Không thể tính giá tối ưu do lỗi dữ liệu hoặc mô hình.")
            else:
                st.error(f"Không tìm thấy dữ liệu cho sản phẩm {product_id}.")
        else:
            st.error("Không thể tối ưu hóa giá vì dữ liệu không tải được hoặc cột giá không tồn tại.")

    # So sánh tất cả mô hình
    if show_comparison:
        st.subheader("So sánh hiệu suất tất cả mô hình")
        models = ['Linear Regression', 'Log-Log Model', 'LightGBM', 'XGBoost', 'Random Forest']
        results = {}
        
        for model_name in models:
            try:
                model, _, metrics, cv_rmse = train_and_predict(X_train, X_test, y_train, y_test, model_name)
                if metrics is not None:
                    results[model_name] = {'R²': metrics['r2'], 'RMSE': metrics['rmse'], 'MAE': metrics['mae'], 'CV RMSE': cv_rmse}
                else:
                    st.warning(f"Mô hình {model_name} không thể huấn luyện.")
            except Exception as e:
                st.warning(f"Lỗi khi huấn luyện mô hình {model_name}: {str(e)}")
        
        if results:
            comparison_data = []
            for model_name in models:
                if model_name in results:
                    row = {'Model': model_name, **results[model_name]}
                    comparison_data.append(row)
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=list(results.keys()), y=[results[model]['R²'] for model in results], name='R²'))
            fig.update_layout(title="So sánh R² giữa các mô hình", xaxis_title="Mô hình", yaxis_title="R² Score")
            st.plotly_chart(fig)
        else:
            st.error("Không có mô hình nào huấn luyện thành công.")

    # Mô phỏng kịch bản giá
    if show_what_if:
        st.subheader("Mô phỏng kịch bản giá")
        price_range = np.linspace(data[price_col].min(), data[price_col].max(), 10)
        what_if_data = []
        for price in price_range:
            X_temp = X_test[0:1].copy()
            if dataset == "Store Sales" and price_col in feature_names:
                price_idx = feature_names.index(price_col)
                X_temp[0, price_idx] = price
                quantity = model.predict(X_temp)[0]
            else:
                quantity = current_quantity * (current_price / price) if 'current_quantity' in locals() else 1.0
            revenue = price * quantity
            what_if_data.append({'Price': price, 'Quantity': quantity, 'Revenue': revenue})
        
        what_if_df = pd.DataFrame(what_if_data)
        st.write(what_if_df)
        
        fig = px.line(what_if_df, x='Price', y='Revenue', title="Doanh thu theo các mức giá")
        st.plotly_chart(fig)

else:
    st.error("Vui lòng kiểm tra đường dẫn dữ liệu hoặc định dạng file.")