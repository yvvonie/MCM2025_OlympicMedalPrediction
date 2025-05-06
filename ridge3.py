import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
# 数据加载路径
data_dir2 = r"E:\MCM2025\2025_Problem_C_Data"

# 加载原始数据
programs = pd.read_csv(f"{data_dir2}\\cleaned_summerOly_programs.csv")
medal_train = pd.read_csv(f"{data_dir2}\\summerOly_medal_train_r.csv")
medal_test = pd.read_csv(f"{data_dir2}\\summerOly_medal_test.csv")
athletes = pd.read_csv(f"{data_dir2}\\summerOly_athletes.csv")
hosts = pd.read_csv(f"{data_dir2}\\cleaned_summerOly_hosts.csv")
hosts = hosts[['Year', 'Country']].rename(columns={'Country':'Host_Country'})

# ------------------
# 数据预处理与特征工程
# ------------------
def reshape_programs(df):
    """将年份列转换为长格式（项目存在性，不涉及国家）"""
    year_cols = [col for col in df.columns if col.isdigit() and len(col) == 4]
    
    df_long = df.melt(
        id_vars=['Sport', 'Discipline', 'Code', 'Sports Governing Body'],
        value_vars=year_cols,
        var_name='Year',
        value_name='Inclusion'
    )
    df_long = df_long[df_long['Inclusion'] > 0]
    df_long['Year'] = df_long['Year'].astype(int)
    return df_long[['Year', 'Sport', 'Discipline', 'Code', 'Inclusion']]  # 包含Inclusion列

medal_by_sport = pd.concat([medal_train, medal_test]).merge(
    athletes[['Year', 'Country', 'Sport']].drop_duplicates(),
    on=['Year', 'Country'],
    how='left'
)
top_sports = medal_by_sport.groupby('Sport')['Total'].sum().nlargest(10).index.tolist()
print("Top 10奖牌关联项目:", top_sports)
#用plt画一张top10奖牌项目的表格 ，每个项目分别对（2012，2016，2020，2024的奖牌数取均值
plt.figure(figsize=(12, 6))  # Increase the figure width

for sport in top_sports:
    # Filter data for the specific sport and years
    sport_data = medal_by_sport[(medal_by_sport['Sport'] == sport) & 
                                (medal_by_sport['Year'].isin([2012, 2016, 2020, 2024]))]
    # Calculate the mean medals for each year
    mean_medals = sport_data.groupby('Year')['Total'].mean()
    # Plot the data
    plt.plot(mean_medals.index, mean_medals.values, marker='o', label=sport)

# Set x-ticks to specific years
plt.xticks([2012, 2016, 2020, 2024])

# Move legend outside the plot
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11)

plt.title('Top 10 Medal-Associated Sports')
plt.xlabel('Year')
plt.ylabel('Total Medals')

# Adjust layout to ensure everything fits
plt.tight_layout(rect=[0, 0, 0.92, 1])  # Adjust the right margin

plt.show()


# 应用转换
programs_long = reshape_programs(programs)
print(programs_long.head())

def create_features(medal_df, programs_long_df, athletes_df):
    """创建综合特征矩阵"""
    # 特征1：每个国家每年参赛人数
    #athletes_df  Team是国家名称列，NOC是缩写列
    athletes_count = athletes_df.groupby(['Year', 'Country'])['Name'].count().reset_index(name='athletes_count')
    
    # 特征2：每个国家每年参与的项目数量
    country_events = athletes_df.groupby(['Year', 'Country', 'Sport'])['Name'].count().reset_index()
    num_events = country_events.groupby(['Year', 'Country'])['Sport'].nunique().reset_index(name='num_events')
    
    # 合并基础数据
    merged = medal_df.merge(
        num_events,
        on=['Year', 'Country'], 
        how='left'
    ).merge(
        athletes_count,
        on=['Year', 'Country'],
        how='left'
    ).fillna(0)  # 处理缺失值
    
    # 特征3：历史奖牌特征
    merged = merged.sort_values(['Country', 'Year'])
    merged['past_3_medals_avg'] = merged.groupby('Country')['Total'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )


    # === 新增特征 ===
    # 特征6：国家参与高奖牌项目的数量
    # 筛选高奖牌项目（示例：游泳、田径、体操等）
    high_medal_sports = ['Swimming', 'Athletics', 'Gymnastics', 'Cycling', 'Wrestling']
    
    # 获取国家每年参与的高奖牌项目
    high_sport_participation = country_events[
        country_events['Sport'].isin(high_medal_sports)
    ].groupby(['Year', 'Country'])['Sport'].nunique().reset_index(name='num_high_sports')
    
    merged = merged.merge(
        high_sport_participation,
        on=['Year', 'Country'],
        how='left'
    ).fillna(0)
    
    # **动态识别东道主**
    merged = merged.merge(
        hosts[['Year', 'Host_Country']], 
        on='Year', 
        how='left'
    )
    merged['is_host'] = (merged['Country'] == merged['Host_Country']).astype(int)
    
    # 特征8：项目变化冲击（比较当前年份与上届的项目数差异）
    merged['event_change'] = merged.groupby('Country')['num_events'].diff().fillna(0)
    
    # Create 'has_sport' features
    for sport in top_sports:
        sport_participation = country_events[country_events['Sport'] == sport]
        sport_flag = sport_participation.groupby(['Year', 'Country']).size().reset_index(name=f'has_{sport}')
        merged = merged.merge(
            sport_flag[['Year', 'Country', f'has_{sport}']],
            on=['Year', 'Country'],
            how='left'
        ).fillna(0)
    
    # Ensure these columns are binary
    for sport in top_sports:
        merged[f'has_{sport}'] = (merged[f'has_{sport}'] > 0).astype(int)
    selected_columns = [
        'Year', 'Country', 'athletes_count', 'num_events', 
        'num_high_sports', 'past_3_medals_avg',
        'is_host', 'event_change', 'Gold', 'Total'
    ] + [f'has_{sport}' for sport in top_sports]
    return merged[selected_columns].dropna()
    

# 生成特征矩阵
programs_long = reshape_programs(programs)
print("programs_long中'Sport'列的缺失值数量:", programs_long['Sport'].isna().sum())
train_data = create_features(medal_train, programs_long, athletes)
test_data = create_features(medal_test, programs_long, athletes)
print("\n训练集特征示例：")
print(train_data.head())
# 模型配置
targets = ["Gold", "Total"]
features = [
    'athletes_count', 'num_events', 'num_high_sports',
    'past_3_medals_avg', 'is_host', 'event_change'
] + [f'has_{sport}' for sport in top_sports]

# ------------------
# 模型构建与验证
# ------------------

# 定义特征和目标变量

X_train = train_data[features]
y_train = train_data[targets]
X_test = test_data[features]
y_test = test_data[targets]

# 使用MultiOutputRegressor包装岭回归
from sklearn.multioutput import MultiOutputRegressor

pipeline = make_pipeline(
    StandardScaler(),
    MultiOutputRegressor(Ridge())
)

# 调整参数网格
param_grid = {
    'multioutputregressor__estimator__alpha': np.logspace(-3, 3, 20)
}

# 增加交叉验证次数
tscv = TimeSeriesSplit(n_splits=5)
# 网格搜索
grid = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='neg_mean_absolute_error')
grid.fit(X_train, y_train)

# 最佳模型
best_model = grid.best_estimator_
print(f"Best alpha: {grid.best_params_['multioutputregressor__estimator__alpha']}")

# ------------------
# 模型评估
# ------------------
def custom_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    mae_gold = mean_absolute_error(y.iloc[:,0], y_pred[:,0])
    mae_total = mean_absolute_error(y.iloc[:,1], y_pred[:,1])
    return - (mae_gold + mae_total)  # 综合MAE

grid = GridSearchCV(pipeline, param_grid, cv=tscv, scoring=custom_scorer)
grid.fit(X_train, y_train)

# 测试集预测
y_pred = grid.best_estimator_.predict(X_test)

# 分项指标计算
print("Gold 指标:")
print(f"MAE: {mean_absolute_error(y_test['Gold'], y_pred[:,0]):.1f}")
print(f"R2: {r2_score(y_test['Gold'], y_pred[:,0]):.2f}")

print("\nTotal 指标:")
print(f"MAE: {mean_absolute_error(y_test['Total'], y_pred[:,1]):.1f}")
print(f"R2: {r2_score(y_test['Total'], y_pred[:,1]):.2f}")
# 特征重要性分析
multioutput_model = best_model.named_steps['multioutputregressor']
ridge_for_gold = multioutput_model.estimators_[0]  # 金牌模型
ridge_for_total = multioutput_model.estimators_[1]  # 总奖牌模型

# 分别提取特征系数
coefs_gold = pd.DataFrame({
    'feature': features,
    'coefficient': ridge_for_gold.coef_,
    'abs_coef': np.abs(ridge_for_gold.coef_)
}).sort_values('abs_coef', ascending=False)

coefs_total = pd.DataFrame({
    'feature': features,
    'coefficient': ridge_for_total.coef_,
    'abs_coef': np.abs(ridge_for_total.coef_)
}).sort_values('abs_coef', ascending=False)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


original_features = [
    'athletes_count', 'num_events', 'num_high_sports',
    'past_3_medals_avg', 'is_host', 'event_change'
]

# 过滤掉不需要的特征
selected_features = [f for f in original_features if f != 'past_3_medals_avg']

# 提取对应特征的系数
coefs_combined = pd.DataFrame({
    'feature': selected_features,
    'coef_gold': ridge_for_gold.coef_[[original_features.index(f) for f in selected_features]],  # 添加方括号 []
    'coef_total': ridge_for_total.coef_[[original_features.index(f) for f in selected_features]]  # 添加方括号 []
})


# ------------------
# 计算评估指标
mae_gold = mean_absolute_error(y_test['Gold'], y_pred[:, 0])
r2_gold = r2_score(y_test['Gold'], y_pred[:, 0])
mae_total = mean_absolute_error(y_test['Total'], y_pred[:, 1])
r2_total = r2_score(y_test['Total'], y_pred[:, 1])

# 设置全局字体和样式
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# 创建子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 金牌预测子图
ax1.scatter(y_test['Gold'], y_pred[:, 0], alpha=0.6, color='#1f77b4', edgecolors='w', linewidth=0.5)
ax1.plot([y_test['Gold'].min(), y_test['Gold'].max()], 
         [y_test['Gold'].min(), y_test['Gold'].max()], '--r', linewidth=1.5)
ax1.set_xlabel('Actual Gold Medals', fontsize=12, labelpad=10)
ax1.set_ylabel('Predicted Gold Medals', fontsize=12, labelpad=10)
ax1.set_title(f'Gold Medal Predictions\nMAE = {mae_gold:.1f}, R² = {r2_gold:.2f}', 
              fontsize=14, pad=15)
ax1.grid(True, linestyle='--', alpha=0.6)

# 总奖牌预测子图
ax2.scatter(y_test['Total'], y_pred[:, 1], alpha=0.6, color='#2ca02c', edgecolors='w', linewidth=0.5)
ax2.plot([y_test['Total'].min(), y_test['Total'].max()], 
         [y_test['Total'].min(), y_test['Total'].max()], '--r', linewidth=1.5)
ax2.set_xlabel('Actual Total Medals', fontsize=12, labelpad=10)
ax2.set_ylabel('Predicted Total Medals', fontsize=12, labelpad=10)
ax2.set_title(f'Total Medal Predictions\nMAE = {mae_total:.1f}, R² = {r2_total:.2f}', 
              fontsize=14, pad=15)
ax2.grid(True, linestyle='--', alpha=0.6)

# 调整布局并显示
plt.tight_layout()
plt.show()

# 生成预测数据（接原始代码）
def prepare_2028_prediction(medal_data, athletes_data, programs_data, top_sports):
    """生成2028年预测特征（基于历史数据推断）"""
    # 获取所有曾参赛的国家
    all_countries = medal_data['Country'].unique()

    programs_data = programs_data.copy()  # 避免修改原始数据
    programs_data['Discipline'] = programs_data['Discipline'].fillna('')
    
    # 创建预测框架
    predict_year = 2028
    predict_df = pd.DataFrame({
        'Year': [predict_year] * len(all_countries),
        'Country': all_countries
    })
    
    # 特征1：运动员数量（使用2024年数据，按2%增长率估算）
    latest_athletes = athletes_data[athletes_data['Year'] == 2024].groupby('Country')['Name'].count().reset_index(name='athletes_2024')
    predict_df = predict_df.merge(latest_athletes, on='Country', how='left')
    predict_df['athletes_count'] = predict_df['athletes_2024'] * (1.02)
    predict_df.drop('athletes_2024', axis=1, inplace=True)

    # 特征2：参与项目数（使用2024年项目设置）
    programs_2024 = programs_data[
        (programs_data['Year'] == 2024) & 
        (programs_data['Inclusion'] > 0)  # 正确引用Inclusion列
    ]['Discipline'].nunique()
    predict_df['num_events'] = programs_2024

    # 特征3：历史奖牌均值
    medal_avg = medal_data[medal_data['Year'].between(2016, 2020)].groupby('Country')['Total'].mean().reset_index(name='past_3_medals_avg')
    predict_df = predict_df.merge(medal_avg, on='Country', how='left').fillna(0)
    
    # === 新增：生成 has_{sport} 特征 ===
    # 获取top_sports列表（需从外部传入或在函数内重新计算）
    for sport in top_sports:
        # 获取2024年各国是否参与过该运动
        latest_participation = athletes_data[athletes_data['Year'] == 2024]
        sport_participation = latest_participation[latest_participation['Sport'] == sport]
        has_sport = sport_participation.groupby('Country').size().reset_index(name=f'has_{sport}')
        has_sport[f'has_{sport}'] = (has_sport[f'has_{sport}'] > 0).astype(int)
        predict_df = predict_df.merge(has_sport, on='Country', how='left').fillna(0)
    
    # === 补充其他缺失特征 ===
    # 特征3：num_high_sports（假设高奖牌项目为top_sports中的项目）
    predict_df['num_high_sports'] = predict_df[[f'has_{sport}' for sport in top_sports]].sum(axis=1)
    
    # 特征8：event_change（假设2028年项目数与2024年相同，变化为0）
    predict_df['event_change'] = 0  
    
    # 特征：is_host（确保正确标记美国为东道主）
    predict_df['is_host'] = (predict_df['Country'] == 'United States').astype(int)
    
    return predict_df


# 执行预测
predict_2028 = prepare_2028_prediction(
    medal_data=pd.concat([medal_train, medal_test]),
    athletes_data=athletes,
    programs_data=programs_long,
    top_sports=top_sports  # 修改函数定义以接收此参数
)
# ========== 预测2028年奖牌 ==========
# 使用最佳模型进行预测
X_2028 = predict_2028[features]  # 确保特征顺序一致

# 执行预测
predicted_medals = grid.best_estimator_.predict(X_2028)

# 将预测结果添加到数据框
predict_2028['Predicted_Gold'] = np.round(predicted_medals[:, 0], 1)  # 金牌预测
predict_2028['Predicted_Total'] = np.round(predicted_medals[:, 1], 1)  # 总奖牌预测

# 处理负值（奖牌数不能为负）
predict_2028['Predicted_Gold'] = predict_2028['Predicted_Gold'].clip(lower=0)
predict_2028['Predicted_Total'] = predict_2028['Predicted_Total'].clip(lower=0)

# 自助法计算预测区间
n_bootstraps = 1000
preds_gold = np.zeros((n_bootstraps, len(X_2028)))
preds_total = np.zeros((n_bootstraps, len(X_2028)))

for i in range(n_bootstraps):
    X_resampled, y_resampled = resample(X_train, y_train)
    model = grid.best_estimator_.fit(X_resampled, y_resampled)
    preds = model.predict(X_2028)
    preds_gold[i, :] = preds[:, 0]
    preds_total[i, :] = preds[:, 1]

# 计算置信区间
ci_lower_gold = np.percentile(preds_gold, 2.5, axis=0)
ci_upper_gold = np.percentile(preds_gold, 97.5, axis=0)

ci_lower_total = np.percentile(preds_total, 2.5, axis=0)
ci_upper_total = np.percentile(preds_total, 97.5, axis=0)

# 将置信区间添加到结果数据框
predict_2028['Gold_lower'] = np.round(ci_lower_gold, 1)
predict_2028['Gold_upper'] = np.round(ci_upper_gold, 1)
predict_2028['Total_lower'] = np.round(ci_lower_total, 1)
predict_2028['Total_upper'] = np.round(ci_upper_total, 1)

# 输出置信区间
print("\n2028年金牌预测置信区间:")
print(predict_2028[['Country', 'Predicted_Gold', 'Gold_lower', 'Gold_upper']])

print("\n2028年总奖牌预测置信区间:")
print(predict_2028[['Country', 'Predicted_Total', 'Total_lower', 'Total_upper']])

# 按总奖牌排序并展示TOP20
result_df = predict_2028[['Country', 'Predicted_Gold', 'Predicted_Total']]\
            .sort_values('Predicted_Total', ascending=False)\
            .reset_index(drop=True)

print("\n2028年夏季奥运会奖牌预测结果（TOP20）：")
print(result_df.head(100))

#画一张TOP20的柱状图（2028年和2024年）
plt.figure(figsize=(13, 7))

# 从 medal_test 中提取 2024 年的奖牌总数
actual_2024_df = medal_test[medal_test['Year'] == 2024][['Country', 'Total']].rename(columns={'Total': 'Actual_Total_2024'})

# 确保只包含前 20 个国家
actual_2024_df = actual_2024_df[actual_2024_df['Country'].isin([
    'United States', 'China', 'ROC', 'Great Britain', 'Russia', 
    'Japan', 'Germany', 'France', 'Australia', 'Italy', 
    'Netherlands', 'Canada', 'South Korea', 'Brazil', 'Spain', 
    'New Zealand', 'Hungary', 'Ukraine', 'Poland', 'Denmark'
])]

# Merge the predicted and actual data for alignment
merged_df = result_df.head(20).merge(actual_2024_df, on='Country')

# Plot the data
bar_width = 0.35
index = np.arange(len(merged_df))

plt.bar(index, merged_df['Predicted_Total'], bar_width, label='Predicted 2028')
plt.bar(index + bar_width, merged_df['Actual_Total_2024'], bar_width, label='Actual 2024')

plt.title('TOP20 2028 Olympic Predicted vs Actual 2024 Total Medals')
plt.xlabel('Country')
plt.ylabel('Total Medals')
plt.xticks(index + bar_width / 2, merged_df['Country'], rotation=45)
plt.legend()

plt.tight_layout()
plt.show()

# 保存预测结果到CSV
result_df.to_csv(f"{data_dir2}\\2028_medal_predictions.csv", index=False)
print("\n预测结果已保存至：2028_medal_predictions.csv")

# 可视化预测分布
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
result_df['Predicted_Gold'].hist(bins=30, color='#00008B')
plt.title('Predicted Gold Medals Distribution')
plt.xlabel('Gold Medals')
plt.ylabel('Country Count')

plt.subplot(1, 2, 2)
result_df['Predicted_Total'].hist(bins=30, color='#FF8C00')
plt.title('Predicted Total Medals Distribution')
plt.xlabel('Total Medals')
plt.tight_layout()
plt.show()

# ========== 增强可视化：模型性能与不确定性 ==========
# 性能指标可视化
plt.figure(figsize=(14, 6))

# 金牌预测性能
plt.subplot(1, 2, 1)
residuals_gold = y_test['Gold'] - y_pred[:,0]
plt.hist(residuals_gold, bins=30, alpha=0.7, color='gold')
plt.axvline(x=0, color='red', linestyle='--')
plt.title('Gold Medal Prediction Error Distribution\nMAE: {:.1f}'.format(mean_absolute_error(y_test['Gold'], y_pred[:,0])))
plt.xlabel('Prediction Error (Actual - Predicted)')
plt.ylabel('Frequency')

# 总奖牌预测性能
plt.subplot(1, 2, 2)
residuals_total = y_test['Total'] - y_pred[:,1]
plt.hist(residuals_total, bins=30, alpha=0.7, color='steelblue')
plt.axvline(x=0, color='red', linestyle='--')
plt.title('Total Medal Prediction Error Distribution\nMAE: {:.1f}'.format(mean_absolute_error(y_test['Total'], y_pred[:,1])))
plt.xlabel('Prediction Error (Actual - Predicted)')
plt.tight_layout()
plt.show()


# ========== 2028预测结果不确定性可视化 ==========
# 计算预测结果的百分位区间
from sklearn.utils import resample

# 自助法计算预测区间
n_bootstraps = 1000
preds_gold = np.zeros((n_bootstraps, len(X_2028)))
preds_total = np.zeros((n_bootstraps, len(X_2028)))

for i in range(n_bootstraps):
    X_resampled, y_resampled = resample(X_train, y_train)
    model = grid.best_estimator_.fit(X_resampled, y_resampled)
    preds = model.predict(X_2028)
    preds_gold[i, :] = preds[:, 0]
    preds_total[i, :] = preds[:, 1]

# 计算置信区间
ci_lower_gold = np.percentile(preds_gold, 2.5, axis=0)
ci_upper_gold = np.percentile(preds_gold, 97.5, axis=0)

ci_lower_total = np.percentile(preds_total, 2.5, axis=0)
ci_upper_total = np.percentile(preds_total, 97.5, axis=0)

# 将置信区间添加到结果数据框
result_df['Gold_lower'] = np.round(ci_lower_gold, 1)
result_df['Gold_upper'] = np.round(ci_upper_gold, 1)
result_df['Total_lower'] = np.round(ci_lower_total, 1)
result_df['Total_upper'] = np.round(ci_upper_total, 1)

# 可视化TOP10国家的预测区间
top10 = result_df.head(10).copy()
top10['Country'] = pd.Categorical(top10['Country'], categories=top10['Country'], ordered=True)

plt.figure(figsize=(14, 8))

# ====================
# 案例分析函数
# ====================
def create_case_study_data():
    """生成日韩案例对比数据"""
    japan_2020 = {
        'Year': 2020, 'Country': 'Japan',
        'athletes_count': 582, 'num_events': 33,
        'num_high_sports': 5, 'past_3_medals_avg': 98, 'is_host': 1
    }
    korea_2020 = {
        'Year': 2020, 'Country': 'South Korea',
        'athletes_count': 532, 'num_events': 30,
        'num_high_sports': 4, 'past_3_medals_avg': 89, 'is_host': 0  
    }
    return pd.DataFrame([japan_2020, korea_2020])

def plot_case_comparison(model):
    """可视化案例对比"""
    case_data = create_case_study_data()
    X_case = case_data[features]
    y_pred = model.predict(X_case)
    
    plt.figure(figsize=(10,6))
    countries = case_data['Country'].values
    plt.bar(countries, y_pred[:,1], alpha=0.6, label='Predicted')
    plt.bar(countries, case_data['past_3_medals_avg'], alpha=0.4, label='Historical Avg')
    plt.title('Case Study: Japan 2020 vs South Korea 2020')
    plt.ylabel('Total Medals')
    plt.legend()
    plt.grid(axis='y', linestyle='--')
    plt.show()

# ====================
# 高尔夫边际影响分析
# ====================
def analyze_golf_impact(model, base_value=30):
    """分析新增高尔夫项目的边际影响"""
    # 创建基准场景（无高尔夫）
    base_features = {
        'athletes_count': 600, 'num_events': base_value,
        'num_high_sports': 6, 'past_3_medals_avg': 110, 'is_host': 1
    }
    
    # 模拟不同运动员数量
    results = []
    for n_players in [10, 20, 30]:
        features = base_features.copy()
        features['num_events'] += 1  # 新增1个项目
        features['athletes_count'] += n_players
        
        # 计算边际影响（奖牌密度0.5）
        impact = n_players * 0.5 * model.named_steps['multioutputregressor'].estimators_[1].coef_[0]
        results.append(impact)
    
    # 可视化
    plt.figure(figsize=(8,5))
    plt.plot([10,20,30], results, marker='o')
    plt.title('Marginal Impact of Adding Golf (USA 2016)')
    plt.xlabel('Number of Golf Athletes')
    plt.ylabel('Additional Total Medals')
    plt.grid(True)
    plt.show()

# ====================
# 主执行流程
# ====================
if __name__ == "__main__":
    plot_case_comparison(grid.best_estimator_)
    analyze_golf_impact(grid.best_estimator_)
    
    # **更新2028预测中的东道主标识**
    predict_2028['is_host'] = (predict_2028['Country'] == 'United States').astype(int)