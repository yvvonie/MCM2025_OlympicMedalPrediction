# 2025 MCM Problem C - Honorable Mention Award
# Who Will Lead the Olympic Medal Count? A Prediction Based on Historical Trials

![Our Work](https://github.com/user-attachments/assets/c40aef70-456a-4223-bd64-e2eb037526b1)

## 项目概述 | Project Overview

奥运会中的"奖牌榜"是一个盲盒，直到比赛结束才会揭晓。我们的研究旨在基于历史数据开发预测模型，以预测未知的2028年奖牌榜，评估一位伟大教练的转会将如何影响一国的奖牌数，以及主办国的优势将如何影响奖项的分配。
The "medal table" in the Olympics is a blind box not open until the end of the game. Topeek inside this enigma, we developed models based on historical data to predict the unknown2028 medal table, assess how the transfer of a great coach will make a difference to a country'smedal counts, and how the host country's advantage may impact the distributions of awards.

## 研究方法与结果 | Methods and Results

### 奥林匹克奖牌预测模型 | Olympic Medal Prediction Model (Question 1)

我们使用岭回归结合TimeSeriesSplit交叉验证来构建一个奥林匹克奖牌数预测模型。该模型：
- 突出了对奖牌数有重大影响的项目，如田径和游泳
- 揭示了主办国的优势
- 预测2028年美国将因主办国优势增加奖牌数

We developed an Olympic medal count prediction model using ridge regression with TimeSeriesSplit cross-validation. This model:
- Highlights events with significant impact on medal counts, such as athletics and swimming
- Reveals the host country advantage
- Predicts that the United States will increase its medal count in 2028 due to the host country advantage
![Figure_7](https://github.com/user-attachments/assets/d277cd0a-9eaf-4053-b2a1-eb12b6bcad90)


### "教练效应"评价分析模型 | "Coach Effect" Evaluation Model (Question 2)

该模型通过差分框架量化"伟大教练"对奥运奖牌结果的影响：
- 研究"伟大教练"效应的存在性，即精英教练的转换对一国奖牌的影响
- 使用郎平等多位著名教练的案例进行分析
- 为中国、英国和美国提供教练投资策略建议

This model quantifies the impact of "great coaches" on Olympic medal outcomes through a differential framework:
- Examines the existence of the "great coach" effect, where elite coach transfers influence a country's medals
- Analyzes cases of several famous coaches, including Lang Ping
- Provides recommendations on coaching investment strategies for China, UK, and USA

### Further Insights (Question 3)

- 使用时间窗口比较方法调查"主办国"效应，证实了其积极影响
- 应用投资回报率(ROI)分数展示新兴运动项目的潜力
- 向国家奥林匹克委员会提供投资建议

- Investigates the "host country" effect using a time-window comparison method, confirming its positive impact
- Applies Return on Investment (ROI) scores to demonstrate the potential of emerging sports
- Provides investment recommendations to National Olympic Committees

## 结论 | Conclusions

本研究：
- 稳健地预测了各国未来的奥运奖牌数量
- 证明了"伟大教练"效应的存在并量化了其影响
- 分析了作为"主办国"的优势
- 发现了新兴体育项目的投资潜力
- 为不同国家和国家奥委会提供了教练策略建议和新兴体育项目推荐

This paper aims to predict the future Olympic medal numbers of each country robustly. It also proved the existence of the "Great Coach" effect and discovered its influences. Further. it analyzed the effect of being "Host Country", and discovered the potential of emerging sports.Finally, it provided coaching strategy suggestions and emerging sports recommendations fordifferent countries and country Olympic committees.

## 关键词 | Keywords

奥运奖牌预测模型; 教练效应; 主办国; 新兴运动

Olympic Medal Prediction Model; Coach Effect; Host Country; Emerging Sports
