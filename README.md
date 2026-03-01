# WorkPals-AI-Communication-Study(N=106)
### 探討生成式 AI 介入溝通環節對職場溝通之影響
### Instructor: [Pecu Tsai](https://github.com/pecu)

## 📝 Project Overview
This repository contains the dataset and analysis for a randomized controlled trial (RCT) investigating the impact of **Generative AI interventions** on workplace communication. By integrating two cohorts (Batch 1 & Batch 2), this study achieves a robust sample size of **106 participants**, providing high statistical power to explore the psychological and performance-based effects of AI tools.

## 📊 Key Findings (Combined N=106)

### 1. Robust Empowerment in Communication ($p < .001$)
* **Performance Leap**: Both groups improved over time, but the **AI Group** reached a significantly higher plateau ($M_{delay}=3.89$) compared to the Control Group ($M_{delay}=3.56$).
* **So What?** AI isn't just a "short-term toy"; it sets a higher performance baseline for early-career professionals, accelerating their workplace integration.

### 2. The "Sleeper Effect" on Psychological Safety ($p = .076$)
* **Delayed Impact**: While immediate results (post-test) were subtle, a **marginal significance** emerged in the 14-day follow-up. The AI group showed a stronger upward trend in feeling safe to take interpersonal risks.
* **So What?** Psychological safety is a "slow-burn" variable. AI-assisted success experiences act as a catalyst that requires time to internalize into team-level safety.

### 3. Emotional Buffer during High-Pressure Periods
* **Stress Resilience**: In the face of workplace stressors (e.g., pre-holiday deadlines), the **Control Group** exhibited a visible spike in anxiety ($M=3.26$). In contrast, the **AI Group** maintained a stable and lower emotional state ($M=2.78$).
* **So What?** AI serves as an "emotional firewall," protecting users from communication-induced burnout during peak work periods.

### 4. The Experience-Trust Loop ($p = .016$)
* **Significant Trust Gain**: Direct interaction with AI led to a significant increase in technology trust, whereas non-users (Control Group) showed stagnation or slight decline.
* **So What?** Familiarity breeds confidence. Practical utility is the strongest driver for Human-AI collaboration trust.

---

## 🔬 Methodology & Statistics
* **Participants**: N = 106 ($n_{exp}=55, n_{ctrl}=51$).
* **Analysis**: Linear Mixed-Effects Models (MixedLM), Mediation, and Moderation Analysis.
* **Reliability**: High internal consistency across all scales (Cronbach's $\alpha$: 0.72–0.87).

## 📖 中文摘要
本研究採兩梯次隨機對照實驗（$N=106$），探討生成式 AI 介入對職場溝通之影響。核心發現如下：
1.  **效能賦能**：AI 顯著拉抬溝通效能感，並建立更高的表現基準點。
2.  **潛伏效應**：心理安全感展現「延遲性趨勢」($p=.076$)，AI 輔助的成功經驗有助於長期心理建設。
3.  **情緒緩衝**：在壓力高峰期，AI 組展現出卓越的情緒韌性，顯著抑制焦慮回彈。
4.  **信任轉化**：實際互動顯著提升對科技的信任感 ($p=.016$)，打破未接觸者的心理防線。
 
 -社會新鮮人更有用，實習生因為需要承受的風險比較低，所以效果沒那麼好到顯著
---

## 🛠️ Usage
Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

Optional (export Plotly figures to PNG in addition to HTML):

```bash
pip install -U kaleido
```

## Run

Place your Excel file in the repo root (default: `data_all.xlsx`) and run:

```bash
python analysis.py
```

## 📜 Reference
[Analysis Code](https://github.com/peculab/genai-psafety)
