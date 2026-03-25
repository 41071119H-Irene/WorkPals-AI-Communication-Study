# WorkPals-AI-Communication-Study(N=106)
### 探討生成式 AI 介入溝通環節對職場溝通之影響
### Instructor: [Pecu Tsai](https://github.com/pecu)

## 📝 Project Overview
This repository contains the dataset and analysis for a randomized controlled trial (RCT) investigating how Generative AI-mediated communication (AI-MC) functions as a psychological and cognitive scaffold for newcomers. 

Situated in Taiwan—a professional environment with high power distance and high-context communication—this study examines the intervention's impact on State Anxiety, Communication Self-Efficacy, and Psychological Safety across a 14-day period.

By integrating two cohorts (Batch 1 & Batch 2), this study achieves a robust sample size of **106 participants**, providing high statistical power to explore the psychological and performance-based effects of AI tools.

## 📊 Key Findings (Combined N=106)

### 1. Robust Empowerment in Communication ($p < .001$)
* Cognitive Scaffolding: Participants using AI-MC (Gemini 3 Flash) demonstrated a significant leap in communication self-efficacy.
* **So What?** AI acts as a "performance cue" that simplifies complex social navigation into manageable technical execution, allowing newcomers to overcome the "silence of uncertainty".

### 2. The "Sleeper Effect" on Psychological Safety ($p = .076$)
* Delayed Internalization: While immediate post-test results were subtle, a significant improvement in psychological safety emerged during the delayed follow-up (Day 14)
* **So What?** Psychological safety requires experiential verification. AI initiates a cycle of social validation that slowly internalizes into environmental safety

### 3. Emotional Buffer during High-Pressure Periods
* **Stress Resilience**: In the face of workplace stressors (e.g., pre-holiday deadlines), the **Control Group** exhibited a visible spike in anxiety ($M=3.26$). In contrast, the **AI Group** maintained a stable and lower emotional state ($M=2.78$).
* **So What?** AI serves as an "emotional firewall," protecting users from communication-induced burnout during peak work periods.

### 4. The Experience-Trust Loop ($p = .016$)
* Trust Dynamics: Skeptical users (low initial trust) initially faced "psychological friction".
* Pragmatic Shift: However, they exhibited the most pronounced long-term reduction in anxiety once the tool's functional utility was proven.
* **So What?** Practical success overrides subjective bias. Instrumental utility is the most effective way to transition skeptics into "performance-based" reliers.

---

## 🔬 Methodology & Statistics
* **Participants**: N = 106 ($n_{exp}=55, n_{ctrl}=51$).
* **Analysis**: Linear Mixed-Effects Models (MixedLM), Mediation, and Moderation Analysis.
* **Reliability**: High internal consistency across all scales (Cronbach's $\alpha$: 0.72–0.87).

## 📖 中文摘要
本研究採兩梯次隨機對照實驗（$N=106$），探討生成式 AI 介入對職場溝通之影響。核心發現如下：
1.  **效能賦能**：AI 顯著拉抬溝通效能感，並建立更高的表現基準點。
2.  **職場護盾**：在東亞職場文化中，AI 為正式員工提供「專業護盾」，顯著緩解因在意「面子」而產生的社交風險感 。
3.  **潛伏效應**：心理安全感展現「延遲性趨勢」($p=.076$)，AI 輔助的成功經驗有助於長期心理建設。
4.  **情緒緩衝**：在壓力高峰期，AI 組展現出卓越的情緒韌性，顯著抑制焦慮回彈。
5.  **信任轉化**：實際互動顯著提升對科技的信任感 ($p=.016$)，打破未接觸者的心理防線。
 
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
