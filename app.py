import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
import pandas as pd

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Exam Predictor",
    page_icon="🎓",
    layout="wide"
)

# ==================== TITLE ====================
st.title("🎓 Exam Predictor")
st.markdown("### Binomial Distribution for Exam Success")
st.markdown("Calculate the probability of passing an exam based on your performance!")
st.markdown("---")

# ==================== SIDEBAR ====================
st.sidebar.header("⚙️ Exam Parameters")

# n - Number of questions
n = st.sidebar.number_input(
    "📝 Number of questions (n):",
    min_value=1,
    max_value=100,
    value=10,
    step=1,
    help="How many questions are on the exam?"
)

# p - Probability of correct answer
p = st.sidebar.slider(
    "🎯 Probability of correct answer (p):",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.01,
    help="Your chance of answering each question correctly"
)

# k - Minimum correct answers to pass
k = st.sidebar.number_input(
    "✅ Minimum correct to pass (k):",
    min_value=0,
    max_value=n,
    value=min(6, n),
    step=1,
    help="How many correct answers do you need to pass?"
)

# Mode
mode = st.sidebar.radio(
    "📊 Calculation type:",
    options=["P(X = k)", "P(X >= k)", "P(X <= k)"],
    index=1
)

# ==================== EXPLANATION IN SIDEBAR ====================
st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 What do these mean?")

st.sidebar.info(f"""
**n = {n}** → There are **{n} questions** on the exam.

**p = {p:.2f}** → Each question has a **{p*100:.0f}% chance** of being correct.

**k = {k}** → You need **at least {k} correct** answers to pass.
""")

# ==================== CALCULATE ====================
dist = binom(n, p)

if mode == "P(X = k)":
    result = dist.pmf(k)
    label = f"P(X = {k})"
    explanation = f"Probability of getting EXACTLY {k} correct answers"
elif mode == "P(X >= k)":
    result = dist.sf(k - 1)
    label = f"P(X >= {k})"
    explanation = f"Probability of getting AT LEAST {k} correct answers (PASS)"
else:
    result = dist.cdf(k)
    label = f"P(X <= {k})"
    explanation = f"Probability of getting AT MOST {k} correct answers"

# ==================== MAIN RESULT ====================
st.markdown(f"## 📊 {label} = **{result*100:.2f}%**")
st.markdown(f"*{explanation}*")

st.markdown("---")

# ==================== KEY METRICS ====================
st.markdown("### 📐 Key Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="📊 E[X] — Expected correct",
        value=f"{n*p:.2f}",
        help="Average number of correct answers you expect"
    )
    st.caption(f"Formula: n × p = {n} × {p} = {n*p:.2f}")

with col2:
    st.metric(
        label="🎯 Pass Threshold",
        value=f"{k} correct",
        help="Minimum required to pass"
    )
    st.caption(f"You need at least {k} out of {n}")

with col3:
    st.metric(
        label="📈 Your Chance to Pass",
        value=f"{dist.sf(k-1)*100:.1f}%",
        help="Probability of passing the exam"
    )
    st.caption(f"Based on your performance")

st.markdown("---")

# ==================== EXPLANATION BOX ====================
st.markdown("### 💡 Explanation")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    **📝 What is n = {n}?**
    
    This is the **total number of questions** on the exam.
    You will answer all **{n} questions**, one by one.
    """)
    
    st.markdown(f"""
    **🎯 What is p = {p:.2f}?**
    
    This is your **probability of answering correctly**.
    In simple terms: for every question, you have a **{p*100:.0f}% chance** to get it right.
    """)

with col2:
    st.markdown(f"""
    **✅ What is k = {k}?**
    
    This is the **minimum number of correct answers** you need.
    If you get **{k} or more** correct → you PASS. ✅
    """)
    
    st.markdown(f"""
    **📊 What does {result*100:.2f}% mean?**
    
    If **100 students** with the same skill level took this exam,
    about **{round(result*100)}** would get this result.
    """)

st.markdown("---")

# ==================== CHART ====================
st.markdown("### 📈 Probability Distribution")

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(0, n + 1)
y = dist.pmf(x) * 100

# Colors
colors = []
for i in x:
    highlight = False
    if mode == "P(X = k)" and i == k:
        highlight = True
    elif mode == "P(X >= k)" and i >= k:
        highlight = True
    elif mode == "P(X <= k)" and i <= k:
        highlight = True
    colors.append('#764ba2' if highlight else '#c5cae9')

bars = ax.bar(x, y, color=colors, edgecolor='white', linewidth=1.5)

# Labels
for bar, val in zip(bars, y):
    if val > 0.5:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f'{val:.1f}%',
            ha='center',
            fontsize=9,
            fontweight='bold'
        )

# E[X] line
ax.axvline(x=n*p, color='red', linestyle='--', alpha=0.7, label=f'E[X] = {n*p:.2f}')
# Pass line
ax.axvline(x=k-0.5, color='green', linestyle=':', alpha=0.7, label=f'Pass = {k}')

ax.set_xlabel('Number of correct answers', fontsize=12, fontweight='bold')
ax.set_ylabel('Probability (%)', fontsize=12, fontweight='bold')
ax.set_title(f'Exam Performance Distribution (n={n}, p={p})', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.legend()
ax.set_xticks(range(0, n + 1, max(1, n // 20)))

st.pyplot(fig)

st.markdown("---")

# ==================== TABLE ====================
st.markdown("### 📋 Full Probability Table")

st.markdown(f"""
**How to read this table:**
- **k** — Number of correct answers
- **P(X = k)** — Probability of getting exactly k correct
- **P(X <= k)** — Probability of getting k or fewer
- **P(X >= k)** — Probability of getting k or more (PASS if k >= {k})
""")

table_data = []
for i in range(n + 1):
    pass_mark = "✅ PASS" if i >= k else "❌ FAIL"
    table_data.append({
        "k": i,
        "P(X = k)": f"{dist.pmf(i)*100:.2f}%",
        "P(X <= k)": f"{dist.cdf(i)*100:.2f}%",
        "P(X >= k)": f"{dist.sf(i-1)*100:.2f}%",
        "Status": pass_mark
    })

st.dataframe(
    pd.DataFrame(table_data),
    use_container_width=True,
    hide_index=True,
    height=400
)

st.markdown("---")

# ==================== EXAMPLE SCENARIO ====================
st.markdown("### 📚 Real Example")

st.success(f"""
**Scenario:** A student takes an exam with **{n} questions**.
They have a **{p*100:.0f}% chance** of answering each question correctly.
To pass, they need **at least {k} correct** answers.

**Question:** What is their probability of passing?

**Answer:** P(X ≥ {k}) = **{dist.sf(k-1)*100:.2f}%**

**Interpretation:** 
- If 100 students with the same skill take this exam:
  - About **{round(dist.sf(k-1)*100)} students** will PASS ✅
  - About **{100 - round(dist.sf(k-1)*100)} students** will FAIL ❌
""")

# ==================== FORMULA ====================
with st.expander("📐 Mathematical Formula"):
    st.latex(r"P(X = k) = \binom{n}{k} \cdot p^k \cdot (1-p)^{n-k}")
    st.markdown(f"""
    **Where:**
    - **n = {n}** — Number of questions
    - **p = {p:.2f}** — Probability of correct answer
    - **k = {k}** — Number of correct answers
    - **C(n,k)** — Combination (how many ways to choose k from n)
    """)

# ==================== FOOTER ====================
st.markdown("---")
st.caption("🎓 Exam Predictor - Binomial Distribution for Exam Success")
