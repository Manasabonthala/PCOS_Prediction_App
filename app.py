import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("pcos_ann_model.keras")

# Load scaler
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="PCOS Prediction App",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 PCOS Prediction using Deep Learning")

st.markdown("""
This application predicts the likelihood of **Polycystic Ovary Syndrome (PCOS)**
using an Artificial Neural Network trained on clinical patient data.
""")

st.sidebar.title("🩺 PCOS Prediction")

st.sidebar.markdown("""
### About
This application predicts the likelihood of PCOS using a Deep Learning (ANN) model.

### Instructions
- Fill all patient details.
- Click Predict PCOS.
- View prediction and recommendations.

---
Developed by ** Bonthala Manasa** """)

st.write("Enter the patient's details below to predict whether PCOS is likely or not.")

# =====================================================
# Personal Details
# =====================================================

st.header("👤 Personal Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (Years)", 10, 60, 25)
    weight = st.number_input("Weight (Kg)", 20.0, 150.0, 60.0)
    height = st.number_input("Height (cm)", 100.0, 220.0, 160.0)
    bmi = st.number_input("BMI", 10.0, 60.0, 22.0)

with col2:
    pulse = st.number_input("Pulse Rate (bpm)", value=72)
    rr = st.number_input("Respiratory Rate", value=18)
    hb = st.number_input("Hemoglobin (g/dL)", value=12.0)

# =====================================================
# Menstrual & Medical Details
# =====================================================
st.header("🩸 Menstrual & Medical Details")

col1, col2 = st.columns(2)

with col1:

    cycle = st.selectbox("Cycle Regular?", ["Regular", "Irregular"])

    # Match training data encoding
    cycle = 2 if cycle == "Regular" else 4

    cycle_length = st.number_input("Cycle Length (days)", value=28)
    marriage = st.number_input("Marriage Duration (Years)", value=0)

    pregnant = st.selectbox("Pregnant?", ["No", "Yes"])
    pregnant = 1 if pregnant == "Yes" else 0

    abortions = st.number_input("No. of Abortions", value=0)

    beta_hcg1 = st.number_input("I Beta-HCG (mIU/mL)", value=0.0)
    beta_hcg2 = st.number_input("II Beta-HCG (mIU/mL)", value=0.0)


with col2:
    fsh = st.number_input("FSH (mIU/mL)", value=5.0)
    lh = st.number_input("LH (mIU/mL)", value=5.0)
    fsh_lh = st.number_input("FSH/LH Ratio", value=1.0)

    hip = st.number_input("Hip (inch)", value=36.0)
    waist = st.number_input("Waist (inch)", value=32.0)
    whr = st.number_input("Waist-Hip Ratio", value=0.88)

    tsh = st.number_input("TSH (mIU/L)", value=2.5)
    amh = st.number_input("AMH (ng/mL)", value=3.0)
    prl = st.number_input("PRL (ng/mL)", value=15.0)
    vit_d3 = st.number_input("Vitamin D3 (ng/mL)", value=30.0)
    prg = st.number_input("Progesterone (ng/mL)", value=1.0)
    rbs = st.number_input("Random Blood Sugar (mg/dL)", value=90.0)

# =====================================================
# Symptoms
# =====================================================

st.header("⚠️ Symptoms")

col1, col2 = st.columns(2)

with col1:
    weight_gain = st.selectbox("Weight Gain", [0,1])
    hair_growth = st.selectbox("Excess Hair Growth", [0,1])
    skin_darkening = st.selectbox("Skin Darkening", [0,1])
    hair_loss = st.selectbox("Hair Loss", [0,1])

with col2:
    pimples = st.selectbox("Pimples", [0,1])
    fast_food = st.selectbox("Fast Food Habit", [0,1])
    exercise = st.selectbox("Regular Exercise", [0,1])

# =====================================================
# Blood Pressure & Ultrasound
# =====================================================

st.header("🩺 Blood Pressure & Ultrasound")

col1, col2 = st.columns(2)

with col1:
    bp_systolic = st.number_input("BP Systolic (mmHg)", value=120)
    bp_diastolic = st.number_input("BP Diastolic (mmHg)", value=80)

    follicle_l = st.number_input("Follicle No. (L)", value=5)
    follicle_r = st.number_input("Follicle No. (R)", value=5)

with col2:
    avg_follicle_l = st.number_input("Avg. Follicle Size (L) (mm)", value=5.0)
    avg_follicle_r = st.number_input("Avg. Follicle Size (R) (mm)", value=5.0)

    endometrium = st.number_input("Endometrium (mm)", value=8.0)

# =====================================================
# Blood Group
# =====================================================

st.header("🩸 Blood Group")

blood_group = st.selectbox(
    "Select Blood Group",
    ["A+","A-","B+","B-","O+","O-","AB+","AB-"]
)

bg11 = 1 if blood_group=="A+" else 0
bg12 = 1 if blood_group=="A-" else 0
bg13 = 1 if blood_group=="B+" else 0
bg14 = 1 if blood_group=="B-" else 0
bg15 = 1 if blood_group=="O+" else 0
bg16 = 1 if blood_group=="O-" else 0
bg17 = 1 if blood_group=="AB+" else 0
bg18 = 1 if blood_group=="AB-" else 0

# =====================================================
# Prediction
# =====================================================

if st.button("🔍 Predict PCOS"):

    # -------------------------------
    # Create Input DataFrame
    # -------------------------------
    features = pd.DataFrame([[

        age,
        weight,
        height,
        bmi,
        pulse,
        rr,
        hb,
        cycle,
        cycle_length,
        marriage,
        pregnant,
        abortions,
        beta_hcg1,
        beta_hcg2,
        fsh,
        lh,
        fsh_lh,
        hip,
        waist,
        whr,
        tsh,
        amh,
        prl,
        vit_d3,
        prg,
        rbs,
        weight_gain,
        hair_growth,
        skin_darkening,
        hair_loss,
        pimples,
        fast_food,
        exercise,
        bp_systolic,
        bp_diastolic,
        follicle_l,
        follicle_r,
        avg_follicle_l,
        avg_follicle_r,
        endometrium,
        bg11,
        bg12,
        bg13,
        bg14,
        bg15,
        bg16,
        bg17,
        bg18

    ]], columns=[

        ' Age (yrs)',
        'Weight (Kg)',
        'Height(Cm) ',
        'BMI',
        'Pulse rate(bpm) ',
        'RR (breaths/min)',
        'Hb(g/dl)',
        'Cycle(R/I)',
        'Cycle length(days)',
        'Marraige Status (Yrs)',
        'Pregnant(Y/N)',
        'No. of aborptions',
        '  I   beta-HCG(mIU/mL)',
        'II    beta-HCG(mIU/mL)',
        'FSH(mIU/mL)',
        'LH(mIU/mL)',
        'FSH/LH',
        'Hip(inch)',
        'Waist(inch)',
        'Waist:Hip Ratio',
        'TSH (mIU/L)',
        'AMH(ng/mL)',
        'PRL(ng/mL)',
        'Vit D3 (ng/mL)',
        'PRG(ng/mL)',
        'RBS(mg/dl)',
        'Weight gain(Y/N)',
        'hair growth(Y/N)',
        'Skin darkening (Y/N)',
        'Hair loss(Y/N)',
        'Pimples(Y/N)',
        'Fast food (Y/N)',
        'Reg.Exercise(Y/N)',
        'BP _Systolic (mmHg)',
        'BP _Diastolic (mmHg)',
        'Follicle No. (L)',
        'Follicle No. (R)',
        'Avg. F size (L) (mm)',
        'Avg. F size (R) (mm)',
        'Endometrium (mm)',
        'Blood Group_11',
        'Blood Group_12',
        'Blood Group_13',
        'Blood Group_14',
        'Blood Group_15',
        'Blood Group_16',
        'Blood Group_17',
        'Blood Group_18'

    ])

    # -------------------------------
    # Scale Features
    # -------------------------------
    features_scaled = scaler.transform(features)

    # -------------------------------
    # Predict
    # -------------------------------
    probability = float(model.predict(features_scaled, verbose=0)[0][0])

    # -------------------------------
    # Prediction Result
    # -------------------------------
    st.divider()
    st.subheader("📊 Prediction Result")

    if probability >= 0.5:

        st.error("🔴 High Risk of PCOS")

        st.metric(
            "Prediction Confidence",
            f"{probability*100:.2f}%"
        )

        st.warning("""
### ⚠️ Recommendations
- Consult a Gynecologist
- Maintain a healthy weight
- Exercise regularly
- Reduce junk food
- Monitor menstrual cycle
""")

    else:

        st.success("🟢 Low Risk of PCOS")

        st.metric(
            "Prediction Confidence",
            f"{(1-probability)*100:.2f}%"
        )

        st.info("""
### ✅ Recommendations
- Continue healthy lifestyle
- Eat balanced meals
- Exercise regularly
- Stay hydrated
- Schedule regular checkups
""")

    # -------------------------------
    # Prediction Probability
    # -------------------------------
    st.divider()
    st.subheader("📈 Prediction Probability")

    st.progress(min(max(probability, 0.0), 1.0))

    st.write(f"**PCOS Probability:** {probability:.2%}")
    st.write(f"**No PCOS Probability:** {(1-probability):.2%}")

    # -------------------------------
    # Patient Summary
    # -------------------------------
    st.divider()
    st.subheader("📋 Patient Summary")

    summary = features.T
    summary.columns = ["Value"]

    st.dataframe(summary, use_container_width=True)

# =====================================================
# Footer
# =====================================================

st.markdown("---")
st.caption(
    "👩‍💻 Developed by Bonthala Manasa | 🧠 Deep Learning Based PCOS Prediction System"
)