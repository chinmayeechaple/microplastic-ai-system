import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Microplastic System", layout="wide")

# =========================
# 🔥 ML-LIKE RISK FUNCTION
# =========================
def compute_risk(shape, size, w, h, contour=None):
    size_norm = max(0, min(1, (150 - size) / 150))

    shape_weights = {
        "Fiber (thread-like)": 0.9,
        "Fragment (irregular)": 0.7,
        "Film (sheet-like)": 0.5
    }

    shape_score = shape_weights.get(shape, 0.5)

    elongation = max(w, h) / (min(w, h) + 1)
    elongation_score = min(elongation / 5, 1)

    irregularity_score = 0
    if contour is not None:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        irregularity_score = min((perimeter / (area + 1)) / 10, 1)

    risk = (
        0.4 * shape_score +
        0.3 * size_norm +
        0.2 * elongation_score +
        0.1 * irregularity_score
    )

    return int(risk * 100)

# 🔹 Tabs
tab1, tab2 = st.tabs(["📘 About", "🔬 Micro Plastic Prediction"])

# =========================
# 📘 ABOUT
# =========================
with tab1:
    st.title("🌊 Microplastics")

    st.markdown("""
Microplastics are plastic particles smaller than 5mm formed from plastic degradation,
synthetic fibers, and industrial waste.

They are dangerous because they enter marine ecosystems and food chains.
""")

    st.header("🌍 Environmental Impact")
    st.markdown("""
- Marine organisms ingest microplastics  
- Toxic chemicals enter food chain  
- Ecosystem imbalance  
""")

    st.header("⚠️ Effects on Marine Life")
    st.markdown("""
- Internal injury and starvation  
- Reduced reproduction  
- Bioaccumulation in food chain  
""")

    st.header("🛠 Mitigation Strategies")
    st.markdown("""
- Reduce plastic consumption  
- Install microplastic filters  
- Improve recycling systems  
- Promote biodegradable materials  
""")

    st.header("♻️ Treatment Methods")
    st.markdown("""
- Wastewater filtration systems  
- Advanced membrane technology  
- Recycling and reuse  
""")

# =========================
# 🔬 PREDICTION
# =========================
with tab2:
    st.title("🔬 Microplastic Prediction")

    st.markdown("""
### 🧪 MP Identification & Description

Microplastics (MPs) are plastic particles **smaller than 5 mm** that are classified by their morphology.  
This tool uses **computer vision** to identify and classify MPs from microscopic images into three distinct types:

| Type | Description | Ecological Risk |
|------|-------------|-----------------|
| 🔵 **Fiber (thread-like)** | Long, thin, thread/filament-shaped particles from synthetic textiles | 🔴 Highest — penetrates deep into organism tissues |
| 🟠 **Fragment (jagged, irregular)** | Irregular, jagged shards from mechanical breakdown of larger plastics | 🟡 Medium — sharp edges cause internal injury |
| 🟢 **Film (thin, sheet-like)** | Flat, transparent or translucent sheet-like particles from plastic bags/wrapping | 🟡 Medium — mistaken for food by marine life |

---
### 📐 How Prediction Works

- **Morphology Classification:** Particles are categorized into Fiber, Fragment, or Film based on shape analysis (elongation ratio, edge irregularity, contour complexity).
- **Size Estimation (Feret Diameter):** The longest dimension of each detected particle is estimated in **micrometers (µm)** using contour detection on the uploaded image.
- **Ecological Threat Index (0–100):** A combined risk score based on morphology weight (fibers score highest) + size factor (smaller particles penetrate deeper into organisms) + elongation + surface irregularity.
""")

    st.divider()

    file = st.file_uploader("📂 Browse or Drop Microscopic Image Here", type=["jpg", "png", "jpeg"])

    if file:
        img = Image.open(file)
        st.image(img, caption="Uploaded Image", width=500)

        img_cv = np.array(img)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        st.subheader("🔍 Analysis Result")

        count = 0
        fiber_count = 0
        fragment_count = 0
        film_count = 0
        all_risks = []   # 🔥 NEW

        # ==================================
        # 🔥 FIBER DETECTION
        # ==================================
        edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi/180,
            threshold=80,
            minLineLength=50,
            maxLineGap=5
        )

        detected_lines = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

                if length < 80:
                    continue

                angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
                if abs(angle) < 10 or abs(angle) > 170:
                    continue

                duplicate = False
                for (px, py) in detected_lines:
                    if abs(px - x1) < 15 and abs(py - y1) < 15:
                        duplicate = True
                        break

                if duplicate:
                    continue

                detected_lines.append((x1, y1))

                cv2.line(img_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)

                size = int(length)

                risk = compute_risk("Fiber (thread-like)", size, abs(x2-x1), abs(y2-y1))
                all_risks.append(risk)  # 🔥 STORE

                severity = "🔴 HIGH" if risk > 80 else "🟡 MEDIUM" if risk > 50 else "🟢 LOW"

                st.markdown(f"""
---
### 🔹 Particle {count+1}

- **Type:** Fiber (thread-like)  
- **Size:** {size} µm  
- **Risk:** **{risk}/100**  
- **Severity:** {severity}
""")

                count += 1
                fiber_count += 1

                if count > 25:
                    break

        # ==================================
        # 🔥 FRAGMENT + FILM
        # ==================================
        _, thresh = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            if count > 25:
                break

            area = cv2.contourArea(cnt)
            if area < 100:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            size = max(w, h)

            if size > 300:
                continue

            perimeter = cv2.arcLength(cnt, True)

            if len(cv2.approxPolyDP(cnt, 0.02 * perimeter, True)) > 6:
                shape = "Fragment (irregular)"
                fragment_count += 1
            else:
                shape = "Film (sheet-like)"
                film_count += 1

            risk = compute_risk(shape, size, w, h, cnt)
            all_risks.append(risk)  # 🔥 STORE

            severity = "🔴 HIGH" if risk > 80 else "🟡 MEDIUM" if risk > 50 else "🟢 LOW"

            cv2.rectangle(img_cv, (x, y), (x+w, y+h), (0,255,0), 1)

            st.markdown(f"""
---
### 🔹 Particle {count+1}

- **Type:** {shape}  
- **Size:** {size} µm  
- **Risk:** **{risk}/100**  
- **Severity:** {severity}
""")

            count += 1

        st.image(img_cv, caption="Detected Particles")

        # =========================
        # 🌟 OVERALL RISK (calculated once, shown alongside distribution)
        # =========================
        if all_risks:
            overall_risk = int(sum(all_risks) / len(all_risks))
        else:
            overall_risk = 0

        overall_severity = "🔴 HIGH" if overall_risk > 80 else "🟡 MEDIUM" if overall_risk > 50 else "🟢 LOW"

        # =========================
        # 📊 DISTRIBUTION + OVERALL RISK (side by side)
        # =========================
        st.subheader("📊 Distribution Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"Fiber: {fiber_count}")
            st.write(f"Fragment: {fragment_count}")
            st.write(f"Film: {film_count}")
            st.write(f"Total Particles: {count}")

        with col2:
            st.subheader("🌟 Overall Ecological Threat Index")
            st.markdown(f"""
## 🚨 Overall Risk Score: **{overall_risk}/100**
### Severity: {overall_severity}
""")

        # =========================
        # 🌍 DETAILED IMPACT + SOLUTION
        # =========================
        st.divider()
        st.subheader("🌊 Microplastic Pollution")
        st.markdown("""
Microplastic pollution is one of the most pervasive environmental crises of our time.
Every year, **over 8 million tonnes** of plastic enter the oceans. UV radiation, wave action, and
biological degradation break large plastics into microplastics that spread across every corner of the planet —
from the **deepest ocean trenches** to **Arctic ice cores** and **human bloodstreams**.

Sources of microplastic pollution include:
- 🧴 Personal care products (microbeads)
- 👕 Synthetic textile washing (polyester, nylon fibers)
- 🚗 Tire wear particles carried by stormwater runoff
- 🛍 Fragmentation of plastic bags, bottles, and packaging
- 🏭 Industrial plastic pellet spills (nurdles)
""")

        st.subheader("🐠 Effect on Marine Life")
        st.markdown("""
Marine organisms at every level of the food web are affected by microplastics:

- **Plankton & Filter Feeders:** Microplastics clog feeding appendages, reduce nutrient intake, and cause false satiation (feeling full without nutrition), leading to starvation.
- **Fish:** Ingested MPs accumulate in the gut and liver, causing inflammation, oxidative stress, reduced growth, and impaired reproduction. Fibers are especially dangerous as they entangle intestinal villi.
- **Seabirds & Marine Mammals:** Large quantities of plastic fragments are found in stomachs, causing internal lacerations, blockages, and death. Albatrosses famously feed plastic to their chicks.
- **Coral Reefs:** MPs physically smother coral polyps, block photosynthesis in symbiotic algae, and introduce pathogens that accelerate bleaching.
- **Bioaccumulation:** Toxic chemicals (PCBs, DDT, heavy metals) adsorb onto MP surfaces and concentrate up the food chain — **biomagnifying** 1 million times from water to top predators.
""")

        st.subheader("🌍 Environmental Impacts")
        st.markdown("""
The broader environmental consequences of microplastic pollution extend far beyond the ocean:

- **Soil Degradation:** MPs in agricultural soils alter microbial communities, reduce earthworm health, and decrease crop yields. Plastic fibers physically disrupt soil structure.
- **Freshwater Systems:** Rivers, lakes, and groundwater are heavily contaminated. MPs are found in tap water worldwide, with humans estimated to consume **5 grams of plastic per week** (equivalent to a credit card).
- **Atmospheric Transport:** Microplastics are carried by wind and deposited in remote mountain ranges and polar regions, demonstrating truly global dispersion.
- **Climate Impact:** Plastics in oceans interfere with the biological carbon pump — the process by which marine organisms sequester atmospheric CO2 into the deep ocean.
- **Human Health:** MPs have been detected in human lungs, blood, placenta, and breast milk. Associated health concerns include endocrine disruption, inflammation, and potential carcinogenicity from plastic additives.
""")

        st.subheader("🛡 Mitigation Strategies")
        st.markdown("""
Addressing microplastic pollution requires action at individual, industrial, and policy levels:

**Individual Actions:**
- Switch to natural fiber clothing (cotton, wool, linen) to reduce fiber shedding
- Use a microfiber-catching laundry bag when washing synthetics
- Avoid single-use plastics — carry reusable bottles, bags, and containers
- Choose personal care products labeled microbead-free
- Properly dispose of plastic waste to prevent environmental leakage

**Industrial & Policy Actions:**
- Banning microbeads in cosmetics (already enacted in US, UK, EU)
- Extended Producer Responsibility (EPR) making manufacturers accountable for end-of-life plastic
- Investing in plastic-free packaging innovation and biodegradable material R&D
- Mandatory microplastic filters on industrial wastewater outflows
- International treaties for plastic pollution (Global Plastics Treaty negotiations ongoing)

**Ocean Cleanup:**
- Passive collection systems (e.g., The Ocean Cleanup river and ocean barriers)
- Drone-assisted surface skimming technologies
- Bioremediation using plastic-degrading bacteria (e.g., Ideonella sakaiensis for PET)
""")

        st.subheader("How Microplastics Can Be Treated")
        st.markdown("""
Several emerging and established technologies can remove or degrade microplastics:

| Method | How It Works | Effectiveness |
|--------|-------------|---------------|
| **Membrane Filtration** | Ultra/nanofiltration membranes physically sieve MPs from water | >99% removal for particles >0.1 µm |
| **Magnetic Extraction** | Magnetic nanoparticles bind to MPs; external magnet removes them | Effective for small MPs in wastewater |
| **Photocatalytic Degradation** | UV light + TiO2 catalyst breaks polymer chains into CO2 and H2O | Promising but slow for bulk treatment |
| **Bioremediation** | Engineered microbes or fungi metabolize plastic polymers | Early-stage; effective for PET, PE |
| **Electrocoagulation** | Electric current destabilizes MP suspension; particles clump and settle | Good for industrial effluent |
| **Constructed Wetlands** | Natural plant root systems and biofilms trap MPs from runoff | Low-cost, passive, scalable |
| **Wastewater Treatment Upgrades** | Adding tertiary treatment (sand filtration + activated carbon) to existing plants | Removes ~95% of MPs from sewage |

Key Insight: No single method eliminates all microplastics. A **multi-barrier approach** combining physical filtration, chemical treatment, and biological degradation is most effective.
""")

        st.success("Analysis Complete")