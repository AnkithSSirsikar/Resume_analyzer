import streamlit as st

from analyzer import analyze, extract_text

st.set_page_config(page_title="Resume Analyzer", page_icon="📄", layout="wide")

st.title("📄 Resume Analyzer")
st.caption("Score your resume against a job description and get specific fixes.")

left, right = st.columns(2)

with left:
    st.subheader("1. Your resume")
    uploaded = st.file_uploader("Upload PDF, DOCX or TXT", type=["pdf", "docx", "txt"])
    pasted = st.text_area("...or paste the resume text", height=200)

with right:
    st.subheader("2. Job description")
    jd_text = st.text_area("Paste the job description", height=330)

if st.button("Analyze my resume", type="primary"):
    resume_text = ""
    if uploaded is not None:
        try:
            resume_text = extract_text(uploaded.read(), uploaded.name)
        except Exception as e:
            st.error(f"Could not read the file: {e}")
    if not resume_text.strip():
        resume_text = pasted

    if not resume_text.strip():
        st.warning("Please upload or paste your resume.")
    elif not jd_text.strip():
        st.warning("Please paste a job description.")
    else:
        result = analyze(resume_text, jd_text)

        st.divider()
        if result.overall >= 75:
            verdict = "Strong match"
        elif result.overall >= 50:
            verdict = "Moderate match - worth improving"
        else:
            verdict = "Weak match - needs work"

        top, bars = st.columns([1, 2])
        with top:
            st.metric("Overall score", f"{result.overall} / 100")
            st.write(f"**{verdict}**")
            st.caption(
                f"{result.stats['word_count']} words | "
                f"{result.stats['action_verbs']} action verbs | "
                f"{result.stats['quantified_achievements']} quantified achievements"
            )
        with bars:
            for label, value in result.breakdown.items():
                st.write(f"{label}: **{value}%**")
                st.progress(min(max(value, 0), 100))

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("✅ Matched skills")
            st.write(", ".join(result.matched_skills) or "None found")
            st.subheader("📑 Sections detected")
            for name, found in result.sections_found.items():
                st.write(f"{'✅' if found else '❌'} {name.title()}")
        with c2:
            st.subheader("❌ Missing skills")
            st.write(", ".join(result.missing_skills) or "None - great!")
            st.subheader("🔑 Missing keywords")
            st.write(", ".join(result.missing_keywords) or "None - great!")

        st.divider()
        st.subheader("💡 Suggestions")
        for tip in result.suggestions:
            st.markdown(f"- {tip}")