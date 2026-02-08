from core.parser import CVPipeline
from core.scorer import CVScorer

def main():
    # 1. CONFIG INPUT
    pdf_folder = "data/cvs"

    job_title = "Senior Accountant / Tax Supervisor"
    job_description = """Handle full set of accounts, monthly closing, and financial reporting.
Experienced in corporate tax compliance (VAT, PPh), auditing, and financial analysis.
Familiar with IFRS and SAP system."""
    
    required_skills =[
    "accounting",
    "microsoft excel",
    "financial reporting",
    "taxation"
    ]
    
    highlights = [
    "SAP",
    "IFRS",
    "Tax Compliance",
    "Auditing",
    "CPA",
    "Tax Planning"
    ]
    
    weights = {
        "experience": 0.4,
        "skills": 0.3,
        "summary": 0.2,
        "education": 0.1
    }

    # 2. PARSE CV
    parser = CVPipeline()
    df = parser.run(pdf_folder)

    # 3. SCORE CV
    scorer = CVScorer(
        job_title,
        job_description,
        required_skills,
        highlights,
        weights
    )

    result = scorer.score_dataframe(df)

    # 4. OUTPUT
    print(result[[
        "cv_id",
        "total_score",
        "score_skills",
        "score_experience_final",
        "score_summary_final",
        "score_education_final"
    ]])

if __name__ == "__main__":
    main()
