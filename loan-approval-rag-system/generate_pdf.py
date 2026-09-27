from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os

def create_policy_pdf():
    # Ensure the docs directory exists
    os.makedirs('docs', exist_ok=True)
    pdf_path = 'docs/credit_policy_guidelines.pdf'
    
    # Set up the PDF document
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("<b>Nexus Bank: Master Credit Policy & Underwriting Guidelines</b>", styles['Title']))
    story.append(Spacer(1, 12))

    # Document Content
    content = [
        "<b>1. General Loan Eligibility & Credit History</b>",
        "• <b>Credit History Requirement:</b> All applicants must possess a verified Credit History score of 1.0 (representing 'Good Standing') to be considered for primary loan products. Applicants with a Credit History of 0.0 (representing 'Default/Bad Standing') are subject to automatic rejection unless an override is approved by the Chief Risk Officer.",
        "• <b>Age Requirement:</b> The primary applicant must be at least 21 years of age at the time of application and no older than 65 years at the time of loan maturity.",
        "• <b>Residency:</b> Applicants must be legal residents or citizens with verifiable addresses for the past 24 months.",
        "",
        "<b>2. Income & Financial Assessment</b>",
        "• <b>Minimum Income Threshold:</b> The combined total income (Applicant Income + Co-applicant Income) must exceed $10,000 annually.",
        "• <b>Debt-to-Income (DTI) Ratio:</b> The applicant's total monthly debt obligations, including the projected new loan payment, must not exceed 40% of their gross monthly income.",
        "• <b>Employment Stability:</b> Salaried applicants must show continuous employment for the last 12 months. Self-employed applicants must provide audited financial statements for the past 24 months.",
        "",
        "<b>3. Loan Limits & Terms</b>",
        "• <b>Maximum Loan Amount:</b> The maximum allowable loan amount for standard personal and property applications is $500,000 (represented as 500 in the system).",
        "• <b>Income-to-Loan Multiplier:</b> As a strict underwriting rule, the combined Applicant and Co-applicant income must be greater than 10 times the requested Loan Amount. Applications failing this ratio require heavy collateral.",
        "• <b>Loan Term Limits:</b> The standard maximum loan term allowed is 360 days. Short-term loans are available in increments of 120, 180, and 240 days.",
        "",
        "<b>4. Property Area & Collateral Rules</b>",
        "Property location heavily influences the required down payment and risk assessment:",
        "• <b>Urban Properties:</b> Due to higher market volatility, loans secured against Urban properties require a strict minimum 20% down payment from the applicant's own funds.",
        "• <b>Semiurban Properties:</b> Require a 15% minimum down payment.",
        "• <b>Rural Properties:</b> To promote rural development, loans secured against Rural properties require only a 10% down payment, provided the applicant holds a Graduate-level education.",
        "",
        "<b>5. Automatic Rejection Criteria (Red Flags)</b>",
        "An application must be immediately declined by the underwriting team if any of the following are detected:",
        "• The applicant has an active bankruptcy filing within the last 5 years.",
        "• The application contains fraudulent, mismatched, or unverified identity documents.",
        "• The applicant requests a loan term exceeding 360 days.",
        "• The applicant's total calculated income is less than $10,000."
    ]

    # Write text to the PDF
    for line in content:
        if line == "":
            story.append(Spacer(1, 12))
        else:
            story.append(Paragraph(line, styles['Normal']))
            story.append(Spacer(1, 4))
            
    doc.build(story)
    print(f"✅ Success! Document created at: {pdf_path}")

if __name__ == "__main__":
    create_policy_pdf()