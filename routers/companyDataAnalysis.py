from fastapi import APIRouter, HTTPException
from typing import Dict
from collections import defaultdict
from bson.objectid import ObjectId
from services.companyDataAnalysis_utils import parsedData, OpenAIService, generate_comprehensive_report_per_job
from services.companyDataAnalysis_utils import AnalysisServiceCompany, generate_word_report_for_company
from pydantic import BaseModel

class companyData(BaseModel):
    companyId: str

router = APIRouter(
    prefix="/company-data-analysis",
    tags=["Company Data Analysis"]
)

@router.post("/job-insights")
async def get_job_insights(request: companyData) -> Dict:
    try:
        assessments = list(parsedData.find({
            "companyId": ObjectId(request.companyId),
            "deleted": {"$ne": True},
            "$or": [
                {"completed": True},
                {"status": "completed"},
                {"assessmentSummary": {"$exists": True, "$ne": ""}}
            ]
        }))

        if not assessments:
            raise HTTPException(status_code=404, detail="No assessments found")

        job_assessments = defaultdict(list)
        for assessment in assessments:
            job_title = assessment.get("jobTitle", "").lower().strip()
            if job_title:
                job_assessments[job_title].append(assessment)

        job_insights = []
        for job_title, job_specific_assessments in job_assessments.items():
            job_insight = OpenAIService.generate_job_insights(job_title, job_specific_assessments)
            job_insights.append(job_insight)

        data = {"job_insights": job_insights}
        generate_comprehensive_report_per_job(data)
        return {"message": "Job insights generated successfully. Check the base folder for the Job wise report."}

    except Exception as e:
        print(f"Error in job insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/company-insights")
async def get_company_analysis(request: companyData) -> Dict:
    try:
        assessments = list(parsedData.find({
            "companyId": ObjectId(request.companyId),
            "deleted": {"$ne": True},
            "$or": [
                {"completed": True},
                {"status": "completed"},
                {"assessmentSummary": {"$exists": True, "$ne": ""}}
            ]
        }))

        if not assessments:
            raise HTTPException(status_code=404, detail="No assessments found")

        skills_analysis = AnalysisServiceCompany.analyze_skills(assessments)
        role_analysis = AnalysisServiceCompany.analyze_roles(assessments)
        
        role_performance_insights = AnalysisServiceCompany.analyze_role_performance(assessments)

        low_fit_roles = [
            role for role in role_performance_insights 
            if role.fit_rate < 10
        ]

        total_candidates = len(assessments)
        
        strengths = []
        weaknesses = []
        fit_scores = []
        
        for assessment in assessments:
            summary = assessment.get('assessmentSummary', '')
            
            strengths.extend([s.strip() for s in summary.split('Strengths:')[1].split('Weaknesses:')[0].split('\n') if '- ' in s and s.strip()] if 'Strengths:' in summary else [])
            weaknesses.extend([w.strip() for w in summary.split('Weaknesses:')[1].split('Fit for the role:')[0].split('\n') if '- ' in w and w.strip()] if 'Weaknesses:' in summary else [])
            
            fit_score = (1 if 'Good fit' in summary else 0 if 'Not a good fit' in summary 
                         else 1 if assessment.get('scoreInPercentage', 0) >= 60 else 0)
            fit_scores.append(fit_score)

        metrics = {
            "performance_metrics": {
                "average_scores": {
                    "overall": round(sum(assessment.get('scoreInPercentage', 0) for assessment in assessments) / total_candidates, 2) if total_candidates > 0 else 0,
                },
                "strengths": list(set(strengths))[:49],
                "weaknesses": list(set(weaknesses))[:49]
            },
            "participation_metrics": {
                "total_candidates": total_candidates,
                "fit_rate": round((sum(fit_scores) / total_candidates) * 100, 2) if fit_scores else 0
            }
        }
        
        recommendations = AnalysisServiceCompany.generate_recommendations(metrics)

        data = {
            "skills_analysis": skills_analysis,
            "role_analysis": role_analysis,
            "role_performance_insights": role_performance_insights,
            "critical_roles": low_fit_roles,
            "metrics": metrics,
            "recommendations": recommendations
        }
        generate_word_report_for_company(data)
        return {"message": "Company insights generated successfully. Check the base folder for the Company wise report."}


    except Exception as e:
        print(f"Error in company analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))